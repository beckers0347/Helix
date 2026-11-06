"""Build / upgrade fixes, each from a live failure:

1. The GPU layer count was picked once at startup, so a program opened later (Unreal Editor, 5 GB of VRAM) left the
   model with more layers than fit and replies stalled for many minutes.
2. Requests to CHANGE an existing workspace tool or plugin were not routed to an update: a tool update edited only the
   tests and reported success although the requested option did not exist.
3. A generated test asserted mean(20, 40) == 20.333333; the verified tool printed 30.0 and the build failed.
"""
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ogun_core import autotune, config, hardware  # noqa: E402
from ogun_core import quality  # noqa: E402
from ogun_core import standalone_builder as sb  # noqa: E402


def _hw(free_gb, total_gb=11.94):
    gpu = hardware.GpuInfo("nvidia", "NVIDIA GeForce RTX 5070", total_gb, free_gb, False, "610.88", "12.0", 0)
    return hardware.HardwareInfo("windows", "amd64", "AMD Ryzen 9 5900X", 12, 24,
                                 {"avx": True, "avx2": True, "avx512": False, "neon": False},
                                 63.9, 40.0, [gpu], gpu, 0.1, [])


def _oll(resident=0.0):
    return {"reachable": True, "installed": [], "resident": [], "resident_vram_gb": resident, "resident_ram_gb": 0.0}


def _expected(free_gb):
    return hardware.auto_configure(_hw(free_gb), overrides={"configured_model": "qwen2.5-coder:14b", "ctx": 8192}).num_gpu


class VramRefreshTests(unittest.TestCase):
    def setUp(self):
        self.saved_active = dict(config._ACTIVE)
        self.saved_refresh = dict(autotune._REFRESH)
        autotune._REFRESH.update(overrides={}, last=0.0, last_unload=0.0)

    def tearDown(self):
        config._ACTIVE.clear()
        config._ACTIVE.update(self.saved_active)
        autotune._REFRESH.clear()
        autotune._REFRESH.update(self.saved_refresh)

    def _start_auto(self, num_gpu):
        config.apply_hardware_profile({"num_gpu": num_gpu, "num_ctx": 8192, "num_thread": 12,
                                       "model_tag": "qwen2.5-coder:14b"})

    def test_program_opened_after_startup_lowers_layers_ctx_fixed(self):
        self._start_auto(43)
        note = autotune.refresh("http://x", "qwen2.5-coder:14b", force=True,
                                probe=lambda: _hw(6.7), query=lambda: _oll(), unload=lambda: None)
        self.assertTrue(note and "43" in note, note)
        self.assertEqual(config.get_num_gpu(), _expected(6.7))
        self.assertLess(config.get_num_gpu(), 40)
        self.assertEqual(config.llm_options()["num_ctx"], 8192)

    def test_small_change_and_rate_limit(self):
        self._start_auto(_expected(6.7))
        self.assertIsNone(autotune.refresh("h", "m", force=True, probe=lambda: _hw(6.8), query=lambda: _oll()))
        calls = []
        self.assertIsNone(autotune.refresh("h", "m", now=autotune._REFRESH["last"] + 5,
                                           probe=lambda: calls.append(1) or _hw(2.0), query=lambda: _oll()))
        self.assertEqual(calls, [])

    def test_resident_model_counts_as_available(self):
        self._start_auto(_expected(10.9))
        self.assertIsNone(autotune.refresh("h", "qwen2.5-coder:14b", force=True,
                                           probe=lambda: _hw(1.2), query=lambda: _oll(9.7)))

    def test_full_gpu_unloads_once_to_measure(self):
        self._start_auto(_expected(10.9))
        state = {"loaded": True}
        unloaded = []

        def unload():
            unloaded.append(1)
            state["loaded"] = False

        probe = lambda: _hw(0.3) if state["loaded"] else _hw(5.4)
        query = lambda: _oll(9.7 if state["loaded"] else 0.0)
        note = autotune.refresh("h", "qwen2.5-coder:14b", force=True, now=1000.0, probe=probe, query=query, unload=unload)
        self.assertTrue(note)
        self.assertEqual(unloaded, [1])
        self.assertEqual(config.get_num_gpu(), _expected(5.4))
        state["loaded"] = True
        autotune.refresh("h", "qwen2.5-coder:14b", force=True, now=1100.0, probe=probe, query=query, unload=unload)
        self.assertEqual(unloaded, [1], "spill unload is rate limited")

    def test_manual_profile_and_forced_layers_respected(self):
        self._start_auto(43)
        autotune._REFRESH["overrides"] = {"gpu_layers": 43}
        self.assertIsNone(autotune.refresh("h", "m", force=True, probe=lambda: _hw(3.0), query=lambda: _oll()))
        autotune._REFRESH["overrides"] = {}
        config.set_vram_profile("shared")
        self.assertIsNone(autotune.refresh("h", "m", force=True, probe=lambda: _hw(3.0), query=lambda: _oll()))

    def test_builder_refreshes_and_never_swaps_model(self):
        seen = {}
        orig_refresh, orig_evict = autotune.refresh, sb.evict_other_loaded_models
        try:
            autotune.refresh = lambda host, model, force=False, **k: seen.setdefault("force", force) and None
            sb.evict_other_loaded_models = lambda host, model: []
            model, opts = sb.select_optimal_builder_model("http://127.0.0.1:1", "qwen2.5-coder:14b")
        finally:
            autotune.refresh, sb.evict_other_loaded_models = orig_refresh, orig_evict
        self.assertTrue(seen.get("force"))
        self.assertEqual((model, opts), ("qwen2.5-coder:14b", {}))


class ChangeRoutingTests(unittest.TestCase):
    def setUp(self):
        self.td = Path(tempfile.mkdtemp())
        self.ws = self.td / "workspace"
        self.pl = self.td / "custom_tools"
        (self.ws / "colledger").mkdir(parents=True)
        (self.ws / "colledger" / "colledger.py").write_text("print(1)\n", encoding="utf-8")
        (self.ws / "notes").mkdir()
        self.pl.mkdir()
        (self.pl / "file_crc32_checksum.py").write_text("# plugin\n", encoding="utf-8")

    def tearDown(self):
        shutil.rmtree(self.td, ignore_errors=True)

    def detect(self, text):
        return sb.detect_change_target(text, workspace_dir=self.ws, plugins_dir=self.pl)

    def test_detects_updates(self):
        self.assertEqual(self.detect("Update the existing colledger tool in workspace/colledger so it has a --precision N option"),
                         ("workspace_tool", "colledger"))
        self.assertEqual(self.detect("Update the file_crc32_checksum plugin so it also returns the file size"),
                         ("plugin", "file_crc32_checksum"))
        self.assertEqual(self.detect("upgrade the file crc32 checksum plugin"), ("plugin", "file_crc32_checksum"))

    def test_not_change_requests(self):
        self.assertIsNone(self.detect("What does colledger do?"))
        self.assertIsNone(self.detect("Update the weather plugin"))
        self.assertIsNone(self.detect("Build me a new command-line tool called colledger2 that adds a total"))
        self.assertIsNone(self.detect("Update my notes folder"))

    def _make_tool(self, name, flags):
        d = self.ws / name
        (d / ".build_log" / "attempt_1").mkdir(parents=True, exist_ok=True)
        opts = "".join(f"p.add_argument('{f}')\n" for f in flags)
        (d / f"{name}.py").write_text("import argparse\np = argparse.ArgumentParser()\n" + opts + "p.parse_args()\n",
                                      encoding="utf-8")
        (d / f"test_{name}.py").write_text("# old tests\n", encoding="utf-8")
        (d / ".build_log" / "attempt_1" / "tool_prompt.txt").write_text(
            f'Write a clean, robust, standalone command-line utility `{name}.py` for this request:\n'
            f'"""Build me a command-line tool called `{name}`. It needs a `--json` option."""\n', encoding="utf-8")
        return d

    def test_update_rebuilds_from_original_request(self):
        orig = sb.WORKSPACE_DIR
        try:
            sb.WORKSPACE_DIR = self.ws
            d = self._make_tool("colledger", ["--json"])
            seen = {}

            def fake_build(engine, request, **kw):
                seen["request"] = request
                self.assertFalse((d / "test_colledger.py").exists(), "old tests set aside")
                (d / "colledger.py").write_text("import argparse\np = argparse.ArgumentParser()\np.add_argument('--json')\n"
                                                "p.add_argument('--precision')\np.parse_args()\n", encoding="utf-8")
                return True, "PASSED", {"status": "success"}

            ok, msg, _ = sb.update_standalone_project(None, "colledger", "Add a --precision N option", build_fn=fake_build)
            self.assertTrue(ok, msg)
            self.assertIn("called `colledger`", seen["request"])
            self.assertIn("--precision N", seen["request"])
        finally:
            sb.WORKSPACE_DIR = orig

    def test_update_without_the_option_fails(self):
        orig = sb.WORKSPACE_DIR
        try:
            sb.WORKSPACE_DIR = self.ws
            self._make_tool("colledger", ["--json"])
            ok, msg, details = sb.update_standalone_project(
                None, "colledger", "Add a --precision N option",
                build_fn=lambda e, r, **k: (True, "The tests have passed successfully.", {"status": "success"}))
            self.assertFalse(ok)
            self.assertIn("--precision", msg)
            self.assertIn("NOT done", msg)
        finally:
            sb.WORKSPACE_DIR = orig


class VerifiedRepairTests(unittest.TestCase):
    TEST_CODE = (
        "import unittest\nfrom _scaffold import ToolTestCase\n\n\nclass T(ToolTestCase):\n"
        "    def test_json(self):\n"
        "        data = self.assertJsonOutput(self.run_tool('s.csv', '--json'))\n"
        "        self.assertAlmostEqual(data['col_a']['mean'], 20.0, delta=0.01)\n"
        "        self.assertAlmostEqual(data['col_b']['mean'], 20.333333, delta=0.01)\n")
    OUTPUT = "AssertionError: 30.0 != 20.333333 within 0.01 delta (9.666667 difference)\n"

    def test_strict_without_verification(self):
        changed, code = quality.repair_observed_output_test_defects(self.TEST_CODE, self.OUTPUT)
        self.assertFalse(changed)

    def test_verified_tool_corrects_wrong_mean(self):
        changed, code = quality.repair_observed_output_test_defects(self.TEST_CODE, self.OUTPUT, tool_verified=True)
        self.assertTrue(changed)
        self.assertNotIn("20.333333", code)
        self.assertIn("30.0", code)
        self.assertIn("20.0", code)

    def test_applicable_oracles(self):
        req = ("Build me a command-line tool called `colledger`. Give it a CSV file with a header row and it prints, "
               "for each numeric column, the count, minimum, maximum and mean.")
        self.assertEqual(quality.OracleRegistry.applicable_oracles(req), ["csv_column_stats"])
        self.assertEqual(quality.OracleRegistry.applicable_oracles("a tool that prints the CRC32 of a file"), [])


class LiveFindingTests(unittest.TestCase):
    FAILING = "import unittest\nclass T(unittest.TestCase):\n    def test_x(self):\n        self.assertEqual(1, 2)\n"
    PASSING = "import unittest\nclass T(unittest.TestCase):\n    def test_x(self):\n        self.assertEqual(2, 2)\n"
    EMPTY = "import unittest\nclass T(unittest.TestCase):\n    pass\n"

    def _run(self, code):
        import subprocess
        from ogun_core.plugin_manager import plugin_test_command
        td = Path(tempfile.mkdtemp())
        try:
            tf = td / "test_probe.py"
            tf.write_text(code, encoding="utf-8")
            return subprocess.run(plugin_test_command(tf), capture_output=True, text=True, timeout=60).returncode
        finally:
            shutil.rmtree(td, ignore_errors=True)

    def test_unittest_plugin_test_without_main_really_runs(self):
        self.assertNotEqual(self._run(self.FAILING), 0)
        self.assertEqual(self._run(self.PASSING), 0)
        self.assertNotEqual(self._run(self.EMPTY), 0, "zero tests is a failure")

    def test_failed_update_restores_tool_and_tests(self):
        td = Path(tempfile.mkdtemp())
        orig = sb.WORKSPACE_DIR
        try:
            sb.WORKSPACE_DIR = td
            d = td / "colledger"
            (d / ".build_log").mkdir(parents=True)
            src = "import argparse\np = argparse.ArgumentParser()\np.add_argument('--json')\np.parse_args()\n"
            (d / "colledger.py").write_text(src, encoding="utf-8")
            (d / "test_colledger.py").write_text("# original tests\n", encoding="utf-8")

            def failing(engine, request, **kw):
                (d / "colledger.py").write_text("broken = (\n", encoding="utf-8")
                return False, "Tests rejected", {"status": "failed"}

            ok, msg, _ = sb.update_standalone_project(None, "colledger", "Add a --precision N option", build_fn=failing)
            self.assertFalse(ok)
            self.assertEqual((d / "colledger.py").read_text(encoding="utf-8"), src)
            self.assertEqual((d / "test_colledger.py").read_text(encoding="utf-8"), "# original tests\n")
            self.assertIn("Restored the previous", msg)
        finally:
            sb.WORKSPACE_DIR = orig
            shutil.rmtree(td, ignore_errors=True)


class PluginUpdateRollbackTests(unittest.TestCase):
    """Live bug: a failed update of the working file_crc32_checksum plugin overwrote it and moved it to quarantine."""

    GOOD = (
        "from __future__ import annotations\n"
        "from typing import Any, Dict\n"
        "from ogun_core.base_tool import BaseTool, ToolResult\n\n"
        "class EchoProbe(BaseTool):\n"
        "    name: str = 'echo_probe'\n"
        "    version: str = '{v}'\n"
        "    description: str = 'Echo a value.'\n"
        "    parameters_schema: Dict[str, Any] = {{'type': 'object', 'properties': {{'value': {{'type': 'string'}}}}, 'required': ['value']}}\n\n"
        "    def execute(self, **kwargs: Any) -> ToolResult:\n"
        "        return ToolResult.success(output=kwargs.get('value'))\n"
    )
    PASSING_TEST = "print('PASS')\n"
    FAILING_TEST = "import sys\nsys.stderr.write('expected value mismatch')\nsys.exit(1)\n"

    def test_failed_update_keeps_the_working_plugin(self):
        from ogun_core.plugin_manager import PluginManager
        td = Path(tempfile.mkdtemp())
        try:
            tools, tests_dir, quar = td / "plugins" / "custom_tools", td / "plugins" / "tests", td / "plugins" / "quarantine"
            for d in (tools, tests_dir):
                d.mkdir(parents=True, exist_ok=True)
            pm = PluginManager(custom_tools_dir=tools, tests_dir=tests_dir, quarantine_dir=quar,
                               registry_path=td / "plugins" / "registry.json", base_dir=td)
            ok = pm.create_or_update_plugin("echo_probe", self.GOOD.format(v="1.0.0"), self.PASSING_TEST)
            self.assertEqual(ok.get("status"), "success", ok)
            bad = pm.create_or_update_plugin("echo_probe", self.GOOD.format(v="1.0.1"), self.FAILING_TEST)
            self.assertEqual(bad.get("status"), "error")
            self.assertIn("1.0.0", (tools / "echo_probe.py").read_text(encoding="utf-8"), "working version kept")
            self.assertEqual((tests_dir / "test_echo_probe.py").read_text(encoding="utf-8"), self.PASSING_TEST)
            self.assertIn("1.0.1", (quar / "echo_probe.py").read_text(encoding="utf-8"), "failed candidate quarantined")
        finally:
            shutil.rmtree(td, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
