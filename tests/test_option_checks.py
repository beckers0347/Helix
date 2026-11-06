"""Behaviour checks for options added by updates.

Live bugs: an update added --top that kept the FIRST N files instead of the N with the most words; another added
--precision that changed only the mean and also leaked internal fields (sum, non_numeric_seen) into --json output.
The fixtures in tests/fixtures/options are small correct tools; BUG=<name> in the environment switches on the bug.
"""
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ogun_core import quality as q  # noqa: E402
from ogun_core import standalone_builder as sb  # noqa: E402

FIX = Path(__file__).resolve().parent / "fixtures" / "options"
CSV_REQ = ("Build me a command-line tool called `colledger`. Give it a CSV file with a header row and it prints, for "
           "each numeric column, the count, minimum, maximum and mean, ignoring empty cells and non-numeric columns, "
           "as an aligned plain-text table with a header separator line. It needs a `--json` option, a `--delimiter` "
           "option (default comma) and a `--column NAME` option to show only that column.")
PREC_REQ = CSV_REQ + " UPDATE: add a --precision N option that sets the number of decimal places for min, max and mean."
TOP_REQ = ("Build me a tool called `t`. For each file print lines, words and characters. It needs --json. "
           "Add a --top N option that shows only the N files with the most words.")


def _with_bug(bug, fn):
    old = os.environ.get("BUG")
    os.environ["BUG"] = bug
    try:
        return fn()
    finally:
        if old is None:
            os.environ.pop("BUG", None)
        else:
            os.environ["BUG"] = old


def test_csv_oracle_accepts_correct_precision_tool():
    td = Path(tempfile.mkdtemp())
    try:
        ok, issues = _with_bug("", lambda: q.OracleRegistry.run_oracles(FIX / "prec_tool.py", PREC_REQ, td))
        assert ok, issues
    finally:
        shutil.rmtree(td, ignore_errors=True)


def test_csv_oracle_rejects_precision_only_on_mean():
    td = Path(tempfile.mkdtemp())
    try:
        ok, issues = _with_bug("meanonly", lambda: q.OracleRegistry.run_oracles(FIX / "prec_tool.py", PREC_REQ, td))
        assert not ok and "decimal place" in issues[0], issues
    finally:
        shutil.rmtree(td, ignore_errors=True)


def test_csv_oracle_rejects_leaked_json_fields():
    td = Path(tempfile.mkdtemp())
    try:
        ok, issues = _with_bug("leak", lambda: q.OracleRegistry.run_oracles(FIX / "prec_tool.py", CSV_REQ, td))
        assert not ok and "exactly" in issues[0] and "sum" in issues[0], issues
    finally:
        shutil.rmtree(td, ignore_errors=True)


def test_top_n_check():
    td = Path(tempfile.mkdtemp())
    try:
        (td / "f1.txt").write_text("a b c d e f g h i\n", encoding="utf-8")
        (td / "f2.txt").write_text("a b\n", encoding="utf-8")
        (td / "f3.txt").write_text("a b c d e f\n", encoding="utf-8")
        files, words = [td / "f2.txt", td / "f1.txt", td / "f3.txt"], {"f1": 9, "f3": 6, "f2": 2}
        ok, err = _with_bug("", lambda: q.check_top_n_option(FIX / "top_tool.py", TOP_REQ, td, files, words))
        assert ok, err
        ok, err = _with_bug("first", lambda: q.check_top_n_option(FIX / "top_tool.py", TOP_REQ, td, files, words))
        assert not ok and "most words" in err, err
        assert q.check_top_n_option(FIX / "top_tool.py", "a tool with --sort", td, files, words) == (True, "")
    finally:
        shutil.rmtree(td, ignore_errors=True)


def test_checked_options_count_as_verified():
    assert "--precision" in q.oracle_verified_flags(PREC_REQ)
    assert "--top" in q.oracle_verified_flags(
        "Build me a command-line tool called `x`. For each file print the number of lines, words and characters. "
        "Add a --top N option that shows only the N files with the most words.")


def test_regression_diffs_flags_changed_unverified_modes_only():
    td = Path(tempfile.mkdtemp())
    try:
        old = td / "old.py"
        new = td / "new.py"
        base = ("import argparse, json\np = argparse.ArgumentParser()\np.add_argument('files', nargs='+')\n"
                "p.add_argument('--json', action='store_true')\np.add_argument('--upper', action='store_true')\n"
                "a = p.parse_args()\nprint(('X' if a.upper else 'x'){extra})\n")
        old.write_text(base.format(extra=""), encoding="utf-8")
        new.write_text(base.format(extra=" + ('!' if a.upper else '')"), encoding="utf-8")
        req = "Build me a tool that reads text files and prints them. It needs --json and --upper options."
        assert sb.regression_diffs(old, new, req, td) == ["--upper"]
        assert sb.regression_diffs(old, new, req, td, oracle_checked={"--upper"}) == []
    finally:
        shutil.rmtree(td, ignore_errors=True)


def test_failed_update_also_restores_original_request():
    td = Path(tempfile.mkdtemp())
    orig_ws = sb.WORKSPACE_DIR
    try:
        sb.WORKSPACE_DIR = td
        d = td / "t1"
        (d / ".build_log").mkdir(parents=True)
        (d / "t1.py").write_text("import argparse\nargparse.ArgumentParser().parse_args()\n", encoding="utf-8")
        (d / ".build_log" / "request.txt").write_text("ORIGINAL", encoding="utf-8")

        def failing(engine, request, **kw):
            (d / ".build_log" / "request.txt").write_text(request, encoding="utf-8")
            return False, "TESTS FAILING", {"status": "failed"}

        sb.update_standalone_project(None, "t1", "Add a --top N option", build_fn=failing)
        assert (d / ".build_log" / "request.txt").read_text(encoding="utf-8") == "ORIGINAL"
    finally:
        sb.WORKSPACE_DIR = orig_ws
        shutil.rmtree(td, ignore_errors=True)


def test_required_fixes_are_emphasised_at_top_and_end():
    """Live bug: the oracle's exact fix for --top was buried mid-prompt and the model repeated the bug."""
    fb = ("Smoke failed\n- Oracle line-word-count defect: --top 1 wrong. "
          "HOW TO FIX: sort by words, largest first, THEN keep N.")
    out = q.emphasize_required_fixes("PREFIX\nSTAGE: FIX\nbody\nmore", fb)
    lines = out.splitlines()
    assert lines[2].startswith("REQUIRED CHANGES"), lines[:4]
    assert out.rstrip().endswith("THEN keep N."), out[-80:]
    assert q.emphasize_required_fixes("STAGE: X\nbody", "no concrete hint") == "STAGE: X\nbody"


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as e:
                fails += 1
                print(f"FAIL {name}: {type(e).__name__}: {e}")
    sys.exit(1 if fails else 0)
