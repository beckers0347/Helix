import unittest
from _scaffold import ToolTestCase


class TestColreport(ToolTestCase):
    def test_help(self):
        res = self.run_tool("--help")
        self.assertOk(res)
        self.assertIn("colreport", res.stdout)

    def test_missing_file(self):
        res = self.run_tool("missing_file.csv")
        self.assertFails(res)

    def test_json_stats(self):
        self.write_file("d.csv", "name,a,b\nx,1,10.5\ny,,-20.5\nz,3,30\n")
        res = self.run_tool("d.csv", "--json")
        self.assertOk(res)
        data = self.assertJsonOutput(res)
        self.assertEqual(sorted(data), ["a", "b"])
        self.assertEqual(data["a"]["count"], 2)

    def test_column_json(self):
        self.write_file("d.csv", "name,a,b\nx,1,10.5\ny,,-20.5\nz,3,30\n")
        res = self.run_tool("d.csv", "--column", "b", "--json")
        self.assertOk(res)
        data = self.assertJsonOutput(res)
        self.assertEqual(list(data), ["b"])

    def test_delimiter_json(self):
        self.write_file("s.csv", "name;a\nx;2\ny;4\n")
        res = self.run_tool("s.csv", "--delimiter", ";", "--json")
        self.assertOk(res)
        data = self.assertJsonOutput(res)
        self.assertAlmostEqual(data["a"]["mean"], 3.0)

    def test_default_table(self):
        self.write_file("d.csv", "name,a,b\nx,1,10.5\ny,,-20.5\nz,3,30\n")
        res = self.run_tool("d.csv")
        self.assertOk(res)
        self.assertIn("a", res.stdout)
        self.assertNotIn("name ", res.stdout)


if __name__ == "__main__":
    unittest.main()
