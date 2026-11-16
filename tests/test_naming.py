import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from acinotes import naming  # noqa: E402


class NamingTest(unittest.TestCase):
    def setUp(self):
        self.pid = naming.PredictionId.new("valid", "ZeroShot", "Qwen3-4B-Instruct-2507",
                                           "4bit bf16", stamp="20261116T1702Z")

    def test_stem(self):
        self.assertEqual(self.pid.stem,
                         "valid__zeroshot__qwen3-4b-instruct-2507__4bit-bf16__20261116T1702Z")

    def test_roundtrip(self):
        self.assertEqual(naming.parse_stem(self.pid.stem), self.pid)

    def test_metric_links_to_prediction(self):
        p = naming.metric_path(self.pid, "entities-v4")
        pid, metric = naming.parse_metric_filename(p.name)
        self.assertEqual(pid, self.pid)
        self.assertEqual(metric, "entities-v4")

    def test_bad_split(self):
        with self.assertRaises(ValueError):
            naming.PredictionId.new("dev", "x", "y", stamp="20261116T1702Z")

    def test_bad_stamp(self):
        with self.assertRaises(ValueError):
            naming.PredictionId("valid", "a", "b", "c", "hoy")

    def test_no_double_underscore_in_fields(self):
        self.assertNotIn("__", naming.slug("a__b"))


if __name__ == "__main__":
    unittest.main()
