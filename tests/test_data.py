import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from acinotes.data import load_split, parse_note, subset_of  # noqa: E402

NOTE = """CHIEF COMPLAINT

Knee pain.

HISTORY OF PRESENT ILLNESS

Martha is a 50-year-old female.

PHYSICAL EXAM

Neck: no pain.

RESULTS

X-ray normal.

ASSESSMENT AND PLAN

Take ibuprofen."""


class DataTest(unittest.TestCase):
    def test_groups(self):
        secs = parse_note(NOTE)
        self.assertEqual([s["group"] for s in secs],
                         ["subjective", "subjective", "objective_exam", "objective_results",
                          "assessment_and_plan"])
        self.assertEqual(secs[0]["text"], "Knee pain.")

    def test_pre_header_text_kept(self):
        secs = parse_note("Texto suelto\n\nPLAN\n\nReposo")
        self.assertEqual(secs[0]["group"], "OTHER")
        self.assertEqual(secs[1]["header"], "PLAN")

    def test_inline_word_is_not_header(self):
        secs = parse_note("PLAN\n\nThe plan is to rest.")
        self.assertEqual(len(secs), 1)

    def test_subset(self):
        self.assertEqual(subset_of("D2N071-virtassist"), "virtassist")

    def test_load_split_sizes(self):
        self.assertEqual({k: len(load_split(k)) for k in ["train", "valid", "test1"]},
                         {"train": 67, "valid": 20, "test1": 40})
        ex = load_split("valid")[0]
        self.assertEqual(set(ex), {"file", "src", "tgt"})


if __name__ == "__main__":
    unittest.main()
