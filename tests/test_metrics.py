import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from acinotes import naming  # noqa: E402
from acinotes.evaluate import resolve_predictions, run_evaluation  # noqa: E402
from acinotes import metrics as metrics_pkg  # noqa: E402
from acinotes.metrics import bleurt, entities  # noqa: E402
from acinotes.sections import split_groups  # noqa: E402

NOTE = "CHIEF COMPLAINT\n\nCough.\n\nPHYSICAL EXAM\n\nLungs clear.\n\nASSESSMENT AND PLAN\n\nRest."
STEM = "valid__zeroshot__qwen3-4b-instruct-2507__nothink-4bit-bf16__20261119T1530Z"


class InterfaceTest(unittest.TestCase):
    def test_every_metric_has_full_interface(self):
        for metric_id in metrics_pkg.METRIC_MODULES:
            mod = metrics_pkg.load(metric_id)
            for attr in ("metric_name", "run", "headline"):
                self.assertTrue(callable(getattr(mod, attr, None)), f"{metric_id} no té {attr}")
            self.assertRegex(mod.metric_name({}), r"^[a-z0-9.\-]+-v\d+$")

    def test_names_do_not_contain_double_underscore(self):
        for metric_id in metrics_pkg.METRIC_MODULES:
            self.assertNotIn("__", metrics_pkg.load(metric_id).metric_name({}))


class SectionsTest(unittest.TestCase):
    def test_markdown_tolerant(self):
        s = split_groups("**CHIEF COMPLAINT**\nCough\n\n## PLAN:\nRest")
        self.assertEqual(s, {"subjective": "Cough", "assessment_and_plan": "Rest"})


class EntitiesPureTest(unittest.TestCase):
    def test_normalize_numbers(self):
        self.assertEqual(entities.normalize("three out of six"), "3 6")
        self.assertEqual(entities.normalize("120 over 80"), "120 80")

    def test_value_conflict_is_not_a_match(self):
        self.assertFalse(entities.is_match("ibuprofen 10 mg", {"ibuprofen 20 mg"}, 0.85))
        self.assertTrue(entities.is_match("ibuprofen 600 mg", {"ibuprofen 600 mg 6 hour"}, 0.85))

    def test_compare_categories(self):
        ref = {"cough": "sign_symptom", "asthma": "disease_disorder", "fever": "sign_symptom"}
        pred = {"cough": "sign_symptom", "headache": "sign_symptom", "diabetes": "disease_disorder"}
        src_norm = entities.normalize("the patient has a cough and fever and a headache")
        r = entities.compare(ref, pred, src_norm, {"cough", "fever", "headache"}, 0.85)
        self.assertEqual({t for t, _ in r["omitted"]}, {"asthma", "fever"})
        self.assertEqual(r["omitted_supported"], ["fever"])               # asthma no és a la transcripció
        self.assertEqual([t for t, _ in r["extra_supported"]], ["headache"])
        self.assertEqual([t for t, _ in r["hallucination_candidates"]], ["diabetes"])


class BleurtLogicTest(unittest.TestCase):
    def test_sections_and_missing(self):
        rows = [{"file": "A", "ref": NOTE, "pred": "CHIEF COMPLAINT\n\nCough.\n\nPLAN\n\nRest."}]
        fake = lambda refs, cands: ([0.5] * len(refs), 0)
        res = bleurt.run_with_scorer(rows, fake)
        secs = res["summary"]["sections"]
        self.assertEqual(secs["subjective"]["n"], 1)
        self.assertEqual(secs["objective_exam"]["missing_in_pred"], 1)   # la ref té exploració i la predicció no
        self.assertEqual(secs["assessment_and_plan"]["n"], 1)

    def test_metric_name(self):
        self.assertEqual(bleurt.metric_name({}), "bleurt-elron-base-512-v1")


class EvaluateRunnerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.pdir, self.mdir = self.tmp / "predictions", self.tmp / "metrics"
        self.pdir.mkdir()
        rows = [{"file": "D2N001-aci", "pred": "CHIEF COMPLAINT\n\nCough.", "ref": NOTE, "hit_limit": False},
                {"file": "D2N002-aci", "pred": "", "ref": NOTE, "hit_limit": True}]
        (self.pdir / f"{STEM}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        (self.pdir / f"{STEM}.meta.json").write_text("{}", encoding="utf-8")

    def test_resolve(self):
        by_all = resolve_predictions(["all"], self.pdir)
        self.assertEqual([p.name for p in by_all], [f"{STEM}.jsonl"])        # el .meta.json no compta
        self.assertEqual(resolve_predictions([STEM], self.pdir), by_all)
        self.assertEqual(resolve_predictions(["*nothink*"], self.pdir), by_all)
        with self.assertRaises(FileNotFoundError):
            resolve_predictions(["no-existeix"], self.pdir)

    def test_run_writes_file_with_same_name_and_skips_existing(self):
        preds = resolve_predictions(["all"], self.pdir)
        out = run_evaluation(preds, ["length"], {}, self.mdir, repo_dir=self.tmp)
        self.assertEqual([p.name for p in out], [f"{STEM}__length-v1.json"])
        pid, metric = naming.parse_metric_filename(out[0].name)
        self.assertEqual((pid.stem, metric), (STEM, "length-v1"))
        data = json.loads(out[0].read_text(encoding="utf-8"))
        self.assertEqual(data["prediction"], STEM)
        self.assertEqual(data["summary"]["n_empty_pred"], 1)
        self.assertEqual(data["summary"]["n_hit_limit"], 1)
        self.assertEqual(len(data["per_note"]), 2)
        self.assertEqual(run_evaluation(preds, ["length"], {}, self.mdir, repo_dir=self.tmp), [])   # no sobreescriu
        self.assertEqual(len(run_evaluation(preds, ["length"], {}, self.mdir, overwrite=True, repo_dir=self.tmp)), 1)


if __name__ == "__main__":
    unittest.main()
