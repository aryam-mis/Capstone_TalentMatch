"""Synthetic fixtures only: these are not TalentMatch evaluation results."""

import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location("prepare_judgement", Path(__file__).resolve().parents[1] / "src/prepare_judgement.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class PrepareJudgementTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cvs = self.root / "cvs.csv"
        self.jobs = self.root / "jobs.tsv"
        self.pop1, self.pop0 = self.root / "pop1.csv", self.root / "pop0.csv"
        self.out = self.root / "output"
        self.write(self.cvs, [{"cv_id": "SYNTHETIC", "state": "CA", "profile_text": "Synthetic profile"}])
        self.recs = [dict(cv_id="SYNTHETIC", state="CA", rank=str(i), JobID=str(i), WindowID="1",
                          score_rrf="DO_NOT_EXPOSE", grader1="2", job_text="TRUNCATED") for i in range(1, 11)]
        # Same JobID across two windows must remain distinct grading items.
        self.other = [dict(r) for r in self.recs]
        self.other[0]["WindowID"] = "2"
        self.job_rows = [dict(JobID=str(i), WindowID="1", Title="Synthetic job", Description="<p>Full duties</p>",
                              Requirements="Synthetic requirements", State="CA") for i in range(1, 11)]
        self.job_rows.append(dict(self.job_rows[0], WindowID="2"))
        self.write(self.pop1, self.recs)
        self.write(self.pop0, self.other)
        self.write(self.jobs, self.job_rows)

    def write(self, path, rows):
        with path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t" if path.suffix == ".tsv" else ",")
            writer.writeheader()
            writer.writerows(rows)

    def run_prepare(self):
        return module.prepare(self.pop1, self.pop0, self.cvs, self.jobs, self.out, expected_cvs=1)

    def test_blinding_deduplication_and_mapping(self):
        result = self.run_prepare()
        self.assertEqual(result["unique_grading_items"], 11)
        self.assertEqual(result["shared_items"], 9)
        forms = []
        for reviewer in (1, 2):
            with (self.out / f"reviewer{reviewer}.csv").open(encoding="utf-8-sig", newline="") as f:
                rows = list(csv.DictReader(f))
            forms.append(rows)
            self.assertEqual(len(rows), 11)
            self.assertEqual(set(rows[0]), set(module.REVIEW_FIELDS))
            self.assertTrue(all(r["score"] == "" and r["notes"] == "" for r in rows))
            self.assertTrue(all(r["job_description"] == "Full duties" for r in rows))
            self.assertNotIn("DO_NOT_EXPOSE", str(rows))
            self.assertNotIn("TRUNCATED", str(rows))
        self.assertEqual({r["item_id"] for r in forms[0]}, {r["item_id"] for r in forms[1]})
        self.assertNotEqual([r["item_id"] for r in forms[0]], [r["item_id"] for r in forms[1]])
        with (self.out / "internal/recommendation_item_map.csv").open(encoding="utf-8-sig", newline="") as f:
            mapping = list(csv.DictReader(f))
        self.assertEqual(len(mapping), 20)
        shared = [r["item_id"] for r in mapping if r["JobID"] == "2"]
        self.assertEqual(shared[0], shared[1])
        different_windows = [r["item_id"] for r in mapping if r["JobID"] == "1"]
        self.assertNotEqual(*different_windows)

    def test_missing_recommendation_fails_without_output(self):
        self.write(self.pop0, self.other[:-1])
        with self.assertRaisesRegex(ValueError, "expected ranks"):
            self.run_prepare()
        self.assertFalse(self.out.exists())

    def test_duplicate_pair_and_duplicate_rank_fail(self):
        duplicate = [dict(r) for r in self.recs]
        duplicate[-1] = dict(duplicate[0])
        self.write(self.pop1, duplicate)
        with self.assertRaisesRegex(ValueError, "duplicate CV-job"):
            self.run_prepare()
        self.assertFalse(self.out.exists())

    def test_missing_or_duplicate_job_fails(self):
        self.write(self.jobs, self.job_rows[:-1])
        with self.assertRaisesRegex(ValueError, "missing 1 requested"):
            self.run_prepare()
        self.write(self.jobs, self.job_rows + [self.job_rows[0]])
        with self.assertRaisesRegex(ValueError, "duplicate requested key"):
            self.run_prepare()
        self.assertFalse(self.out.exists())

    def test_refuses_overwriting(self):
        self.run_prepare()
        original = (self.out / "reviewer1.csv").read_bytes()
        with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
            self.run_prepare()
        self.assertEqual(original, (self.out / "reviewer1.csv").read_bytes())

    def test_unknown_cv_and_missing_cv_and_state_mismatch(self):
        changed = [dict(r) for r in self.recs]
        changed[0]["cv_id"] = "UNKNOWN"
        self.write(self.pop1, changed)
        with self.assertRaisesRegex(ValueError, "unknown cv_id"):
            self.run_prepare()
        self.write(self.pop1, self.recs)
        self.write(self.cvs, [
            {"cv_id": "SYNTHETIC", "state": "CA", "profile_text": "Synthetic"},
            {"cv_id": "MISSING", "state": "CA", "profile_text": "Synthetic"},
        ])
        with self.assertRaisesRegex(ValueError, "MISSING: expected ranks"):
            module.prepare(self.pop1, self.pop0, self.cvs, self.jobs, self.out, expected_cvs=2)
        self.write(self.cvs, [{"cv_id": "SYNTHETIC", "state": "TX", "profile_text": "Synthetic"}])
        with self.assertRaisesRegex(ValueError, "state differs"):
            self.run_prepare()


if __name__ == "__main__":
    unittest.main()
