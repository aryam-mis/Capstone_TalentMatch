"""Synthetic data only; none of these ratings are real project results."""

import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location("evaluate_judgement", Path(__file__).resolve().parents[1] / "src/evaluate_judgement.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class EvaluateJudgementTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.paths = [self.root / n for n in ("reviewer1.csv", "reviewer2.csv", "mapping.csv")]
        self.out = self.root / "evaluation"
        self.mapping = [dict(configuration=conf, cv_id="SYNTHETIC", JobID=str(i), WindowID="1", rank=str(rank), item_id=f"I{i}")
                        for conf, start in zip(module.CONFIGS, (1, 3)) for rank, i in enumerate(range(start, start + 10), 1)]
        self.first = [dict(item_id=f"I{i}", score=str((i - 1) % 3)) for i in range(1, 13)]
        self.second = [dict(item_id=f"I{i}", score=str(i % 3)) for i in range(12, 0, -1)]
        self.save()

    def write(self, path, rows):
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def save(self):
        for path, rows in zip(self.paths, (self.first, self.second, self.mapping)):
            self.write(path, rows)

    def run_eval(self, pooled=False):
        return module.evaluate(*self.paths, self.out, expected_cvs=1, include_pooled=pooled)

    def assert_rejected(self, pattern):
        self.save()
        with self.assertRaisesRegex(ValueError, pattern):
            self.run_eval()
        self.assertFalse(self.out.exists())

    def test_shared_items_disagreement_and_exact_metrics(self):
        result = self.run_eval(pooled=True)
        self.assertEqual(result["agreement"]["unique_items"], 12)  # not 20 occurrences
        self.assertEqual(result["agreement"]["percentage_agreement"], 0)
        self.assertAlmostEqual(result["agreement"]["cohens_kappa"], -0.5)
        lookup = {(r["configuration"], r["reviewer"], r["k"]): r for r in result["metrics"]}
        self.assertAlmostEqual(lookup[module.CONFIGS[0], "reviewer1", 10]["mean_relevance"], .9)
        self.assertAlmostEqual(lookup[module.CONFIGS[1], "reviewer1", 10]["mean_relevance"], 1.1)
        self.assertAlmostEqual(lookup[module.CONFIGS[1], "reviewer1", 10]["score2_proportion"], .4)
        self.assertAlmostEqual(lookup[module.CONFIGS[0], "reviewer2", 10]["mean_relevance"], 1)
        for conf in module.CONFIGS:
            for reviewer in ("reviewer1", "reviewer2"):
                self.assertAlmostEqual(lookup[conf, reviewer, 3]["mean_relevance"], 1)
                self.assertAlmostEqual(lookup[conf, reviewer, 3]["score2_proportion"], 1/3)
        pooled = lookup[module.CONFIGS[0], "equal_reviewer_average_not_consensus", 10]
        self.assertAlmostEqual(pooled["mean_relevance"], .95)
        comparison = next(r for r in result["comparison"] if r["reviewer"] == "reviewer1" and r["k"] == 10)
        self.assertAlmostEqual(comparison["score2_percentage_point_delta"], -10)
        with (self.out / "recommendation_ratings.csv").open(encoding="utf-8-sig") as f:
            self.assertEqual(len(list(csv.DictReader(f))), 20)
        self.assertTrue((self.out / "SUMMARY.md").exists())

    def test_perfect_and_undefined_kappa(self):
        self.second = list(reversed([dict(r) for r in self.first]))
        self.save()
        result = self.run_eval()
        self.assertEqual(result["agreement"]["percentage_agreement"], 100)
        self.assertEqual(result["agreement"]["cohens_kappa"], 1)
        self.assertEqual(len(result["metrics"]), 8)  # separate reviewers only
        undefined, _ = module.agreement_metrics({"a": 2, "b": 2}, {"a": 2, "b": 2})
        self.assertIsNone(undefined["cohens_kappa"])

    def test_invalid_blank_and_missing_ratings(self):
        for value in ("", "3", "-1", "1.5", "NaN"):
            with self.subTest(value=value):
                self.first[0]["score"] = value
                self.assert_rejected("must be 0, 1 or 2")
        self.first = self.first[1:]
        self.assert_rejected("missing 1 items")

    def test_duplicate_and_unknown_reviewer_ids(self):
        self.first.append(dict(self.first[0]))
        self.assert_rejected("duplicate item_id")
        self.first[-1]["item_id"] = "UNKNOWN"
        self.assert_rejected("unknown item_id")

    def test_missing_rank_and_duplicate_recommendation(self):
        removed = self.mapping.pop()
        self.assert_rejected("ranks must be 1..10")
        self.mapping.append(removed)
        self.mapping.append(dict(self.mapping[0]))
        self.assert_rejected("duplicate recommendation")

    def test_inconsistent_pair_item_mapping(self):
        self.mapping[10]["item_id"] = "I1"
        self.assert_rejected("item_id refers to different")
        self.mapping[10]["item_id"] = "NEW"
        self.assert_rejected("pair has multiple item_ids")

    def test_wrong_cv_and_display_text(self):
        self.first[0]["cv_id"] = "WRONG"
        # Ensure rectangular CSV columns for all rows.
        for r in self.first[1:]:
            r["cv_id"] = "SYNTHETIC"
        self.assert_rejected("CV identity differs")
        for r in self.first:
            r.pop("cv_id")
            r["job_title"] = "Original"
        for r in self.second:
            r["job_title"] = "Changed"
        self.assert_rejected("different CV/job display")

    def test_existing_results_are_not_overwritten(self):
        self.run_eval()
        original = (self.out / "SUMMARY.md").read_bytes()
        with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
            self.run_eval()
        self.assertEqual(original, (self.out / "SUMMARY.md").read_bytes())

    def test_macro_averages_two_cvs(self):
        self.mapping += [{**r, "cv_id": "SECOND_SYNTHETIC", "item_id": "B" + r["item_id"]} for r in list(self.mapping)]
        self.first += [{"item_id": f"BI{i}", "score": "2"} for i in range(1, 13)]
        self.second += [{"item_id": f"BI{i}", "score": "0"} for i in range(1, 13)]
        self.save()
        result = module.evaluate(*self.paths, self.out, expected_cvs=2)
        metric = next(r for r in result["metrics"] if r["configuration"] == module.CONFIGS[0] and r["reviewer"] == "reviewer1" and r["k"] == 10)
        self.assertAlmostEqual(metric["mean_relevance"], 1.45)
        self.assertAlmostEqual(metric["score2_proportion"], .65)
        self.assertEqual(metric["rating_count"], 20)


if __name__ == "__main__":
    unittest.main()
