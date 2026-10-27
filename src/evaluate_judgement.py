"""Evaluate completed independent human ratings; standard library only.

Usage:
    python3 src/evaluate_judgement.py --reviewer1 reviewer1.csv \
        --reviewer2 reviewer2.csv --mapping internal/recommendation_item_map.csv \
        --out-dir results/judgement/evaluation1

Requires both configurations, exactly 10 ranks for every CV (30 by default),
and complete unique-item ratings. Refuses existing output directories.
No missing ratings are imputed. Kappa is unweighted. Optional pooled rows
average reviewer metrics equally, never treating that average as consensus.
"""

import argparse
import csv
import hashlib
import sys
from collections import Counter, defaultdict
from pathlib import Path


CONFIGS = ("pop_weight=1.0", "pop_weight=0.0")
PAIR = ("cv_id", "JobID", "WindowID")


def read_csv(path, required):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        if len(fields) != len(set(fields)) or not set(required).issubset(fields):
            raise ValueError(f"{path}: missing required columns or duplicate headers")
        rows = []
        for row in reader:
            if None in row or any(v is None for v in row.values()):
                raise ValueError(f"{path}: malformed CSV row")
            rows.append({k: v.strip() for k, v in row.items()})
        return rows


def load_mapping(path, expected_cvs):
    rows = read_csv(path, ("configuration", *PAIR, "rank", "item_id"))
    item_pairs, pair_items, seen = {}, {}, set()
    ranks = defaultdict(list)
    cv_sets = {c: set() for c in CONFIGS}
    for row in rows:
        conf, item = row["configuration"], row["item_id"]
        pair = tuple(row[k] for k in PAIR)
        if conf not in CONFIGS or not item or not all(pair):
            raise ValueError("Mapping: invalid configuration or empty identifier")
        try:
            row["rank"] = int(row["rank"])
        except ValueError:
            raise ValueError("Mapping: invalid rank") from None
        if item in item_pairs and item_pairs[item] != pair:
            raise ValueError("Mapping: item_id refers to different CV-job pairs")
        if pair in pair_items and pair_items[pair] != item:
            raise ValueError("Mapping: one CV-job pair has multiple item_ids")
        if (conf, pair) in seen:
            raise ValueError("Mapping: duplicate recommendation")
        seen.add((conf, pair))
        item_pairs[item], pair_items[pair] = pair, item
        cv_sets[conf].add(row["cv_id"])
        ranks[conf, row["cv_id"]].append(row["rank"])
    if expected_cvs < 1 or any(len(s) != expected_cvs for s in cv_sets.values()):
        raise ValueError(f"Mapping: expected {expected_cvs} CVs per configuration")
    if cv_sets[CONFIGS[0]] != cv_sets[CONFIGS[1]]:
        raise ValueError("Mapping: configurations have different CV sets")
    for key, values in ranks.items():
        if sorted(values) != list(range(1, 11)):
            raise ValueError(f"Mapping: {key}: ranks must be 1..10 exactly once")
    return rows, item_pairs


def load_reviewer(path, item_pairs):
    rows = read_csv(path, ("item_id", "score"))
    scores, display = {}, {}
    for row in rows:
        item = row["item_id"]
        if item in scores:
            raise ValueError(f"{path}: duplicate item_id {item}")
        if item not in item_pairs:
            raise ValueError(f"{path}: unknown item_id {item}")
        if row["score"] not in ("0", "1", "2"):
            raise ValueError(f"{path}: score for {item} must be 0, 1 or 2; blank ratings are incomplete")
        if row.get("status", "").casefold() == "needs_information":
            raise ValueError(f"{path}: unresolved needs_information for {item}")
        if "cv_id" in row and row["cv_id"] != item_pairs[item][0]:
            raise ValueError(f"{path}: CV identity differs from mapping for {item}")
        scores[item] = int(row["score"])
        # If full prepared forms are supplied, ensure both reviewers saw the same text.
        display[item] = {k: v for k, v in row.items() if k.startswith("cv_") or k.startswith("job_")}
    missing = item_pairs.keys() - scores.keys()
    if missing:
        raise ValueError(f"{path}: incomplete ratings; missing {len(missing)} items")
    return scores, display


def agreement_metrics(first, second):
    n = len(first)
    if not n or first.keys() != second.keys():
        raise ValueError("Agreement requires identical nonempty item sets")
    confusion = Counter((first[i], second[i]) for i in first)
    a, b = Counter(first.values()), Counter(second.values())
    po = sum(confusion[c, c] for c in range(3)) / n
    expected_numerator = sum(a[c] * b[c] for c in range(3))
    pe = expected_numerator / (n * n)
    kappa = None if expected_numerator == n * n else (po - pe) / (1 - pe)
    return {"unique_items": n, "percentage_agreement": 100 * po,
            "observed_agreement": po, "expected_agreement": pe,
            "cohens_kappa": kappa,
            "kappa_status": "undefined: expected agreement is 1" if kappa is None else "defined"}, confusion


def write_csv(path, rows):
    with path.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def evaluate(reviewer1, reviewer2, mapping_path, out_dir, expected_cvs=30, include_pooled=False):
    csv.field_size_limit(100_000_000)
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise ValueError("Output directory already exists; refusing to overwrite")
    if Path(reviewer1).resolve() == Path(reviewer2).resolve():
        raise ValueError("Supply separate reviewer files")
    mapping, item_pairs = load_mapping(mapping_path, expected_cvs)
    first, display1 = load_reviewer(reviewer1, item_pairs)
    second, display2 = load_reviewer(reviewer2, item_pairs)
    if display1 != display2:
        raise ValueError("Reviewers have different CV/job display fields or text")
    agreement, confusion = agreement_metrics(first, second)
    joined = [{**r, "reviewer1_score": first[r["item_id"]], "reviewer2_score": second[r["item_id"]]} for r in mapping]
    if len(joined) != 20 * expected_cvs:
        raise ValueError("Joining changed the expected recommendation count")
    per_cv = []
    for conf in CONFIGS:
        for cv in sorted({r["cv_id"] for r in mapping}):
            for k in (3, 10):
                selected = [r for r in joined if r["configuration"] == conf and r["cv_id"] == cv and r["rank"] <= k]
                if len(selected) != k:
                    raise ValueError("Incomplete Top-K after joining")
                for reviewer in ("reviewer1", "reviewer2"):
                    values = [r[f"{reviewer}_score"] for r in selected]
                    per_cv.append({"configuration": conf, "cv_id": cv, "reviewer": reviewer, "k": k,
                                   "recommendations": k, "mean_relevance": sum(values) / k,
                                   "score2_proportion": values.count(2) / k})
    metrics = []
    for conf in CONFIGS:
        for k in (3, 10):
            for reviewer in ("reviewer1", "reviewer2"):
                selected = [r for r in per_cv if (r["configuration"], r["k"], r["reviewer"]) == (conf, k, reviewer)]
                metrics.append({"configuration": conf, "reviewer": reviewer, "k": k,
                                "cv_count": expected_cvs, "rating_count": expected_cvs * k,
                                "mean_relevance": sum(r["mean_relevance"] for r in selected) / expected_cvs,
                                "score2_proportion": sum(r["score2_proportion"] for r in selected) / expected_cvs})
            if include_pooled:
                left, right = metrics[-2:]
                metrics.append({"configuration": conf, "reviewer": "equal_reviewer_average_not_consensus", "k": k,
                                "cv_count": expected_cvs, "rating_count": expected_cvs * k * 2,
                                "mean_relevance": (left["mean_relevance"] + right["mean_relevance"]) / 2,
                                "score2_proportion": (left["score2_proportion"] + right["score2_proportion"]) / 2})
    lookup = {(r["configuration"], r["reviewer"], r["k"]): r for r in metrics}
    comparison = []
    for reviewer, k in sorted({(r["reviewer"], r["k"]) for r in metrics}):
        one, zero = (lookup[c, reviewer, k] for c in CONFIGS)
        comparison.append({"reviewer": reviewer, "k": k,
                           "mean_relevance_pop1": one["mean_relevance"], "mean_relevance_pop0": zero["mean_relevance"],
                           "mean_relevance_delta_pop1_minus_pop0": one["mean_relevance"] - zero["mean_relevance"],
                           "score2_proportion_pop1": one["score2_proportion"], "score2_proportion_pop0": zero["score2_proportion"],
                           "score2_percentage_point_delta": 100 * (one["score2_proportion"] - zero["score2_proportion"])})
    unique = [{"item_id": i, **dict(zip(PAIR, item_pairs[i])), "reviewer1_score": first[i], "reviewer2_score": second[i]}
              for i in sorted(item_pairs)]
    provenance = [{"input": label, "path": str(path), "sha256": sha256(path)} for label, path in
                  [("reviewer1", reviewer1), ("reviewer2", reviewer2), ("mapping", mapping_path)]]
    # All validation and calculations precede creation of any final result file.
    out_dir.mkdir(parents=True, exist_ok=False)
    outputs = {
        "agreement.csv": [agreement],
        "confusion_matrix.csv": [{"reviewer1_score": a, "reviewer2_score": b, "count": confusion[a, b]} for a in range(3) for b in range(3)],
        "unique_item_ratings.csv": unique, "recommendation_ratings.csv": joined,
        "per_cv_metrics.csv": per_cv, "configuration_metrics.csv": metrics,
        "comparison.csv": comparison, "input_fingerprints.csv": provenance,
    }
    for filename, rows in outputs.items():
        write_csv(out_dir / filename, rows)
    kappa_text = "غير معرّفة لأن Pe=1" if agreement["cohens_kappa"] is None else f"{agreement['cohens_kappa']:.6f}"
    summary = ["# نتائج التقييم البشري", "",
               f"عدد CVs: {expected_cvs}. عدد ظهور التوصيات: {len(mapping)}. العناصر الفريدة: {len(unique)}.",
               "جميع العناصر لها درجتان صحيحتان؛ الرتب 1–10 مكتملة في الإعدادين. لم تُعوّض أي درجة مفقودة.", "",
               f"الاتفاق التام: {agreement['percentage_agreement']:.4f}%. Cohen's kappa غير الموزونة: {kappa_text}.",
               "الاتفاق محسوب على العناصر الفريدة فقط. كابا = (Po − Pe) / (1 − Pe)، وPe من توزيعي درجات المقيّمين.", "",
               "| الإعداد | المقيّم | K | متوسط الصلة (0–2) | نسبة الدرجة 2 |", "|---|---|---:|---:|---:|"]
    summary.extend(f"| {r['configuration']} | {r['reviewer']} | {r['k']} | {r['mean_relevance']:.4f} | {100*r['score2_proportion']:.2f}% |" for r in metrics)
    summary += ["", "## المقارنة: الإعداد 1 ناقص الإعداد 0", "",
                "| المقيّم | K | فرق متوسط الصلة | فرق نسبة الدرجة 2 (نقطة مئوية) |", "|---|---:|---:|---:|"]
    summary.extend(f"| {r['reviewer']} | {r['k']} | {r['mean_relevance_delta_pop1_minus_pop0']:.4f} | {r['score2_percentage_point_delta']:.2f} |" for r in comparison)
    summary += ["", "## طريقة الحساب والقيود", "",
                "يُحسب متوسط الدرجات ونسبة أحكام الدرجة 2 داخل Top-K لكل CV، ثم المتوسط بالتساوي عبر CVs. لا يشترط اتفاق المقيّمين على الدرجة 2.",
                "الأعمدة score2_proportion في CSV بين 0 و1؛ العرض في الملخص بالنسبة المئوية. الفروق دائمًا pop1 ناقص pop0.",
                "المقيّمان معروضان منفصلين. خيار --include-pooled يضيف متوسطًا متساوي الوزن لمقاييسهما فقط؛ لا يمثل حكمًا متفقًا عليه، ولا يدخل في حساب كابا.",
                "الزوج المشترك له حكم واحد لكل مقيّم، ويعاد استخدامه داخل قائمة كل إعداد في رتبته الأصلية. يحتوي per_cv_metrics.csv على النتائج اللازمة للمقارنة المزدوجة لكل CV.",
                "التحقق بنيوي؛ لا يثبت استقلال المقيّمين أو صحة تعريف الإعدادات. يجب حفظ ملفات التحكيم الأصلية وبيانات التوليد. لا تُستنتج Recall أو دلالة إحصائية من هذه المخرجات.",
                "إذا كانت المدخلات مصطنعة، فهذه نتائج اختبار فقط وليست نتائج المشروع. لا تُولّد الأداة أحكامًا بشرية.", ""]
    with (out_dir / "SUMMARY.md").open("x", encoding="utf-8") as handle:
        handle.write("\n".join(summary))
    return {"agreement": agreement, "metrics": metrics, "comparison": comparison}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reviewer1", required=True, type=Path)
    parser.add_argument("--reviewer2", required=True, type=Path)
    parser.add_argument("--mapping", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--expected-cvs", type=int, default=30)
    parser.add_argument("--include-pooled", action="store_true")
    args = parser.parse_args()
    try:
        evaluate(args.reviewer1, args.reviewer2, args.mapping, args.out_dir, args.expected_cvs, args.include_pooled)
    except (ValueError, OSError, csv.Error) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Validated completed ratings; results saved to {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
