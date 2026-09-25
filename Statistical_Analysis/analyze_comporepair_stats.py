#!/usr/bin/env python3
"""CompoRepair-RAG statistical analysis from the raw-results ZIP.

Primary design implemented here
-------------------------------
* Safe Composer is intentionally excluded from primary analyses.
* B1 is not present in the supplied raw-results archive and is not analyzed.
* Compound primary comparisons are:
    Fixed vs B0 (failure state)
    Fixed vs B2
    Fixed vs B3
    Fixed vs Reverse
* Method comparisons use contrast-specific trace intersections.
* A trace is semantically evaluable only when BOTH the failure-stage evaluation
  and the compared final evaluation(s) are free of generation failures and
  semantic-judge errors and contain a Boolean semantic_correct value.
* Final semantic accuracy differences use paired bootstrap 95% percentile CIs.
* Binary paired differences use exact two-sided McNemar/binomial tests.
* Holm correction is applied within each pre-specified comparison family.

The script reads the ZIP directly; it does not need to extract the ~400 MB raw
JSON collection to disk.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import beta, binomtest


SINGLE_CONDITIONS = ("M", "D", "L")
COMPOUND_CONDITIONS = ("M_D", "M_L", "D_L", "M_D_L")
PRIMARY_COMPARATORS = ("b0", "b2", "b3", "reverse")
ANALYZED_METHODS = ("fixed", "reverse", "b2", "b3")

DATASET_LABELS = {
    "2wikimultihopqa": "2Wiki",
    "hotpotqa": "Hotpot",
}
MODEL_LABELS = {
    "llama3.1:8b": "Llama",
    "qwen3.5:9b": "Qwen",
}


@dataclass(frozen=True)
class Outcome:
    trace_id: str
    baseline_correct: bool
    final_correct: bool
    baseline_em: float
    final_em: float
    baseline_f1: float
    final_f1: float
    structural_clear: bool
    joint_full_repair: bool
    new_failure: bool
    latency_ms: Optional[float]
    total_tokens: Optional[float]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze CompoRepair-RAG raw results.")
    p.add_argument("input_zip", type=Path, help="Raw-results ZIP archive")
    p.add_argument("--output", type=Path, default=Path("analysis_output"), help="Output directory")
    p.add_argument("--bootstrap", type=int, default=10_000, help="Paired bootstrap samples")
    p.add_argument("--seed", type=int, default=20260830, help="Random seed")
    return p.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def is_semantic_evaluable(ev: Any) -> bool:
    return (
        isinstance(ev, dict)
        and isinstance(ev.get("semantic_correct"), bool)
        and not bool(ev.get("generation_failed", False))
        and not bool(ev.get("semantic_judge_error", False))
    )


def eval_value(ev: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    value = ev.get(key, default)
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return float(default)


def exact_mcnemar_p(a_only: int, b_only: int) -> float:
    """Exact two-sided McNemar test via Binomial(n_discordant, 0.5)."""
    n = int(a_only) + int(b_only)
    if n == 0:
        return 1.0
    return float(binomtest(int(a_only), n=n, p=0.5, alternative="two-sided").pvalue)


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> Tuple[Optional[float], Optional[float]]:
    if n <= 0:
        return (None, None)
    if k < 0 or k > n:
        raise ValueError("k must be in [0,n]")
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2.0, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1.0 - alpha / 2.0, k + 1, n - k))
    return lo, hi


def paired_bootstrap_ci(
    a: Sequence[float],
    b: Sequence[float],
    n_boot: int,
    rng: np.random.Generator,
    alpha: float = 0.05,
) -> Tuple[Optional[float], Optional[float]]:
    if len(a) != len(b):
        raise ValueError("Paired arrays must have equal length")
    n = len(a)
    if n == 0:
        return (None, None)
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if n_boot <= 0:
        return (None, None)
    values: List[np.ndarray] = []
    # Batched to keep memory modest while remaining fast.
    batch = min(2000, n_boot)
    done = 0
    while done < n_boot:
        cur = min(batch, n_boot - done)
        idx = rng.integers(0, n, size=(cur, n))
        values.append(diff[idx].mean(axis=1))
        done += cur
    boots = np.concatenate(values)
    q = np.quantile(boots, [alpha / 2.0, 1.0 - alpha / 2.0], method="linear")
    return float(q[0]), float(q[1])


def holm_adjust(p_values: Sequence[Optional[float]]) -> List[Optional[float]]:
    indexed = [(i, float(p)) for i, p in enumerate(p_values) if p is not None and math.isfinite(float(p))]
    out: List[Optional[float]] = [None] * len(p_values)
    if not indexed:
        return out
    indexed.sort(key=lambda x: x[1])
    m = len(indexed)
    running = 0.0
    for rank, (idx, p) in enumerate(indexed):
        adj = min(1.0, (m - rank) * p)
        running = max(running, adj)
        out[idx] = running
    return out


def safe_mean(values: Sequence[float]) -> Optional[float]:
    return None if not values else float(sum(values) / len(values))


def safe_median(values: Sequence[float]) -> Optional[float]:
    return None if not values else float(statistics.median(values))


def active_failures(condition: str) -> set[str]:
    return set(condition.split("_"))


def result_filename(condition: str, method: str) -> str:
    if method == "fixed":
        return f"{condition}_repair_results.json"
    return f"{condition}_{method}_repair_results.json"


def discover_setting_dirs(zf: zipfile.ZipFile) -> List[str]:
    marker = "/repaired_result/"
    dirs = set()
    for name in zf.namelist():
        if marker in name and name.endswith("_repair_results.json"):
            dirs.add(name.split(marker, 1)[0])
    return sorted(dirs)


def load_json_list(zf: zipfile.ZipFile, path: str) -> List[Dict[str, Any]]:
    try:
        raw = zf.read(path)
    except KeyError as e:
        raise FileNotFoundError(f"Missing archive member: {path}") from e
    data = json.loads(raw)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return [x for x in data if isinstance(x, dict)]


def infer_setting_label(records: Sequence[Mapping[str, Any]], folder: str) -> Tuple[str, str, str]:
    record = next((r for r in records if r.get("trace_id")), None)
    if record is None:
        return folder, folder, folder
    dataset_raw = str(record.get("dataset", "unknown"))
    model_raw = str((record.get("model_manifest") or {}).get("model", "unknown"))
    dataset = DATASET_LABELS.get(dataset_raw, dataset_raw)
    model = MODEL_LABELS.get(model_raw, model_raw)
    return f"{dataset}-{model}", dataset_raw, model_raw


def parse_result_records(
    records: Sequence[Mapping[str, Any]], condition: str
) -> Tuple[Dict[str, Outcome], Dict[str, Any]]:
    active = active_failures(condition)
    out: Dict[str, Outcome] = {}
    duplicate_ids: List[str] = []
    missing_trace_id = 0
    baseline_unevaluable = 0
    final_unevaluable = 0

    for r in records:
        tid = r.get("trace_id")
        if not isinstance(tid, str) or not tid:
            missing_trace_id += 1
            continue
        if tid in out:
            duplicate_ids.append(tid)
            continue

        baseline_ev = r.get("failure_evaluation")
        final_ev = r.get("repair_evaluation") or r.get("evaluation")
        if not is_semantic_evaluable(baseline_ev):
            baseline_unevaluable += 1
            continue
        if not is_semantic_evaluable(final_ev):
            final_unevaluable += 1
            continue

        baseline_correct = bool(baseline_ev["semantic_correct"])
        final_correct = bool(final_ev["semantic_correct"])
        remaining = set(r.get("failure_after_repair") or [])
        structural_clear = not bool(active & remaining)
        joint = bool(final_correct and structural_clear)
        new_failure = bool(r.get("new_failures_after_repair") or [])

        latency = r.get("repair_latency_ms")
        latency_ms = float(latency) if isinstance(latency, (int, float)) and math.isfinite(float(latency)) else None
        token_usage = r.get("repair_token_usage") or {}
        tokens = token_usage.get("total_tokens") if isinstance(token_usage, dict) else None
        total_tokens = float(tokens) if isinstance(tokens, (int, float)) and math.isfinite(float(tokens)) else None

        out[tid] = Outcome(
            trace_id=tid,
            baseline_correct=baseline_correct,
            final_correct=final_correct,
            baseline_em=eval_value(baseline_ev, "exact_match"),
            final_em=eval_value(final_ev, "exact_match"),
            baseline_f1=eval_value(baseline_ev, "token_f1"),
            final_f1=eval_value(final_ev, "token_f1"),
            structural_clear=structural_clear,
            joint_full_repair=joint,
            new_failure=new_failure,
            latency_ms=latency_ms,
            total_tokens=total_tokens,
        )

    audit = {
        "raw_record_count": len(records),
        "usable_trace_count": len(out),
        "missing_trace_id_count": missing_trace_id,
        "duplicate_trace_id_count": len(duplicate_ids),
        "duplicate_trace_ids": duplicate_ids,
        "baseline_unevaluable_count": baseline_unevaluable,
        "final_unevaluable_count": final_unevaluable,
    }
    return out, audit


def consensus_baseline(method_maps: Mapping[str, Mapping[str, Outcome]]) -> Tuple[Dict[str, bool], int]:
    all_ids: set[str] = set()
    for m in method_maps.values():
        all_ids.update(m.keys())
    baseline: Dict[str, bool] = {}
    mismatches = 0
    for tid in all_ids:
        vals = {m[tid].baseline_correct for m in method_maps.values() if tid in m}
        if len(vals) > 1:
            mismatches += 1
            continue
        if vals:
            baseline[tid] = next(iter(vals))
    return baseline, mismatches


def summarize_method(outcomes: Mapping[str, Outcome]) -> Dict[str, Any]:
    rows = list(outcomes.values())
    n = len(rows)
    if n == 0:
        return {"N": 0}
    before_correct = sum(r.baseline_correct for r in rows)
    after_correct = sum(r.final_correct for r in rows)
    recovered = sum((not r.baseline_correct) and r.final_correct for r in rows)
    regressed = sum(r.baseline_correct and (not r.final_correct) for r in rows)
    stayed_correct = sum(r.baseline_correct and r.final_correct for r in rows)
    stayed_wrong = sum((not r.baseline_correct) and (not r.final_correct) for r in rows)
    recovery_den = n - before_correct
    regression_den = before_correct
    rec_ci = clopper_pearson(recovered, recovery_den)
    reg_ci = clopper_pearson(regressed, regression_den)
    lat = [r.latency_ms for r in rows if r.latency_ms is not None]
    tok = [r.total_tokens for r in rows if r.total_tokens is not None]
    return {
        "N": n,
        "before_acc": before_correct / n,
        "after_acc": after_correct / n,
        "delta": (after_correct - before_correct) / n,
        "recovered": recovered,
        "recovery_den": recovery_den,
        "recovery_rate": (recovered / recovery_den) if recovery_den else None,
        "recovery_ci_exact": list(rec_ci),
        "regressed": regressed,
        "regression_den": regression_den,
        "regression_rate": (regressed / regression_den) if regression_den else None,
        "regression_ci_exact": list(reg_ci),
        "stayed_correct": stayed_correct,
        "stayed_wrong": stayed_wrong,
        "structural_clearance": sum(r.structural_clear for r in rows) / n,
        "joint_full_repair": sum(r.joint_full_repair for r in rows) / n,
        "new_failure_rate": sum(r.new_failure for r in rows) / n,
        "em_before": sum(r.baseline_em for r in rows) / n,
        "em_after": sum(r.final_em for r in rows) / n,
        "f1_before": sum(r.baseline_f1 for r in rows) / n,
        "f1_after": sum(r.final_f1 for r in rows) / n,
        "latency_mean_ms": safe_mean(lat),
        "latency_median_ms": safe_median(lat),
        "tokens_mean": safe_mean(tok),
    }


def compare_methods(
    a_map: Mapping[str, Outcome],
    b_map: Optional[Mapping[str, Outcome]],
    comparator: str,
    n_boot: int,
    rng: np.random.Generator,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Compare A (normally Fixed) with comparator B.

    For B0, B's final outcome is A's failure-stage outcome and B structural/joint
    metrics are zero by construction because the injected failures remain.
    """
    if comparator == "b0":
        ids = sorted(a_map.keys())
        baseline_mismatch = 0
    else:
        assert b_map is not None
        ids = sorted(set(a_map) & set(b_map))
        baseline_mismatch = sum(a_map[i].baseline_correct != b_map[i].baseline_correct for i in ids)
        ids = [i for i in ids if a_map[i].baseline_correct == b_map[i].baseline_correct]

    a = [a_map[i] for i in ids]
    if comparator == "b0":
        b_final = [r.baseline_correct for r in a]
        b_em = [r.baseline_em for r in a]
        b_f1 = [r.baseline_f1 for r in a]
        b_structural = [False] * len(a)
        b_joint = [False] * len(a)
        b_latency: List[float] = []
        b_tokens: List[float] = []
    else:
        assert b_map is not None
        b = [b_map[i] for i in ids]
        b_final = [r.final_correct for r in b]
        b_em = [r.final_em for r in b]
        b_f1 = [r.final_f1 for r in b]
        b_structural = [r.structural_clear for r in b]
        b_joint = [r.joint_full_repair for r in b]
        b_latency = [r.latency_ms for r in b if r.latency_ms is not None]
        b_tokens = [r.total_tokens for r in b if r.total_tokens is not None]

    a_final = [r.final_correct for r in a]
    n = len(ids)
    a_only = sum(x and not y for x, y in zip(a_final, b_final))
    b_only = sum((not x) and y for x, y in zip(a_final, b_final))
    both_correct = sum(x and y for x, y in zip(a_final, b_final))
    both_wrong = sum((not x) and (not y) for x, y in zip(a_final, b_final))
    delta_ci = paired_bootstrap_ci(a_final, b_final, n_boot, rng)

    # Recovery comparison: restrict to common traces that were wrong at failure stage.
    rec_idx = [j for j, r in enumerate(a) if not r.baseline_correct]
    a_rec = [a_final[j] for j in rec_idx]
    b_rec = [b_final[j] for j in rec_idx]
    rec_a_only = sum(x and not y for x, y in zip(a_rec, b_rec))
    rec_b_only = sum((not x) and y for x, y in zip(a_rec, b_rec))
    rec_ci = paired_bootstrap_ci(a_rec, b_rec, n_boot, rng) if rec_idx else (None, None)

    # Regression comparison: restrict to common traces that were correct at failure stage.
    reg_idx = [j for j, r in enumerate(a) if r.baseline_correct]
    a_reg = [not a_final[j] for j in reg_idx]
    b_reg = [not b_final[j] for j in reg_idx]
    reg_a_only = sum(x and not y for x, y in zip(a_reg, b_reg))
    reg_b_only = sum((not x) and y for x, y in zip(a_reg, b_reg))
    reg_ci = paired_bootstrap_ci(a_reg, b_reg, n_boot, rng) if reg_idx else (None, None)

    a_structural = [r.structural_clear for r in a]
    a_joint = [r.joint_full_repair for r in a]
    st_a_only = sum(x and not y for x, y in zip(a_structural, b_structural))
    st_b_only = sum((not x) and y for x, y in zip(a_structural, b_structural))
    joint_a_only = sum(x and not y for x, y in zip(a_joint, b_joint))
    joint_b_only = sum((not x) and y for x, y in zip(a_joint, b_joint))

    a_em = [r.final_em for r in a]
    a_f1 = [r.final_f1 for r in a]
    em_ci = paired_bootstrap_ci(a_em, b_em, n_boot, rng)
    f1_ci = paired_bootstrap_ci(a_f1, b_f1, n_boot, rng)
    st_ci = paired_bootstrap_ci(a_structural, b_structural, n_boot, rng)
    joint_ci = paired_bootstrap_ci(a_joint, b_joint, n_boot, rng)

    a_latency = [r.latency_ms for r in a if r.latency_ms is not None]
    a_tokens = [r.total_tokens for r in a if r.total_tokens is not None]

    result = {
        "N": n,
        "A_acc": safe_mean(a_final),
        "B_acc": safe_mean(b_final),
        "delta": (safe_mean(a_final) - safe_mean(b_final)) if n else None,
        "delta_ci": list(delta_ci),
        "both_correct": both_correct,
        "A_only_correct": a_only,
        "B_only_correct": b_only,
        "both_wrong": both_wrong,
        "mcnemar_p": exact_mcnemar_p(a_only, b_only),
        "baseline_wrong_N": len(rec_idx),
        "A_recovery": safe_mean(a_rec),
        "B_recovery": safe_mean(b_rec),
        "recovery_delta": (safe_mean(a_rec) - safe_mean(b_rec)) if rec_idx else None,
        "recovery_delta_ci": list(rec_ci),
        "recovery_mcnemar_p": exact_mcnemar_p(rec_a_only, rec_b_only) if rec_idx else None,
        "baseline_correct_N": len(reg_idx),
        "A_regression": safe_mean(a_reg),
        "B_regression": safe_mean(b_reg),
        "regression_delta": (safe_mean(a_reg) - safe_mean(b_reg)) if reg_idx else None,
        "regression_delta_ci": list(reg_ci),
        "regression_mcnemar_p": exact_mcnemar_p(reg_a_only, reg_b_only) if reg_idx else None,
        "A_structural_clearance": safe_mean(a_structural),
        "B_structural_clearance": safe_mean(b_structural),
        "structural_delta": (safe_mean(a_structural) - safe_mean(b_structural)) if n else None,
        "structural_delta_ci": list(st_ci),
        "structural_mcnemar_p": exact_mcnemar_p(st_a_only, st_b_only),
        "A_joint_full_repair": safe_mean(a_joint),
        "B_joint_full_repair": safe_mean(b_joint),
        "joint_delta": (safe_mean(a_joint) - safe_mean(b_joint)) if n else None,
        "joint_delta_ci": list(joint_ci),
        "joint_mcnemar_p": exact_mcnemar_p(joint_a_only, joint_b_only),
        "A_em": safe_mean(a_em),
        "B_em": safe_mean(b_em),
        "em_delta": (safe_mean(a_em) - safe_mean(b_em)) if n else None,
        "em_delta_ci": list(em_ci),
        "A_f1": safe_mean(a_f1),
        "B_f1": safe_mean(b_f1),
        "f1_delta": (safe_mean(a_f1) - safe_mean(b_f1)) if n else None,
        "f1_delta_ci": list(f1_ci),
        "A_latency_mean_ms": safe_mean(a_latency),
        "A_latency_median_ms": safe_median(a_latency),
        "B_latency_mean_ms": safe_mean(b_latency),
        "B_latency_median_ms": safe_median(b_latency),
        "A_tokens_mean": safe_mean(a_tokens),
        "B_tokens_mean": safe_mean(b_tokens),
    }
    audit = {
        "pairwise_trace_count": n,
        "baseline_mismatch_excluded_count": baseline_mismatch,
        "A_available_trace_count": len(a_map),
        "B_available_trace_count": len(b_map) if b_map is not None else len(a_map),
    }
    return result, audit


def apply_holm_to_rows(rows: List[Dict[str, Any]], family_key: str = "comparison") -> None:
    p_fields = [
        ("mcnemar_p", "holm_p"),
        ("recovery_mcnemar_p", "recovery_holm_p"),
        ("regression_mcnemar_p", "regression_holm_p"),
        ("structural_mcnemar_p", "structural_holm_p"),
        ("joint_mcnemar_p", "joint_holm_p"),
    ]
    families = sorted({str(r[family_key]) for r in rows})
    for fam in families:
        idxs = [i for i, r in enumerate(rows) if str(r[family_key]) == fam]
        for raw_field, adj_field in p_fields:
            adjusted = holm_adjust([rows[i].get(raw_field) for i in idxs])
            for i, adj in zip(idxs, adjusted):
                rows[i][adj_field] = adj


def apply_holm_single(rows: List[Dict[str, Any]]) -> None:
    adjusted = holm_adjust([r.get("mcnemar_p") for r in rows])
    for r, adj in zip(rows, adjusted):
        r["holm_p"] = adj


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Optional[Sequence[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    if fields is None:
        keys: List[str] = []
        seen = set()
        for row in rows:
            for k in row.keys():
                if k not in seen:
                    seen.add(k)
                    keys.append(k)
        fields = keys
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        w.writeheader()
        for row in rows:
            clean = {}
            for k in fields:
                v = row.get(k)
                if isinstance(v, (list, dict)):
                    clean[k] = json.dumps(v, ensure_ascii=False)
                else:
                    clean[k] = v
            w.writerow(clean)


def pct(x: Optional[float]) -> str:
    return "NA" if x is None else f"{100.0 * x:.1f}%"


def pp(x: Optional[float]) -> str:
    return "NA" if x is None else f"{100.0 * x:+.1f} pp"


def p_text(p: Optional[float]) -> str:
    if p is None:
        return "NA"
    if p < 0.001:
        return f"{p:.2e}"
    return f"{p:.4f}"


def generate_markdown_summary(
    meta: Mapping[str, Any],
    single_rows: Sequence[Mapping[str, Any]],
    pair_rows: Sequence[Mapping[str, Any]],
) -> str:
    lines = [
        "# CompoRepair-RAG Statistical Analysis",
        "",
        "This report is generated directly from the raw trace-level JSON files.",
        "Safe Composer is excluded from the primary analysis by design.",
        "",
        "## Analysis configuration",
        "",
        f"- Bootstrap resamples: **{meta['bootstrap_samples']:,}**",
        f"- Bootstrap seed: **{meta['bootstrap_seed']}**",
        "- Paired test: **exact two-sided McNemar/binomial test**",
        "- Multiple testing: **Holm family-wise correction** within each pre-specified contrast/metric family",
        "- Pairing: **contrast-specific trace intersections**",
        "- Exclusions: generation failures and semantic-judge errors are not treated as semantic outcomes",
        "",
        "## Single-failure repair",
        "",
        "| Setting | Failure | N | Before | After | Gain | 95% CI | Recovery | Regression | Holm p |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in single_rows:
        ci = r.get("delta_ci") or [None, None]
        ci_txt = f"[{pp(ci[0])}, {pp(ci[1])}]"
        lines.append(
            f"| {r['setting']} | {r['condition']} | {r['N']} | {pct(r['before_acc'])} | {pct(r['after_acc'])} | "
            f"{pp(r['delta'])} | {ci_txt} | {pct(r['recovery_rate'])} | {pct(r['regression_rate'])} | {p_text(r.get('holm_p'))} |"
        )

    lines += ["", "## Primary compound comparisons", ""]
    for comparison in ("Fixed_vs_B0", "Fixed_vs_B2", "Fixed_vs_B3", "Fixed_vs_Reverse"):
        subset = [r for r in pair_rows if r["comparison"] == comparison]
        sig = sum((r.get("holm_p") is not None and r["holm_p"] < 0.05) for r in subset)
        lines.append(f"### {comparison.replace('_', ' ')}")
        lines.append("")
        lines.append(f"Semantic accuracy significant after Holm: **{sig}/{len(subset)}** comparisons.")
        lines.append("")
        lines.append("| Setting | Condition | N | A acc | B acc | Delta | 95% CI | A-only | B-only | Holm p |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for r in subset:
            ci = r.get("delta_ci") or [None, None]
            lines.append(
                f"| {r['setting']} | {r['condition']} | {r['N']} | {pct(r['A_acc'])} | {pct(r['B_acc'])} | "
                f"{pp(r['delta'])} | [{pp(ci[0])}, {pp(ci[1])}] | {r['A_only_correct']} | {r['B_only_correct']} | {p_text(r.get('holm_p'))} |"
            )
        lines.append("")

    order_sig = [r for r in pair_rows if r["comparison"] == "Fixed_vs_Reverse" and r.get("holm_p") is not None and r["holm_p"] < 0.05]
    lines += ["## Order effects surviving Holm", ""]
    if order_sig:
        for r in order_sig:
            lines.append(
                f"- **{r['setting']} {r['condition']}**: Fixed {pct(r['A_acc'])} vs Reverse {pct(r['B_acc'])}, "
                f"delta {pp(r['delta'])}, Holm p={p_text(r['holm_p'])}."
            )
    else:
        lines.append("No Fixed-vs-Reverse semantic comparison survived Holm correction.")

    lines += [
        "",
        "## Interpretation notes",
        "",
        "- Positive semantic/recovery/structural/joint deltas favor Fixed.",
        "- For regression, **negative** Fixed-minus-comparator deltas favor Fixed because lower regression is better.",
        "- `joint_full_repair` means final semantic correctness AND clearance of all originally injected failure families for that condition.",
        "- B0 is the failure-stage state, not a newly generated answer.",
        "- B1 is not included because no B1 trace-level result files are present in this archive.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    input_zip: Path = args.input_zip
    output: Path = args.output
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    if not input_zip.exists():
        print(f"ERROR: input ZIP not found: {input_zip}", file=sys.stderr)
        return 2

    single_rows: List[Dict[str, Any]] = []
    method_metric_rows: List[Dict[str, Any]] = []
    pair_rows: List[Dict[str, Any]] = []
    method_vs_b0_rows: List[Dict[str, Any]] = []
    audit_rows: List[Dict[str, Any]] = []
    nested_single: Dict[str, Any] = {}
    nested_pairs: Dict[str, Any] = {}
    nested_methods: Dict[str, Any] = {}

    with zipfile.ZipFile(input_zip) as zf:
        setting_dirs = discover_setting_dirs(zf)
        if not setting_dirs:
            raise RuntimeError("No repaired_result directories were found in the ZIP")

        for setting_dir in setting_dirs:
            # Infer setting label from M single results.
            probe_path = f"{setting_dir}/repaired_result/M_repair_results.json"
            probe_records = load_json_list(zf, probe_path)
            setting_label, dataset_raw, model_raw = infer_setting_label(probe_records, setting_dir)
            nested_single.setdefault(setting_label, {})
            nested_pairs.setdefault(setting_label, {})
            nested_methods.setdefault(setting_label, {})

            # Singles: fixed repair vs its failure-stage answer.
            for cond in SINGLE_CONDITIONS:
                pth = f"{setting_dir}/repaired_result/{cond}_repair_results.json"
                records = load_json_list(zf, pth)
                outcomes, audit = parse_result_records(records, cond)
                summary = summarize_method(outcomes)
                rows = list(outcomes.values())
                a = [r.final_correct for r in rows]
                b = [r.baseline_correct for r in rows]
                ci = paired_bootstrap_ci(a, b, args.bootstrap, rng)
                a_only = sum(x and not y for x, y in zip(a, b))
                b_only = sum((not x) and y for x, y in zip(a, b))
                summary.update({
                    "setting": setting_label,
                    "dataset": dataset_raw,
                    "model": model_raw,
                    "condition": cond,
                    "delta_ci": list(ci),
                    "mcnemar_p": exact_mcnemar_p(a_only, b_only),
                })
                single_rows.append(summary)
                nested_single[setting_label][cond] = summary
                audit_rows.append({
                    "setting": setting_label,
                    "condition": cond,
                    "method": "fixed",
                    "source_file": pth,
                    **audit,
                })

            # Compounds.
            for cond in COMPOUND_CONDITIONS:
                maps: Dict[str, Dict[str, Outcome]] = {}
                audits: Dict[str, Any] = {}
                for method in ANALYZED_METHODS:
                    fname = result_filename(cond, method)
                    pth = f"{setting_dir}/repaired_result/{fname}"
                    records = load_json_list(zf, pth)
                    outcomes, audit = parse_result_records(records, cond)
                    maps[method] = outcomes
                    audits[method] = audit
                    sm = summarize_method(outcomes)
                    method_metric_rows.append({
                        "setting": setting_label,
                        "dataset": dataset_raw,
                        "model": model_raw,
                        "condition": cond,
                        "method": method,
                        **sm,
                    })
                    nested_methods[setting_label].setdefault(cond, {})[method] = sm
                    audit_rows.append({
                        "setting": setting_label,
                        "condition": cond,
                        "method": method,
                        "source_file": pth,
                        **audit,
                    })

                baseline_consensus, baseline_mismatches = consensus_baseline(maps)
                # B0 descriptive row from the consensus failure-stage outcomes.
                b0_n = len(baseline_consensus)
                b0_acc = (sum(baseline_consensus.values()) / b0_n) if b0_n else None
                method_metric_rows.append({
                    "setting": setting_label,
                    "dataset": dataset_raw,
                    "model": model_raw,
                    "condition": cond,
                    "method": "b0",
                    "N": b0_n,
                    "before_acc": b0_acc,
                    "after_acc": b0_acc,
                    "delta": 0.0 if b0_n else None,
                    "recovered": 0,
                    "recovery_den": (sum(not x for x in baseline_consensus.values())) if b0_n else 0,
                    "recovery_rate": 0.0 if b0_n else None,
                    "regressed": 0,
                    "regression_den": (sum(baseline_consensus.values())) if b0_n else 0,
                    "regression_rate": 0.0 if b0_n else None,
                    "structural_clearance": 0.0 if b0_n else None,
                    "joint_full_repair": 0.0 if b0_n else None,
                    "new_failure_rate": 0.0 if b0_n else None,
                })
                nested_methods[setting_label].setdefault(cond, {})["b0"] = {
                    "N": b0_n,
                    "after_acc": b0_acc,
                    "structural_clearance": 0.0 if b0_n else None,
                    "joint_full_repair": 0.0 if b0_n else None,
                }
                audit_rows.append({
                    "setting": setting_label,
                    "condition": cond,
                    "method": "baseline_consensus",
                    "source_file": "merged failure_evaluation from fixed/reverse/b2/b3",
                    "raw_record_count": "",
                    "usable_trace_count": b0_n,
                    "missing_trace_id_count": "",
                    "duplicate_trace_id_count": "",
                    "duplicate_trace_ids": "",
                    "baseline_unevaluable_count": "",
                    "final_unevaluable_count": "",
                    "baseline_mismatch_count": baseline_mismatches,
                })

                # Primary Fixed comparisons.
                for comp in PRIMARY_COMPARATORS:
                    bmap = None if comp == "b0" else maps[comp]
                    stats, pair_audit = compare_methods(maps["fixed"], bmap, comp, args.bootstrap, rng)
                    comparison = f"Fixed_vs_{comp.upper() if comp in ('b0','b2','b3') else 'Reverse'}"
                    row = {
                        "comparison": comparison,
                        "setting": setting_label,
                        "dataset": dataset_raw,
                        "model": model_raw,
                        "condition": cond,
                        "A_method": "fixed",
                        "B_method": comp,
                        **stats,
                    }
                    pair_rows.append(row)
                    nested_pairs.setdefault(comparison, {}).setdefault(setting_label, {})[cond] = row
                    audit_rows.append({
                        "setting": setting_label,
                        "condition": cond,
                        "method": comparison,
                        "source_file": "pairwise intersection",
                        **pair_audit,
                    })

                # Supplementary: each implemented method vs B0.
                for method in ANALYZED_METHODS:
                    stats, pair_audit = compare_methods(maps[method], None, "b0", args.bootstrap, rng)
                    comparison = f"{method}_vs_B0"
                    method_vs_b0_rows.append({
                        "comparison": comparison,
                        "setting": setting_label,
                        "dataset": dataset_raw,
                        "model": model_raw,
                        "condition": cond,
                        "A_method": method,
                        "B_method": "b0",
                        **stats,
                    })

    # Multiplicity correction.
    apply_holm_single(single_rows)  # one 12-test family: 3 singles x 4 settings
    apply_holm_to_rows(pair_rows, family_key="comparison")
    apply_holm_to_rows(method_vs_b0_rows, family_key="comparison")

    # Copy adjusted p values back into nested structures.
    for r in single_rows:
        nested_single[r["setting"]][r["condition"]]["holm_p"] = r["holm_p"]
    for r in pair_rows:
        target = nested_pairs[r["comparison"]][r["setting"]][r["condition"]]
        for k in ("holm_p", "recovery_holm_p", "regression_holm_p", "structural_holm_p", "joint_holm_p"):
            target[k] = r.get(k)

    meta = {
        "input_zip": str(input_zip),
        "input_sha256": sha256_file(input_zip),
        "bootstrap_samples": args.bootstrap,
        "bootstrap_seed": args.seed,
        "paired_ci": "percentile paired bootstrap",
        "paired_binary_test": "exact two-sided McNemar/binomial",
        "single_proportion_ci": "Clopper-Pearson exact 95%",
        "multiplicity": "Holm correction within pre-specified comparison/metric families",
        "safe_composer_in_primary_analysis": False,
        "b1_present": False,
        "note": "Contrast-specific intersections; semantic analyses exclude generation failures and semantic judge errors.",
    }
    full = {
        "meta": meta,
        "singles": nested_single,
        "compound_method_metrics": nested_methods,
        "primary_method_contrasts": nested_pairs,
        "supplementary_method_vs_b0": method_vs_b0_rows,
    }

    (output / "statistical_results_full.json").write_text(json.dumps(full, indent=2, ensure_ascii=False), encoding="utf-8")
    (output / "run_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    write_csv(output / "single_repair_stats.csv", single_rows)
    write_csv(output / "compound_method_metrics.csv", method_metric_rows)
    write_csv(output / "primary_pairwise_all_metrics.csv", pair_rows)
    write_csv(output / "all_method_vs_b0.csv", method_vs_b0_rows)
    write_csv(output / "pairing_and_quality_audit.csv", audit_rows)

    semantic_fields = [
        "comparison", "setting", "condition", "N", "A_method", "B_method",
        "A_acc", "B_acc", "delta", "delta_ci", "both_correct", "A_only_correct",
        "B_only_correct", "both_wrong", "mcnemar_p", "holm_p",
    ]
    write_csv(output / "primary_pairwise_semantic.csv", pair_rows, semantic_fields)

    summary_md = generate_markdown_summary(meta, single_rows, pair_rows)
    (output / "paper_ready_summary.md").write_text(summary_md, encoding="utf-8")

    print(f"Analysis complete: {output}")
    print(f"Single tests: {len(single_rows)}")
    print(f"Primary compound comparisons: {len(pair_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
