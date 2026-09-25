# CompoRepair-RAG public statistical analysis package

This public package provides the analysis code, tests, aggregate statistical
outputs, and validation report for the CompoRepair-RAG study.

The trace-level `raw_results.zip` is intentionally not redistributed because it
contains benchmark-derived content. Its SHA-256 identifier is:

```text
18ed827e962204ec7122a7f86eee3cbe521acea299a314ebcde44a197803c497
```

The raw archive may be provided for research verification upon reasonable
request, subject to the source datasets' applicable terms. The included
aggregate outputs can be inspected without the raw archive.

## What it calculates

The program automatically analyzes all four dataset-model settings and the M, D, L, M_D, M_L, D_L, and M_D_L conditions.

Primary compound comparisons:

- Fixed vs B0 (the failure-stage / no-repair state)
- Fixed vs B2
- Fixed vs B3
- Fixed vs Reverse

For each primary paired comparison it calculates:

- contrast-specific paired N
- final semantic accuracy for A and B
- paired accuracy difference
- 10,000-resample paired-bootstrap 95% CI
- both-correct / A-only / B-only / both-wrong counts
- exact two-sided McNemar/binomial p-value
- Holm-adjusted p-value across the 16 setting x compound tests in that contrast family
- paired recovery difference, CI, exact p, and Holm p
- paired regression difference, CI, exact p, and Holm p
- structural-clearance difference, CI, exact p, and Holm p
- Joint Full Repair difference, CI, exact p, and Holm p
- EM and token-F1 differences with paired-bootstrap CIs
- descriptive repair latency and token use when available

For single M/D/L repairs it calculates before/after semantic accuracy, gain, paired-bootstrap CI, recovery, regression, exact McNemar p, Holm p across the 12 single-repair tests, structural clearance, and Joint Full Repair.

## Important analysis decisions

**Safe Composer is excluded from the primary analysis.** The private raw archive contains Safe files, but this program intentionally ignores them for the paper's main statistical families.

**B1 is not analyzed.** There are no B1 trace-level result files in the supplied archive.

**Contrast-specific intersections are used.** A missing or unevaluable B3 trace does not reduce Fixed-vs-Reverse N, for example.

A trace is semantically evaluable only if the failure-stage evaluation and all final evaluations used in that contrast have a Boolean `semantic_correct` value and have neither `generation_failed` nor `semantic_judge_error` set.

`Joint Full Repair` is computed as:

`final semantic correctness AND clearance of every originally injected failure family for that condition`.

For regression comparisons, negative `Fixed - comparator` values favor Fixed because lower regression is better.

## Reproduce the analysis

To rerun the analysis, obtain the authorized trace-level archive and place it in
this directory with the exact filename `raw_results.zip`. Verify that its
SHA-256 equals the value recorded above before continuing.

### Windows

1. Install Python 3.10+.
2. Open Command Prompt in this folder.
3. Run:

```text
pip install -r requirements.txt
run_analysis.bat
```

### Linux/macOS

```text
python3 -m pip install -r requirements.txt
./run_analysis.sh
```

Or run directly:

```text
python analyze_comporepair_stats.py raw_results.zip --output outputs --bootstrap 10000 --seed 20260830
```

The raw ZIP is read directly; it does not need to be extracted first. It is not
included in this public package.

## Output files

- `outputs/statistical_results_full.json` - complete machine-readable results
- `outputs/single_repair_stats.csv` - M/D/L single-repair analysis
- `outputs/compound_method_metrics.csv` - descriptive metrics for B0/B2/B3/Fixed/Reverse
- `outputs/primary_pairwise_semantic.csv` - compact paper-oriented semantic comparison table
- `outputs/primary_pairwise_all_metrics.csv` - complete paired analysis for Fixed vs B0/B2/B3/Reverse
- `outputs/all_method_vs_b0.csv` - supplementary method-vs-failure analyses
- `outputs/pairing_and_quality_audit.csv` - missing/duplicate/evaluation-quality/pairing audit
- `outputs/paper_ready_summary.md` - concise result summary for manuscript drafting
- `outputs/run_metadata.json` - seed, bootstrap count, input SHA-256 and analysis choices

## Statistical procedures

Semantic/effect-size CIs for paired differences use a percentile paired bootstrap. Each bootstrap draw resamples complete trace pairs with replacement, preserving the relationship between methods on the same trace.

The exact paired significance test is the two-sided binomial form of McNemar's test, using only discordant pairs.

Holm correction is applied separately within each pre-specified comparison and metric family. For example, `Fixed_vs_Reverse` semantic accuracy contains 16 tests: 4 settings x 4 compound conditions.

Recovery and regression single-proportion CIs use exact Clopper-Pearson intervals. Paired recovery/regression *differences* use the paired bootstrap.

## Validation

After the analysis finishes, run:

```text
python validate_results.py
```

The included validation checks confirm the main multiplicity-adjusted findings
and the two replicated Qwen M_D order effects for the identified raw-results
archive. The package was validated before publication; see
`outputs/validation_report.txt`.
