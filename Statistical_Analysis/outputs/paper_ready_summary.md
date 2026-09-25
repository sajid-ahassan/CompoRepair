# CompoRepair-RAG Statistical Analysis

This report is generated directly from the raw trace-level JSON files.
Safe Composer is excluded from the primary analysis by design.

## Analysis configuration

- Bootstrap resamples: **10,000**
- Bootstrap seed: **20260830**
- Paired test: **exact two-sided McNemar/binomial test**
- Multiple testing: **Holm family-wise correction** within each pre-specified contrast/metric family
- Pairing: **contrast-specific trace intersections**
- Exclusions: generation failures and semantic-judge errors are not treated as semantic outcomes

## Single-failure repair

| Setting | Failure | N | Before | After | Gain | 95% CI | Recovery | Regression | Holm p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2Wiki-Llama | M | 108 | 55.6% | 64.8% | +9.3 pp | [-0.9 pp, +19.4 pp] | 43.8% | 18.3% | 0.3306 |
| 2Wiki-Llama | D | 108 | 35.2% | 74.1% | +38.9 pp | [+29.6 pp, +48.1 pp] | 60.0% | 0.0% | 4.09e-12 |
| 2Wiki-Llama | L | 107 | 67.3% | 83.2% | +15.9 pp | [+8.4 pp, +23.4 pp] | 51.4% | 1.4% | 6.10e-04 |
| 2Wiki-Qwen | M | 108 | 64.8% | 78.7% | +13.9 pp | [+3.7 pp, +24.1 pp] | 60.5% | 11.4% | 0.0534 |
| 2Wiki-Qwen | D | 108 | 20.4% | 64.8% | +44.4 pp | [+35.2 pp, +53.7 pp] | 55.8% | 0.0% | 7.82e-14 |
| 2Wiki-Qwen | L | 108 | 71.3% | 77.8% | +6.5 pp | [+0.0 pp, +13.9 pp] | 35.5% | 5.2% | 0.3306 |
| Hotpot-Llama | M | 108 | 68.5% | 85.2% | +16.7 pp | [+9.3 pp, +25.0 pp] | 58.8% | 2.7% | 8.48e-04 |
| Hotpot-Llama | D | 108 | 44.4% | 86.1% | +41.7 pp | [+32.4 pp, +50.9 pp] | 75.0% | 0.0% | 5.68e-13 |
| Hotpot-Llama | L | 103 | 86.4% | 94.2% | +7.8 pp | [+1.9 pp, +13.6 pp] | 64.3% | 1.1% | 0.0859 |
| Hotpot-Qwen | M | 108 | 79.6% | 91.7% | +12.0 pp | [+4.6 pp, +19.4 pp] | 72.7% | 3.5% | 0.0266 |
| Hotpot-Qwen | D | 108 | 41.7% | 89.8% | +48.1 pp | [+38.9 pp, +57.4 pp] | 82.5% | 0.0% | 5.33e-15 |
| Hotpot-Qwen | L | 108 | 86.1% | 88.9% | +2.8 pp | [-1.9 pp, +7.4 pp] | 33.3% | 2.2% | 0.4531 |

## Primary compound comparisons

### Fixed vs B0

Semantic accuracy significant after Holm: **16/16** comparisons.

| Setting | Condition | N | A acc | B acc | Delta | 95% CI | A-only | B-only | Holm p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2Wiki-Llama | M_D | 108 | 54.6% | 24.1% | +30.6 pp | [+20.4 pp, +40.7 pp] | 37 | 4 | 2.05e-07 |
| 2Wiki-Llama | M_L | 107 | 80.4% | 42.1% | +38.3 pp | [+28.0 pp, +48.6 pp] | 44 | 3 | 1.73e-09 |
| 2Wiki-Llama | D_L | 108 | 75.9% | 22.2% | +53.7 pp | [+43.5 pp, +63.0 pp] | 59 | 1 | 1.38e-15 |
| 2Wiki-Llama | M_D_L | 107 | 70.1% | 15.0% | +55.1 pp | [+44.9 pp, +65.4 pp] | 61 | 2 | 4.81e-15 |
| 2Wiki-Qwen | M_D | 107 | 62.6% | 11.2% | +51.4 pp | [+41.1 pp, +61.7 pp] | 57 | 2 | 6.14e-14 |
| 2Wiki-Qwen | M_L | 108 | 78.7% | 40.7% | +38.0 pp | [+26.9 pp, +49.1 pp] | 47 | 6 | 1.74e-08 |
| 2Wiki-Qwen | D_L | 108 | 59.3% | 18.5% | +40.7 pp | [+31.5 pp, +50.0 pp] | 45 | 1 | 1.07e-11 |
| 2Wiki-Qwen | M_D_L | 108 | 64.8% | 7.4% | +57.4 pp | [+46.3 pp, +67.6 pp] | 65 | 3 | 4.27e-15 |
| Hotpot-Llama | M_D | 108 | 68.5% | 41.7% | +26.9 pp | [+15.7 pp, +38.0 pp] | 37 | 8 | 1.54e-05 |
| Hotpot-Llama | M_L | 101 | 88.1% | 48.5% | +39.6 pp | [+28.7 pp, +50.5 pp] | 43 | 3 | 2.31e-09 |
| Hotpot-Llama | D_L | 101 | 82.2% | 46.5% | +35.6 pp | [+25.7 pp, +45.5 pp] | 37 | 1 | 1.73e-09 |
| Hotpot-Llama | M_D_L | 104 | 76.9% | 33.7% | +43.3 pp | [+33.7 pp, +52.9 pp] | 46 | 1 | 6.14e-12 |
| Hotpot-Qwen | M_D | 108 | 89.8% | 25.0% | +64.8 pp | [+55.6 pp, +74.1 pp] | 70 | 0 | 2.71e-20 |
| Hotpot-Qwen | M_L | 108 | 89.8% | 53.7% | +36.1 pp | [+25.9 pp, +46.3 pp] | 43 | 4 | 1.11e-08 |
| Hotpot-Qwen | D_L | 108 | 87.0% | 34.3% | +52.8 pp | [+43.5 pp, +62.0 pp] | 57 | 0 | 1.94e-16 |
| Hotpot-Qwen | M_D_L | 108 | 81.5% | 19.4% | +62.0 pp | [+52.8 pp, +71.3 pp] | 67 | 0 | 2.03e-19 |

### Fixed vs B2

Semantic accuracy significant after Holm: **15/16** comparisons.

| Setting | Condition | N | A acc | B acc | Delta | 95% CI | A-only | B-only | Holm p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2Wiki-Llama | M_D | 108 | 54.6% | 51.9% | +2.8 pp | [-11.1 pp, +16.7 pp] | 33 | 30 | 0.8013 |
| 2Wiki-Llama | M_L | 107 | 80.4% | 29.0% | +51.4 pp | [+39.3 pp, +63.6 pp] | 63 | 8 | 1.13e-10 |
| 2Wiki-Llama | D_L | 108 | 75.9% | 58.3% | +17.6 pp | [+4.6 pp, +30.6 pp] | 35 | 16 | 0.0219 |
| 2Wiki-Llama | M_D_L | 107 | 70.1% | 44.9% | +25.2 pp | [+12.1 pp, +38.3 pp] | 42 | 15 | 0.0018 |
| 2Wiki-Qwen | M_D | 107 | 62.6% | 23.4% | +39.3 pp | [+28.0 pp, +50.5 pp] | 47 | 5 | 8.99e-09 |
| 2Wiki-Qwen | M_L | 108 | 78.7% | 28.7% | +50.0 pp | [+38.0 pp, +62.0 pp] | 63 | 9 | 4.18e-10 |
| 2Wiki-Qwen | D_L | 108 | 59.3% | 33.3% | +25.9 pp | [+13.0 pp, +38.0 pp] | 41 | 13 | 8.76e-04 |
| 2Wiki-Qwen | M_D_L | 108 | 64.8% | 16.7% | +48.1 pp | [+37.0 pp, +59.3 pp] | 57 | 5 | 3.68e-11 |
| Hotpot-Llama | M_D | 108 | 68.5% | 46.3% | +22.2 pp | [+10.2 pp, +34.3 pp] | 37 | 13 | 0.0028 |
| Hotpot-Llama | M_L | 101 | 88.1% | 35.6% | +52.5 pp | [+42.6 pp, +62.4 pp] | 54 | 1 | 4.35e-14 |
| Hotpot-Llama | D_L | 101 | 82.2% | 42.6% | +39.6 pp | [+26.7 pp, +52.5 pp] | 49 | 9 | 5.38e-07 |
| Hotpot-Llama | M_D_L | 104 | 76.9% | 33.7% | +43.3 pp | [+31.7 pp, +54.8 pp] | 51 | 6 | 4.54e-09 |
| Hotpot-Qwen | M_D | 108 | 89.8% | 38.9% | +50.9 pp | [+40.7 pp, +61.1 pp] | 58 | 3 | 4.27e-13 |
| Hotpot-Qwen | M_L | 108 | 89.8% | 39.8% | +50.0 pp | [+40.7 pp, +59.3 pp] | 55 | 1 | 2.53e-14 |
| Hotpot-Qwen | D_L | 108 | 87.0% | 44.4% | +42.6 pp | [+31.5 pp, +53.7 pp] | 51 | 5 | 1.05e-09 |
| Hotpot-Qwen | M_D_L | 108 | 81.5% | 28.7% | +52.8 pp | [+42.6 pp, +63.0 pp] | 59 | 2 | 2.53e-14 |

### Fixed vs B3

Semantic accuracy significant after Holm: **15/16** comparisons.

| Setting | Condition | N | A acc | B acc | Delta | 95% CI | A-only | B-only | Holm p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2Wiki-Llama | M_D | 108 | 54.6% | 38.9% | +15.7 pp | [+4.6 pp, +26.9 pp] | 28 | 11 | 0.0284 |
| 2Wiki-Llama | M_L | 107 | 80.4% | 48.6% | +31.8 pp | [+20.6 pp, +42.1 pp] | 39 | 5 | 1.12e-06 |
| 2Wiki-Llama | D_L | 107 | 76.6% | 62.6% | +14.0 pp | [+3.7 pp, +24.3 pp] | 24 | 9 | 0.0284 |
| 2Wiki-Llama | M_D_L | 107 | 70.1% | 30.8% | +39.3 pp | [+28.0 pp, +50.5 pp] | 48 | 6 | 3.58e-08 |
| 2Wiki-Qwen | M_D | 107 | 62.6% | 17.8% | +44.9 pp | [+34.6 pp, +55.1 pp] | 50 | 2 | 7.96e-12 |
| 2Wiki-Qwen | M_L | 108 | 78.7% | 60.2% | +18.5 pp | [+9.3 pp, +27.8 pp] | 25 | 5 | 0.0013 |
| 2Wiki-Qwen | D_L | 108 | 59.3% | 29.6% | +29.6 pp | [+19.4 pp, +39.8 pp] | 35 | 3 | 6.01e-07 |
| 2Wiki-Qwen | M_D_L | 108 | 64.8% | 20.4% | +44.4 pp | [+33.3 pp, +54.7 pp] | 52 | 4 | 1.32e-10 |
| Hotpot-Llama | M_D | 106 | 68.9% | 44.3% | +24.5 pp | [+13.2 pp, +35.8 pp] | 36 | 10 | 7.82e-04 |
| Hotpot-Llama | M_L | 101 | 88.1% | 52.5% | +35.6 pp | [+25.7 pp, +46.5 pp] | 39 | 3 | 5.63e-08 |
| Hotpot-Llama | D_L | 101 | 82.2% | 74.3% | +7.9 pp | [-1.0 pp, +16.8 pp] | 15 | 7 | 0.1338 |
| Hotpot-Llama | M_D_L | 104 | 76.9% | 47.1% | +29.8 pp | [+18.3 pp, +40.4 pp] | 38 | 7 | 2.18e-05 |
| Hotpot-Qwen | M_D | 108 | 89.8% | 28.7% | +61.1 pp | [+51.9 pp, +70.4 pp] | 66 | 0 | 4.34e-19 |
| Hotpot-Qwen | M_L | 108 | 89.8% | 73.1% | +16.7 pp | [+9.3 pp, +25.0 pp] | 20 | 2 | 7.27e-04 |
| Hotpot-Qwen | D_L | 108 | 87.0% | 43.5% | +43.5 pp | [+34.3 pp, +52.8 pp] | 47 | 0 | 1.99e-13 |
| Hotpot-Qwen | M_D_L | 108 | 81.5% | 29.6% | +51.9 pp | [+41.7 pp, +62.0 pp] | 57 | 1 | 6.14e-15 |

### Fixed vs Reverse

Semantic accuracy significant after Holm: **2/16** comparisons.

| Setting | Condition | N | A acc | B acc | Delta | 95% CI | A-only | B-only | Holm p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2Wiki-Llama | M_D | 108 | 54.6% | 55.6% | -0.9 pp | [-7.4 pp, +6.5 pp] | 7 | 8 | 1.0000 |
| 2Wiki-Llama | M_L | 107 | 80.4% | 78.5% | +1.9 pp | [-3.7 pp, +7.5 pp] | 6 | 4 | 1.0000 |
| 2Wiki-Llama | D_L | 107 | 75.7% | 68.2% | +7.5 pp | [-0.9 pp, +15.9 pp] | 15 | 7 | 1.0000 |
| 2Wiki-Llama | M_D_L | 107 | 70.1% | 62.6% | +7.5 pp | [-0.9 pp, +15.9 pp] | 15 | 7 | 1.0000 |
| 2Wiki-Qwen | M_D | 107 | 62.6% | 36.4% | +26.2 pp | [+16.8 pp, +35.5 pp] | 31 | 3 | 1.23e-05 |
| 2Wiki-Qwen | M_L | 108 | 78.7% | 73.1% | +5.6 pp | [+0.0 pp, +11.1 pp] | 8 | 2 | 1.0000 |
| 2Wiki-Qwen | D_L | 108 | 59.3% | 61.1% | -1.9 pp | [-8.3 pp, +4.6 pp] | 6 | 8 | 1.0000 |
| 2Wiki-Qwen | M_D_L | 108 | 64.8% | 53.7% | +11.1 pp | [+2.8 pp, +19.4 pp] | 17 | 5 | 0.2366 |
| Hotpot-Llama | M_D | 106 | 67.9% | 67.9% | +0.0 pp | [-7.5 pp, +7.5 pp] | 8 | 8 | 1.0000 |
| Hotpot-Llama | M_L | 96 | 87.5% | 88.5% | -1.0 pp | [-5.2 pp, +3.1 pp] | 2 | 3 | 1.0000 |
| Hotpot-Llama | D_L | 99 | 81.8% | 79.8% | +2.0 pp | [-5.1 pp, +9.1 pp] | 7 | 5 | 1.0000 |
| Hotpot-Llama | M_D_L | 99 | 79.8% | 76.8% | +3.0 pp | [-3.0 pp, +9.1 pp] | 6 | 3 | 1.0000 |
| Hotpot-Qwen | M_D | 108 | 89.8% | 71.3% | +18.5 pp | [+11.1 pp, +26.9 pp] | 21 | 1 | 1.65e-04 |
| Hotpot-Qwen | M_L | 108 | 89.8% | 88.0% | +1.9 pp | [+0.0 pp, +4.6 pp] | 2 | 0 | 1.0000 |
| Hotpot-Qwen | D_L | 108 | 87.0% | 87.0% | +0.0 pp | [-3.7 pp, +3.7 pp] | 2 | 2 | 1.0000 |
| Hotpot-Qwen | M_D_L | 108 | 81.5% | 79.6% | +1.9 pp | [-2.8 pp, +7.4 pp] | 5 | 3 | 1.0000 |

## Order effects surviving Holm

- **2Wiki-Qwen M_D**: Fixed 62.6% vs Reverse 36.4%, delta +26.2 pp, Holm p=1.23e-05.
- **Hotpot-Qwen M_D**: Fixed 89.8% vs Reverse 71.3%, delta +18.5 pp, Holm p=1.65e-04.

## Interpretation notes

- Positive semantic/recovery/structural/joint deltas favor Fixed.
- For regression, **negative** Fixed-minus-comparator deltas favor Fixed because lower regression is better.
- `joint_full_repair` means final semantic correctness AND clearance of all originally injected failure families for that condition.
- B0 is the failure-stage state, not a newly generated answer.
- B1 is not included because no B1 trace-level result files are present in this archive.
