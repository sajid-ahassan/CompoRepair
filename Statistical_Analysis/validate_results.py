#!/usr/bin/env python3
"""Sanity checks for the generated analysis output."""
import csv
import math
from pathlib import Path

BASE = Path(__file__).resolve().parent / "outputs"
rows = list(csv.DictReader((BASE / "primary_pairwise_semantic.csv").open(encoding="utf-8")))
singles = list(csv.DictReader((BASE / "single_repair_stats.csv").open(encoding="utf-8")))


def row(comp, setting, cond):
    return next(r for r in rows if r["comparison"] == comp and r["setting"] == setting and r["condition"] == cond)


def close(a, b, tol=1e-12):
    return abs(float(a) - float(b)) <= tol

checks = []
checks.append(("Fixed vs B0 significant 16/16", sum(float(r["holm_p"]) < 0.05 for r in rows if r["comparison"] == "Fixed_vs_B0") == 16))
checks.append(("Fixed vs B2 significant 15/16", sum(float(r["holm_p"]) < 0.05 for r in rows if r["comparison"] == "Fixed_vs_B2") == 15))
checks.append(("Fixed vs B3 significant 15/16", sum(float(r["holm_p"]) < 0.05 for r in rows if r["comparison"] == "Fixed_vs_B3") == 15))
checks.append(("Fixed vs Reverse significant 2/16", sum(float(r["holm_p"]) < 0.05 for r in rows if r["comparison"] == "Fixed_vs_Reverse") == 2))
checks.append(("Singles significant 7/12", sum(float(r["holm_p"]) < 0.05 for r in singles) == 7))

r = row("Fixed_vs_Reverse", "2Wiki-Qwen", "M_D")
checks.append(("2Wiki-Qwen M_D order N=107", int(r["N"]) == 107))
checks.append(("2Wiki-Qwen M_D order delta=0.261682...", close(r["delta"], 0.2616822429906542)))
checks.append(("2Wiki-Qwen M_D order discordant=31 vs 3", int(r["A_only_correct"]) == 31 and int(r["B_only_correct"]) == 3))
checks.append(("2Wiki-Qwen M_D Holm p=1.2256205e-05", close(r["holm_p"], 1.2256205081939697e-05)))

r = row("Fixed_vs_Reverse", "Hotpot-Qwen", "M_D")
checks.append(("Hotpot-Qwen M_D order delta=0.185185...", close(r["delta"], 0.18518518518518523)))
checks.append(("Hotpot-Qwen M_D order discordant=21 vs 1", int(r["A_only_correct"]) == 21 and int(r["B_only_correct"]) == 1))

failed = [name for name, ok in checks if not ok]
for name, ok in checks:
    print(("PASS" if ok else "FAIL") + " - " + name)

if failed:
    raise SystemExit(1)
print(f"\nAll {len(checks)} validation checks passed.")
