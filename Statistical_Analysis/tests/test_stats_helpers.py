import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPT = HERE.parent / "analyze_comporepair_stats.py"
spec = importlib.util.spec_from_file_location("comporepair_stats_module", SCRIPT)
stats = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = stats
spec.loader.exec_module(stats)


def test_exact_mcnemar_known_case():
    p = stats.exact_mcnemar_p(21, 1)
    assert 0 < p < 0.001


def test_holm_monotonic_and_bounded():
    p = [0.001, 0.01, 0.04, 0.2]
    adj = stats.holm_adjust(p)
    assert all(0 <= x <= 1 for x in adj)
    assert adj[0] <= adj[1] <= adj[2] <= adj[3]


def test_clopper_pearson_zero_successes():
    lo, hi = stats.clopper_pearson(0, 38)
    assert lo == 0.0
    assert 0 < hi < 0.2
