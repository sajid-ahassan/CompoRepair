$ErrorActionPreference = "Stop"
python -m src.comporepair.run_single_failure
python -m src.comporepair.run_compound_failure
python -m src.comporepair.analysis.analyze_failures
python -m src.comporepair.run_repair
python -m src.comporepair.run_safe
python -m src.comporepair.b2_b3
python -m src.comporepair.analysis.analyze_repair