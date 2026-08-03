python -m src.comporepair.build_base_traces 
 -- python -m src.comporepair.retrieval.build_vector_store
python -m src.comporepair.run_baseline
python -m src.comporepair.failure_injection
python -m src.comporepair.run_single_failure
python -m src.comporepair.run_compound_failure

 python -m src.comporepair.analysis.analyze_failures
 python -m src.comporepair.run_repair 
  python -m src.comporepair.analysis.analyze_repair







python -m src.comporepair.retrieval.build_vector_store
python -m src.comporepair.run_baseline

Failure
python -m src.comporepair.failure_injection
python -m src.comporepair.run_failure
python -m src.comporepair.run_g_failure
python -m src.comporepair.run_d_failure


validate trace
python -m src.comporepair.validate_trace

git fetch origin
git reset --hard origin/main
git clean -fd


C:\Users\sajid\OneDrive\Desktop\CompoRepair\src\comporepair\run_single_failure.py