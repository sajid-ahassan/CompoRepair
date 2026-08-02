
python -m src.comporepair.build_base_traces
python -m src.comporepair.retrieval.build_vector_store
python -m src.comporepair.run_baseline

Failure
python -m src.comporepair.failure_injection
python -m src.comporepair.run_failure
python -m src.comporepair.run_g_failure
python -m src.comporepair.run_d_failure


git fetch origin
git reset --hard origin/main
git clean -fd


