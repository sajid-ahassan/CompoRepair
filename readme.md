
python -m src.comporepair.build_base_traces
python -m src.comporepair.retrieval.build_vector_store
python -m src.comporepair.run_baseline


git fetch origin
git reset --hard origin/main
git clean -fd


