


MDG
python -m src.comporepair.build_2wiki_base_trace 
python -m src.comporepair.build_musique_base_trace 
python -m src.comporepair.build_base_traces
python -m src.comporepair.retrieval.build_vector_store

python -m src.comporepair.run_baseline

python -m src.comporepair.failure_injection
python -m src.comporepair.select_final_experiment_traces
python -m src.comporepair.run_single_failure
python -m src.comporepair.run_compound_failure

python -m src.comporepair.analysis.analyze_failures

python -m src.comporepair.run_repair
python -m src.comporepair.run_safe

python -m src.comporepair.b2_b3

python -m src.comporepair.analysis.analyze_repair

 

validate trace
python -m src.comporepair.validate_trace

git fetch origin
git reset --hard origin/main
git clean -fd

