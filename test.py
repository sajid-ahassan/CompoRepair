from src.comporepair.models.local_llm import get_local_ollama_llm
import json
from typing import TypedDict
# llm = get_local_ollama_llm()

# prompt = f"""
# say hello in one word
# """
    
# resp = llm.invoke(prompt)
# print(resp)


# with open("data\processed\pilot_baseline_results.json", "r", encoding="utf-8") as file:
#         traces = json.load(file)
#         counter = 0 
#         g_counter = 0
#         correct_and_all_gold = 0
#         count_true_gold_2 = 0
#         coutnt_true_gold_4 = 0
#         for trace in traces:
#             total_evidence = 0
#             present = 0
#             absent = 0
#             supporting = trace["gold_supporting_passage_ids"]
#             c = 0
#             total_evidence += len(supporting)
            
#             for retrieve in trace['retrieval_events']:
#                 if retrieve['passage_id'] in supporting:
#                     present += 1
#                 if retrieve['passage_id'] in supporting:
#                     c += 1
#             if c == len(supporting):
#                 g_counter += 1
                    
#             if trace["evaluation"]['semantic_correct'] == True:
#                 counter += 1
#                 if len(supporting) == 2 and c == len(supporting):
#                     count_true_gold_2 += 1
#                 if len(supporting) == 4 and c == len(supporting):
#                     coutnt_true_gold_4 += 1
#                 if c == len(supporting):
#                     correct_and_all_gold += 1
                
                
# print(f"Number of traces with semantic_correct == True: {counter}")
# print(f"Number of traces with all gold supporting passages: {g_counter}")
# print(f"Number of traces with semantic_correct == True and all gold supporting passages: {correct_and_all_gold}")
# print(f"Number of traces with semantic_correct == True and 2 gold supporting passages: {count_true_gold_2}")
# print(f"Number of traces with semantic_correct == True and 4 gold supporting passages: {coutnt_true_gold_4}")






# from collections import Counter
# import json

# with open(
#     "data/repaired_result/M_D_safe_repair_results.json",
#     "r",
#     encoding="utf-8",
# ) as f:
#     traces = json.load(f)

# counts = Counter(
#     "->".join(
#         t.get("safe_composer_history", {}).get("planned_order", [])
#     )
#     for t in traces
# )

# print(counts)






# import json
# from collections import Counter

# with open("data/results/single/linkage_results.json",
#           encoding="utf-8") as f:
#     traces = json.load(f)

# print(Counter(
#     next(
#         h["link_type"]
#         for h in t["failure_history"]
#         if h["failure"] == "L"
#     )
#     for t in traces
# ))



import json
from collections import Counter

BASELINE_PATH = "data/processed/pilot_baseline_results.json"
SKIPPED_PATH = "data/failures/single/distractor_skipped.json"
OUTPUT_PATH = "data/analysis/hotpot_unsupported_d_questions.json"

with open(BASELINE_PATH, "r", encoding="utf-8") as f:
    baseline = json.load(f)

with open(SKIPPED_PATH, "r", encoding="utf-8") as f:
    skipped = json.load(f)

unsupported_ids = {
    item["trace_id"]
    for item in skipped
    if item["reason"] == "D_no_supported_direct_relation_template"
}

baseline_map = {
    trace["trace_id"]: trace
    for trace in baseline
}

unsupported = []

for trace_id in sorted(unsupported_ids):
    trace = baseline_map.get(trace_id)

    if trace is None:
        continue

    unsupported.append(
        {
            "trace_id": trace_id,
            "question": trace.get("question", ""),
            "canonical_answer": trace.get("canonical_answer", ""),
            "type": (
                trace.get("dataset_metadata", {})
                .get("type", "")
            ),
        }
    )

print(f"Unsupported D questions: {len(unsupported)}")

for index, item in enumerate(unsupported, start=1):
    print(
        f"{index:03d}. "
        f"[{item['type']}] "
        f"{item['question']} "
        f"-> {item['canonical_answer']}"
    )

import os
os.makedirs(
    os.path.dirname(OUTPUT_PATH),
    exist_ok=True,
)

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(
        unsupported,
        f,
        indent=2,
        ensure_ascii=False,
    )

print(f"Saved: {OUTPUT_PATH}")