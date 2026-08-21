from typing import Any, Dict, List, TypedDict


class RetrievalEvent(TypedDict, total=False):
    title: str
    passage_id: str
    rank: int
    text: str
    score: float
    is_supporting: bool
    document_role: str


class SelectedEvidence(TypedDict, total=False):
    passage_id: str
    rank: int
    observable_signals: Dict[str, Any]
    title: str
    text: str
    is_supporting: bool
    document_role: str


class AnswerClaim(TypedDict, total=False):
    claim: str
    source: str
    evidence_ids: List[str]


class VerificationState(TypedDict, total=False):
    support: float
    completeness: float
    conflict: float
    decision: str


class EvaluationState(TypedDict, total=False):
    predicted_answer: str
    canonical_answer: str
    exact_match: bool
    token_f1: float
    semantic_correct: bool
    generation_failed: bool
    semantic_judge_error: bool


class CompoRepairTraceState(TypedDict, total=False):
    trace_id: str
    question_id: str
    dataset: str
    partition: str
    experiment_stage: str
    dataset_metadata: Dict[str, Any]

    question: str
    canonical_answer: str
    gold_supporting_passage_ids: List[str]
    baseline_evaluation: EvaluationState
    answer_aliases: List[str]

    retrieval_events: List[RetrievalEvent]
    selected_evidence: List[SelectedEvidence]

    answer_claims: List[AnswerClaim]
    verification: VerificationState

    baseline_answer: str
    failure_answer: str
    final_answer: str

    failure_evaluation: EvaluationState
    repair_evaluation: EvaluationState
    evaluation: EvaluationState

    predicted_failures: List[str]
    true_failures: List[str]
    failure_history: List[Dict[str, Any]]

    failure_after_repair: List[str]
    new_failures_after_repair: List[str]
    semantic_regression_detected: bool
    structural_regression_detected: bool
    regression_detected: bool

    l_target_passage_id: str
    l_source_passage_ids: List[str]
    l_original_link_phrase: str
    l_replacement_phrase: str
    l_original_passage_text: str
    l_corrupted_passage_text: str

    repair_type: str
    repair_mode: str
    repair_order: List[str]
    repair_history: List[Dict[str, Any]]
    repair_diagnosis: Dict[str, Any]
    repair_latency_ms: int
    repair_token_usage: Dict[str, int]
    safe_composer_history: Dict[str, Any]

    model_manifest: Dict[str, Any]
    prompt_hashes: Dict[str, str]
    latency_ms: int
    token_usage: Dict[str, int]


class RAGState(TypedDict, total=False):
    question: str
    retrieved_documents: List[Any]
    context: str
    answer: str
    generation_failed: bool
    generation_attempts: int
    latency_ms: int
    token_usage: Dict[str, int]
    prompt_hash: str
