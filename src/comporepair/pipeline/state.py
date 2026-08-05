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
    answer_aliases: List[str]

    retrieval_events: List[RetrievalEvent]
    selected_evidence: List[SelectedEvidence]

    answer_claims: List[AnswerClaim]
    verification: VerificationState

    baseline_answer: str
    failure_answer: str
    final_answer: str

    baseline_evaluation: EvaluationState
    failure_evaluation: EvaluationState
    repair_evaluation: EvaluationState
    evaluation: EvaluationState

    predicted_failures: List[str]
    true_failures: List[str]
    failure_history: List[Dict[str, Any]]

    failure_after_repair: List[str]
    new_failures_after_repair: List[str]
    regression_detected: bool

    repair_type: str
    repair_history: List[Dict[str, Any]]
    repair_diagnosis: Dict[str, Any]
    repair_latency_ms: int
    repair_token_usage: Dict[str, int]
    reasoning_constraint: str

    reasoning_plan: Dict[str, Any]
    corrupted_reasoning_plan: Dict[str, Any]
    reasoning_outputs: Dict[str, str]
    failure_reasoning_outputs: Dict[str, str]
    g_removed_dependency: str
    g_corrupted_step_id: str
    g_clean_control_answer: str
    g_clean_control_evaluation: EvaluationState
    g_clean_control_reused_reasoning_outputs: bool
    g_clean_control_latency_ms: int
    g_clean_control_token_usage: Dict[str, int]
    repaired_reasoning_plan: Dict[str, Any]
    g_intervention_active: bool

    model_manifest: Dict[str, Any]
    prompt_hashes: Dict[str, str]
    latency_ms: int
    token_usage: Dict[str, int]


class RAGState(TypedDict, total=False):
    question: str
    retrieved_documents: List[Any]
    context: str
    answer: str
    reasoning_plan: Dict[str, Any]
    reasoning_outputs: Dict[str, str]
    generation_failed: bool
    generation_attempts: int
    latency_ms: int
    token_usage: Dict[str, int]
    prompt_hash: str