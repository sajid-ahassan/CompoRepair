from typing import TypedDict, List, Dict, Any

class RetrievalEvent(TypedDict):
    query: str
    passage_ids: List[str]
    scores: List[float]

class SelectedEvidence(TypedDict):
    passage_id: str
    rank: int
    observable_signals: Dict[str, Any]
    title: str 
    text: str

class AnswerClaim(TypedDict):
    claim_id: str
    text: str
    evidence_ids: List[str]

class VerificationState(TypedDict):
    support: float
    completeness: float
    conflict: float
    decision: str

class CompoRepairTraceState(TypedDict):
    trace_id: str
    question_id: str
    dataset: str
    partition: str
    question: str
    canonical_answer: str 
    answer_aliases: List[str]
    retrieval_events: List[RetrievalEvent]
    selected_evidence: List[SelectedEvidence]
    answer_claims: List[AnswerClaim]
    verification: VerificationState
    final_answer: str
    predicted_failures: List[str]
    true_failures: List[str] 
    repair_history: List[Dict[str, Any]]
    model_manifest: Dict[str, Any]
    prompt_hashes: Dict[str, str]
    latency_ms: int
    token_usage: Dict[str, int]