"""Request and response models for the public API."""
from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=2, max_length=1000)
    stream: bool = False


class Citation(BaseModel):
    node_id: str
    path: str = Field(..., description="Breadcrumb, e.g. '234-qaror > 2-ilova > 3-bob'")
    url: str
    snippet: str


class AskResponse(BaseModel):
    answer: str
    grounded: bool = Field(..., description="False when the refusal was returned")
    citations: list[Citation] = []
    quotes: list[str] = Field(
        default=[], description="Verbatim spans verified against the retrieved text")
    refusal_reason: str | None = Field(
        default=None, description="Which gate refused: retrieval|sufficiency|quote|citation")
    latency_ms: int = 0


class SearchHit(BaseModel):
    node_id: str
    path: str
    url: str
    text: str
    rerank_score: float
    rrf_score: float


class SearchResponse(BaseModel):
    query: str
    variants: list[str]
    hits: list[SearchHit]
    latency_ms: int


class HealthResponse(BaseModel):
    status: str
    device: str
    ollama_host: str
    ollama_model: str
    ollama_reachable: bool
    ollama_model_present: bool
    embed_model_cached: bool
    rerank_model_cached: bool
    index_chunks: int
    corpus_hash_ok: bool
