from pydantic import BaseModel, ConfigDict

from llmp.llm.types import PromptRef


class FieldCounts(BaseModel):
    """Extraction outcome counts for one field. A wrong value counts as both FP and FN."""

    model_config = ConfigDict(frozen=True)

    tp: int = 0
    fp: int = 0
    fn: int = 0

    def __add__(self, other: "FieldCounts") -> "FieldCounts":
        return FieldCounts(tp=self.tp + other.tp, fp=self.fp + other.fp, fn=self.fn + other.fn)

    def metrics(self) -> "FieldMetrics":
        # Conventions: nothing predicted and nothing expected → perfect score (1.0).
        precision = self.tp / (self.tp + self.fp) if self.tp + self.fp else 1.0
        recall = self.tp / (self.tp + self.fn) if self.tp + self.fn else 1.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return FieldMetrics(
            tp=self.tp, fp=self.fp, fn=self.fn, precision=precision, recall=recall, f1=f1
        )


class FieldMetrics(BaseModel):
    model_config = ConfigDict(frozen=True)

    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float


class DatasetRef(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    version: int
    split: str
    manifest_sha256: str


class CaseResult(BaseModel):
    case_id: str
    counts: dict[str, FieldCounts] = {}
    scores: dict[str, float] = {}
    """Per-case scores averaged over the run (e.g. exact_match)."""
    prompts: list[PromptRef] = []
    latency_s: float = 0.0
    error: str | None = None


class RunReport(BaseModel):
    run_id: str
    suite: str
    created_at: str
    git_sha: str
    dataset: DatasetRef
    llm_backend: str
    llm_model: str
    prompts: list[PromptRef]
    gated_metrics: list[str]
    fields: dict[str, FieldMetrics]
    micro: FieldMetrics
    scores: dict[str, float]
    latency_p50_s: float
    latency_p95_s: float
    n_cases: int
    n_errors: int
    cases: list[CaseResult]

    def flat_metrics(self) -> dict[str, float]:
        """Every comparable number under a stable dotted name, e.g. `fields.number.f1`."""
        flat = {
            "micro.precision": self.micro.precision,
            "micro.recall": self.micro.recall,
            "micro.f1": self.micro.f1,
            "latency.p50_s": self.latency_p50_s,
            "latency.p95_s": self.latency_p95_s,
            "n_errors": float(self.n_errors),
        }
        flat.update({f"scores.{k}": v for k, v in self.scores.items()})
        flat.update({f"fields.{k}.f1": m.f1 for k, m in self.fields.items()})
        return flat
