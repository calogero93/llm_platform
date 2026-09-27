from typing import TYPE_CHECKING, Protocol, runtime_checkable

from llmp.eval.types import CaseResult, DatasetRef

if TYPE_CHECKING:
    from llmp.modules import PlatformContext


@runtime_checkable
class EvalSuite(Protocol):
    """A module's evaluation: a dataset split plus a way to predict and score one case."""

    @property
    def name(self) -> str: ...

    @property
    def gated_metrics(self) -> tuple[str, ...]:
        """Dotted names from `RunReport.flat_metrics()` that must not regress (higher is better)."""
        ...

    def dataset(self, split: str) -> DatasetRef: ...

    def case_ids(self, split: str) -> list[str]: ...

    async def run_case(self, split: str, case_id: str, ctx: "PlatformContext") -> CaseResult:
        """Predict and score one case. Exceptions are recorded by the runner as case errors."""
        ...
