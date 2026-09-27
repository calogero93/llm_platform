import pytest

from llmp.config import Settings
from llmp.eval import CaseResult, DatasetRef, FieldCounts, RunReport
from llmp.eval.compare import compare_markdown, gate
from llmp.eval.runner import run_suite
from llmp.llm.mock import MockLLMClient
from llmp.modules import PlatformContext


def test_field_counts_metrics() -> None:
    m = FieldCounts(tp=3, fp=1, fn=2).metrics()
    assert (m.precision, m.recall) == (0.75, 0.6)
    assert m.f1 == pytest.approx(2 * 0.75 * 0.6 / 1.35)


def test_nothing_expected_nothing_predicted_is_perfect() -> None:
    assert FieldCounts().metrics().f1 == 1.0


def test_counts_add() -> None:
    assert FieldCounts(tp=1, fn=1) + FieldCounts(fp=2) == FieldCounts(tp=1, fp=2, fn=1)


class ToySuite:
    name = "toy"
    gated_metrics = ("micro.f1", "scores.exact")

    def dataset(self, split: str) -> DatasetRef:
        return DatasetRef(name="toy", version=1, split=split, manifest_sha256="0" * 64)

    def case_ids(self, split: str) -> list[str]:
        return ["perfect", "partial", "broken"]

    async def run_case(self, split: str, case_id: str, ctx: PlatformContext) -> CaseResult:
        if case_id == "broken":
            raise ValueError("unparseable document")
        if case_id == "perfect":
            return CaseResult(
                case_id=case_id, counts={"a": FieldCounts(tp=2)}, scores={"exact": 1.0}
            )
        return CaseResult(
            case_id=case_id,
            counts={"a": FieldCounts(tp=1, fp=1, fn=1), "b": FieldCounts(fn=1)},
            scores={"exact": 0.0},
        )


@pytest.fixture
def ctx() -> PlatformContext:
    settings = Settings(_env_file=None, llm_backend="mock", llm_model="toy-model")
    return PlatformContext(settings=settings, llm=MockLLMClient("schema", model="toy-model"))


@pytest.fixture
async def report(ctx: PlatformContext) -> RunReport:
    return await run_suite(ToySuite(), "ci", ctx)


def test_runner_aggregates_counts_and_records_errors(report: RunReport) -> None:
    assert (report.n_cases, report.n_errors) == (3, 1)
    assert report.fields["a"].tp == 3  # 2 + 1, broken case excluded
    assert report.micro.tp == 3
    assert (report.micro.fp, report.micro.fn) == (1, 2)
    (broken,) = [c for c in report.cases if c.error]
    assert broken.error == "ValueError: unparseable document"


def test_errored_cases_count_as_zero_in_scores(report: RunReport) -> None:
    assert report.scores["exact"] == pytest.approx(1 / 3)


def test_report_metadata(report: RunReport) -> None:
    assert report.llm_model == "toy-model"
    assert report.dataset.split == "ci"
    assert report.gated_metrics == ["micro.f1", "scores.exact"]
    assert all(c.latency_s >= 0 for c in report.cases)
    assert {"micro.f1", "fields.a.f1", "scores.exact", "n_errors"} <= set(report.flat_metrics())


def degraded(report: RunReport, f1_drop: float) -> RunReport:
    micro = report.micro.model_copy(update={"f1": report.micro.f1 - f1_drop})
    return report.model_copy(update={"micro": micro})


def test_gate_passes_on_identical_run(report: RunReport) -> None:
    assert gate(report, report, tolerance=0.005) == []


def test_gate_fails_on_deliberate_regression(report: RunReport) -> None:
    failures = gate(degraded(report, 0.05), report, tolerance=0.005)
    assert len(failures) == 1
    assert failures[0].startswith("micro.f1")


def test_gate_tolerates_noise_within_tolerance(report: RunReport) -> None:
    assert gate(degraded(report, 0.001), report, tolerance=0.005) == []


def test_gate_fails_when_errors_increase(report: RunReport) -> None:
    worse = report.model_copy(update={"n_errors": report.n_errors + 1})
    assert gate(worse, report, tolerance=0.005) == ["n_errors: 2 > baseline 1"]


def test_compare_markdown_shows_both_runs_and_delta(report: RunReport) -> None:
    table = compare_markdown(report, degraded(report, 0.1))
    (row,) = [line for line in table.splitlines() if line.startswith("| micro.f1 ")]
    assert row.endswith("| -0.1000 |")
