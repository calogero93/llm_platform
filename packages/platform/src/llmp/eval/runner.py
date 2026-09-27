import asyncio
import shutil
import statistics
import subprocess
import time
from datetime import UTC, datetime

from llmp.eval.suite import EvalSuite
from llmp.eval.types import CaseResult, FieldCounts, RunReport
from llmp.modules import PlatformContext


async def run_suite(
    suite: EvalSuite, split: str, ctx: PlatformContext, concurrency: int = 4
) -> RunReport:
    semaphore = asyncio.Semaphore(concurrency)

    async def one(case_id: str) -> CaseResult:
        async with semaphore:
            start = time.perf_counter()
            try:
                result = await suite.run_case(split, case_id, ctx)
            except Exception as e:  # recorded in the report, never silently dropped
                result = CaseResult(case_id=case_id, error=f"{type(e).__name__}: {e}")
            return result.model_copy(update={"latency_s": time.perf_counter() - start})

    cases = await asyncio.gather(*(one(c) for c in suite.case_ids(split)))
    return build_report(suite, split, ctx, list(cases))


def build_report(
    suite: EvalSuite, split: str, ctx: PlatformContext, cases: list[CaseResult]
) -> RunReport:
    ok = [c for c in cases if c.error is None]
    totals: dict[str, FieldCounts] = {}
    for case in ok:
        for field, counts in case.counts.items():
            totals[field] = totals.get(field, FieldCounts()) + counts
    micro = sum(totals.values(), FieldCounts())
    score_names = sorted({name for c in ok for name in c.scores})
    latencies = sorted(c.latency_s for c in cases) or [0.0]
    prompts = {p.hash: p for c in ok for p in c.prompts}
    now = datetime.now(UTC)
    return RunReport(
        run_id=f"{now:%Y%m%dT%H%M%SZ}_{suite.name}_{split}",
        suite=suite.name,
        created_at=now.isoformat(),
        git_sha=_git_sha(),
        dataset=suite.dataset(split),
        llm_backend=ctx.settings.llm_backend,
        llm_model=ctx.settings.llm_model,
        prompts=sorted(prompts.values(), key=lambda p: (p.id, p.version)),
        gated_metrics=list(suite.gated_metrics),
        fields={k: v.metrics() for k, v in sorted(totals.items())},
        micro=micro.metrics(),
        # Cases that errored score 0: an error must never make a run look better.
        scores={n: sum(c.scores.get(n, 0.0) for c in ok) / len(cases) for n in score_names},
        latency_p50_s=statistics.median(latencies),
        latency_p95_s=latencies[min(len(latencies) - 1, round(0.95 * (len(latencies) - 1)))],
        n_cases=len(cases),
        n_errors=len(cases) - len(ok),
        cases=cases,
    )


def _git_sha() -> str:
    git = shutil.which("git")
    if git is None:
        return "unknown"  # e.g. running inside a container without git
    try:
        # Constant arguments and a resolved git path: no untrusted input reaches the command.
        sha = subprocess.run(  # noqa: S603
            [git, "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(  # noqa: S603
            [git, "status", "--porcelain"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except subprocess.CalledProcessError:
        return "unknown"  # not a git checkout
    return f"{sha}-dirty" if dirty else sha
