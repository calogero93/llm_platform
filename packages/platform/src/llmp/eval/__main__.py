"""Eval CLI.

    python -m llmp.eval run SUITE [--split ci] [--out evals/runs]
    python -m llmp.eval compare REPORT_A REPORT_B
    python -m llmp.eval gate REPORT BASELINE [--tolerance 0.005]

The LLM backend comes from the usual `LLMP_*` settings (mock by default in CI).
"""

import argparse
import asyncio
import sys
from pathlib import Path

from llmp.config import Settings
from llmp.eval.compare import compare_markdown, gate
from llmp.eval.runner import run_suite
from llmp.eval.suite import EvalSuite
from llmp.eval.types import RunReport
from llmp.llm.factory import build_llm_client
from llmp.logs import configure_logging
from llmp.modules import PlatformContext, discover_modules
from llmp.tracing import build_tracer_provider


def _find_suite(name: str) -> EvalSuite:
    suites = {s.name: s for m in discover_modules() for s in m.eval_suites()}
    if name not in suites:
        raise SystemExit(f"unknown suite {name!r}; available: {', '.join(sorted(suites))}")
    return suites[name]


async def _run(suite_name: str, split: str, out: Path) -> RunReport:
    settings = Settings()
    configure_logging(settings.log_level)
    provider = build_tracer_provider(settings)
    ctx = PlatformContext(
        settings=settings, llm=build_llm_client(settings, provider.get_tracer("llmp.eval"))
    )
    try:
        report = await run_suite(_find_suite(suite_name), split, ctx)
    finally:
        await ctx.llm.aclose()
        provider.shutdown()
    path = out / report.run_id / "report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.model_dump_json(indent=2) + "\n")
    print(
        f"{report.suite} [{split}] micro F1={report.micro.f1:.4f} "
        f"errors={report.n_errors}/{report.n_cases} p50={report.latency_p50_s:.3f}s"
    )
    print(path)
    return report


def _load(path: Path) -> RunReport:
    return RunReport.model_validate_json(path.read_text())


def main() -> int:
    parser = argparse.ArgumentParser(prog="llmp.eval")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("suite")
    run.add_argument("--split", default="ci")
    run.add_argument("--out", type=Path, default=Path("evals/runs"))
    cmp = sub.add_parser("compare")
    cmp.add_argument("a", type=Path)
    cmp.add_argument("b", type=Path)
    gt = sub.add_parser("gate")
    gt.add_argument("report", type=Path)
    gt.add_argument("baseline", type=Path)
    gt.add_argument("--tolerance", type=float, default=0.005)
    args = parser.parse_args()

    if args.command == "run":
        report = asyncio.run(_run(args.suite, args.split, args.out))
        return 1 if report.n_errors else 0
    if args.command == "compare":
        print(compare_markdown(_load(args.a), _load(args.b)))
        return 0
    failures = gate(_load(args.report), _load(args.baseline), args.tolerance)
    for failure in failures:
        print(f"REGRESSION {failure}", file=sys.stderr)
    if not failures:
        print("gate passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
