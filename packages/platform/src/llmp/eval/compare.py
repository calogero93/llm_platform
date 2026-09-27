from llmp.eval.types import RunReport


def compare_markdown(a: RunReport, b: RunReport) -> str:
    """Side-by-side table of every metric of two runs (e.g. model A vs B, prompt v1 vs v2)."""
    fa, fb = a.flat_metrics(), b.flat_metrics()
    lines = [
        f"| metric | A: {_label(a)} | B: {_label(b)} | Δ (B-A) |",
        "|---|---:|---:|---:|",
    ]
    for name in sorted(set(fa) | set(fb)):
        va, vb = fa.get(name), fb.get(name)
        delta = f"{vb - va:+.4f}" if va is not None and vb is not None else "n/a"
        lines.append(f"| {name} | {_fmt(va)} | {_fmt(vb)} | {delta} |")
    return "\n".join(lines)


def gate(current: RunReport, baseline: RunReport, tolerance: float) -> list[str]:
    """Return one message per gated metric that regressed beyond `tolerance`; empty = pass."""
    cur, base = current.flat_metrics(), baseline.flat_metrics()
    failures = []
    for name in baseline.gated_metrics:
        if name not in cur:
            failures.append(f"{name}: missing from current run")
        elif cur[name] < base[name] - tolerance:
            failures.append(f"{name}: {cur[name]:.4f} < baseline {base[name]:.4f} - {tolerance}")
    if current.n_errors > baseline.n_errors:
        failures.append(f"n_errors: {current.n_errors} > baseline {baseline.n_errors}")
    return failures


def _label(r: RunReport) -> str:
    return f"{r.llm_model} @ {r.git_sha}"


def _fmt(v: float | None) -> str:
    return "—" if v is None else f"{v:.4f}"
