"""Field-level comparison of a predicted `Document` against ground truth."""

import difflib
from collections.abc import Callable
from datetime import date
from decimal import Decimal

from doc_extraction.schemas import Document, LineItem
from llmp.eval import FieldCounts

Value = str | Decimal | date | tuple[str, ...] | None

HEADER_FIELDS: dict[str, Callable[[Document], Value]] = {
    "number": lambda d: d.number,
    "date": lambda d: d.date,
    "supplier.name": lambda d: d.supplier.name,
    "supplier.vat_number": lambda d: d.supplier.vat_number,
    "customer.name": lambda d: d.customer.name,
    "customer.vat_number": lambda d: d.customer.vat_number,
    "order_refs": lambda d: d.order_refs or None,
    "ddt_refs": lambda d: d.ddt_refs or None,
    "taxable_amount": lambda d: d.taxable_amount,
    "vat_amount": lambda d: d.vat_amount,
    "total_amount": lambda d: d.total_amount,
}

LINE_FIELDS: dict[str, Callable[[LineItem], Value]] = {
    "line.code": lambda x: x.code,
    "line.description": lambda x: x.description,
    "line.quantity": lambda x: x.quantity,
    "line.unit": lambda x: x.unit,
    "line.unit_price": lambda x: x.unit_price,
    "line.total": lambda x: x.total,
    "line.vat_rate": lambda x: x.vat_rate,
}

MIN_DESCRIPTION_SIMILARITY = 0.6


def normalize(value: Value) -> Value:
    """Case/whitespace-insensitive strings, order-insensitive references; Decimal/date as is
    (Decimal equality already ignores trailing zeros: 1.20 == 1.2)."""
    if isinstance(value, str):
        return " ".join(value.casefold().split())
    if isinstance(value, tuple):
        return tuple(sorted(" ".join(v.casefold().split()) for v in value))
    return value


def count(gold: Value, pred: Value) -> FieldCounts:
    gold, pred = normalize(gold), normalize(pred)
    if gold is None:
        return FieldCounts(fp=int(pred is not None))
    if pred is None:
        return FieldCounts(fn=1)
    return FieldCounts(tp=1) if gold == pred else FieldCounts(fp=1, fn=1)


def align_lines(
    gold: tuple[LineItem, ...], pred: tuple[LineItem, ...]
) -> tuple[list[tuple[LineItem, LineItem]], list[LineItem], list[LineItem]]:
    """Pair lines by article code, then exact description, then most similar description.

    Returns (pairs, unmatched gold, unmatched predicted). Greedy, which is optimal enough for
    invoices of tens of lines with mostly unique codes.
    """
    free_gold, free_pred = list(gold), list(pred)
    pairs: list[tuple[LineItem, LineItem]] = []

    def take(key: Callable[[LineItem], Value]) -> None:
        for g in list(free_gold):
            k = normalize(key(g))
            match = next((p for p in free_pred if k is not None and normalize(key(p)) == k), None)
            if match is not None:
                pairs.append((g, match))
                free_gold.remove(g)
                free_pred.remove(match)

    take(lambda x: x.code)
    take(lambda x: x.description)
    candidates = sorted(
        (
            (
                difflib.SequenceMatcher(
                    None, str(normalize(g.description)), str(normalize(p.description))
                ).ratio(),
                gi,
                pi,
            )
            for gi, g in enumerate(free_gold)
            for pi, p in enumerate(free_pred)
        ),
        reverse=True,
    )
    used_g: set[int] = set()
    used_p: set[int] = set()
    for ratio, gi, pi in candidates:
        if ratio < MIN_DESCRIPTION_SIMILARITY:
            break
        if gi not in used_g and pi not in used_p:
            pairs.append((free_gold[gi], free_pred[pi]))
            used_g.add(gi)
            used_p.add(pi)
    unmatched_gold = [g for i, g in enumerate(free_gold) if i not in used_g]
    unmatched_pred = [p for i, p in enumerate(free_pred) if i not in used_p]
    return pairs, unmatched_gold, unmatched_pred


def compare_documents(gold: Document, pred: Document) -> dict[str, FieldCounts]:
    counts = {name: count(get(gold), get(pred)) for name, get in HEADER_FIELDS.items()}
    line_counts = dict.fromkeys(LINE_FIELDS, FieldCounts())
    pairs, missed, spurious = align_lines(gold.lines, pred.lines)
    for g, p in pairs:
        for name, get in LINE_FIELDS.items():
            line_counts[name] += count(get(g), get(p))
    for g in missed:
        for name, get in LINE_FIELDS.items():
            line_counts[name] += count(get(g), None)
    for p in spurious:
        for name, get in LINE_FIELDS.items():
            line_counts[name] += count(None, get(p))
    return counts | line_counts
