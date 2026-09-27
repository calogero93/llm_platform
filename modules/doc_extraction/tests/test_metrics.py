from decimal import Decimal

from doc_extraction.metrics import align_lines, compare_documents, count
from doc_extraction.synth.generator import generate_chain
from llmp.eval import FieldCounts

GOLD = generate_chain(42, 1).invoice


def test_count_semantics() -> None:
    assert count("A", "a ") == FieldCounts(tp=1)  # case/whitespace-insensitive
    assert count(Decimal("1.20"), Decimal("1.2")) == FieldCounts(tp=1)
    assert count("A", "B") == FieldCounts(fp=1, fn=1)
    assert count("A", None) == FieldCounts(fn=1)
    assert count(None, "B") == FieldCounts(fp=1)
    assert count(None, None) == FieldCounts()
    assert count(("b", "a"), ("A", "B")) == FieldCounts(tp=1)


def test_identical_documents_are_all_true_positives() -> None:
    counts = compare_documents(GOLD, GOLD)
    assert all(c.fp == 0 and c.fn == 0 for c in counts.values())


def test_hand_computed_errors() -> None:
    lines = list(GOLD.lines)
    wrong_price = lines[0].model_copy(update={"unit_price": Decimal("999.99")})
    pred = GOLD.model_copy(
        update={"number": "WRONG", "total_amount": None, "lines": (wrong_price, *lines[2:])}
    )
    counts = compare_documents(GOLD, pred)
    assert counts["number"] == FieldCounts(fp=1, fn=1)
    assert counts["total_amount"] == FieldCounts(fn=1)
    # line 0 matched with a wrong price; line 1 missing entirely (7 line fields, all fn).
    n = len(lines)
    assert counts["line.unit_price"] == FieldCounts(tp=n - 2, fp=1, fn=2)
    assert counts["line.code"] == FieldCounts(tp=n - 1, fn=1)


def test_lines_align_by_code_then_description() -> None:
    a, b = GOLD.lines[0], GOLD.lines[1]
    no_code = b.model_copy(update={"code": None, "description": b.description + " (rif.)"})
    pairs, missed, spurious = align_lines((a, b), (no_code, a))
    assert {(g.line_number, p.code) for g, p in pairs} == {
        (a.line_number, a.code),
        (b.line_number, None),
    }
    assert (missed, spurious) == ([], [])
