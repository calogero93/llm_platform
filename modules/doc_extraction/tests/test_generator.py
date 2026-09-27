from decimal import Decimal

import pytest

from doc_extraction.schemas import DiscrepancyType
from doc_extraction.synth.generator import Chain, generate_chain, money
from doc_extraction.synth.piva import is_valid_piva

CHAINS = [generate_chain(42, i) for i in range(120)]


def test_same_seed_and_index_give_identical_chain() -> None:
    assert generate_chain(42, 5) == generate_chain(42, 5)
    assert generate_chain(42, 5) != generate_chain(43, 5)


@pytest.mark.parametrize("chain", CHAINS, ids=lambda c: c.chain_id)
def test_invoice_arithmetic_is_consistent(chain: Chain) -> None:
    inv = chain.invoice
    for line in inv.lines:
        assert line.unit_price is not None
        assert line.total == money(line.quantity * line.unit_price)
    for s in inv.vat_summary:
        bucket = [x.total or Decimal(0) for x in inv.lines if x.vat_rate == s.vat_rate]
        assert s.taxable_amount == sum(bucket, Decimal(0))
        assert s.vat_amount == money(s.taxable_amount * s.vat_rate / 100)
    assert inv.total_amount == inv.taxable_amount + inv.vat_amount  # type: ignore[operator]


@pytest.mark.parametrize("chain", CHAINS, ids=lambda c: c.chain_id)
def test_documents_are_linked_and_parties_valid(chain: Chain) -> None:
    assert chain.ddt.order_refs == (chain.order.number,)
    assert chain.invoice.ddt_refs == (chain.ddt.number,)
    assert chain.order.date < chain.ddt.date <= chain.invoice.date
    assert is_valid_piva(chain.invoice.supplier.vat_number)
    assert is_valid_piva(chain.invoice.customer.vat_number)


@pytest.mark.parametrize("chain", CHAINS, ids=lambda c: c.chain_id)
def test_injected_discrepancies_are_visible_in_the_documents(chain: Chain) -> None:
    order = {x.code: x for x in chain.order.lines}
    ddt = {x.code: x for x in chain.ddt.lines}
    invoice = {x.code: x for x in chain.invoice.lines}
    injected = {(d.type, d.item_code) for d in chain.discrepancies}
    for d in chain.discrepancies:
        match d.type:
            case DiscrepancyType.MISSING_LINE:
                assert d.item_code in order
                assert d.item_code not in ddt
                assert d.item_code not in invoice
            case DiscrepancyType.EXTRA_LINE:
                assert d.item_code in invoice
                assert d.item_code not in order
            case DiscrepancyType.QTY_MISMATCH:
                assert invoice[d.item_code].quantity == d.actual != ddt[d.item_code].quantity
            case DiscrepancyType.PRICE_MISMATCH:
                assert invoice[d.item_code].unit_price == d.actual
                assert order[d.item_code].unit_price == d.expected != d.actual
    # Lines not named in the ground truth must agree everywhere.
    for code, line in invoice.items():
        if not any(item == code for _, item in injected):
            assert line.quantity == ddt[code].quantity
            assert line.unit_price == order[code].unit_price


def test_all_discrepancy_types_and_clean_chains_occur() -> None:
    types = {d.type for c in CHAINS for d in c.discrepancies}
    assert types == set(DiscrepancyType)
    assert any(not c.discrepancies for c in CHAINS)
