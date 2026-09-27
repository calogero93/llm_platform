"""Domain model shared by the synthetic generator (ground truth), parsers and eval metrics."""

from datetime import date
from decimal import Decimal
from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True)


class DocumentKind(StrEnum):
    ORDER = "order"
    DDT = "ddt"
    INVOICE = "invoice"


class Party(_Frozen):
    name: str
    vat_number: str
    """Italian partita IVA: 11 digits, without the `IT` prefix."""
    address: str
    postal_code: str
    city: str
    province: str


class LineItem(_Frozen):
    line_number: int
    code: str | None
    description: str
    quantity: Decimal
    unit: str | None
    unit_price: Decimal | None
    """None on documents without prices (DDT)."""
    total: Decimal | None
    vat_rate: Decimal | None
    """Percentage, e.g. 22.00. Invoices only."""


class VatSummary(_Frozen):
    vat_rate: Decimal
    taxable_amount: Decimal
    vat_amount: Decimal


class Document(_Frozen):
    kind: DocumentKind
    number: str
    date: date
    supplier: Party
    customer: Party
    lines: tuple[LineItem, ...]
    order_refs: tuple[str, ...] = ()
    """Purchase order numbers this document refers to (DDT, invoice)."""
    ddt_refs: tuple[str, ...] = ()
    """Delivery note numbers an invoice refers to."""
    vat_summary: tuple[VatSummary, ...] = ()
    taxable_amount: Decimal | None = None
    vat_amount: Decimal | None = None
    total_amount: Decimal | None = None


class DiscrepancyType(StrEnum):
    QTY_MISMATCH = "qty_mismatch"
    """Invoiced quantity differs from the delivered quantity."""
    PRICE_MISMATCH = "price_mismatch"
    """Invoiced unit price differs from the ordered unit price."""
    MISSING_LINE = "missing_line"
    """Ordered line never delivered nor invoiced."""
    EXTRA_LINE = "extra_line"
    """Invoiced line that was neither ordered nor delivered."""


class Discrepancy(_Frozen):
    type: DiscrepancyType
    item_code: str
    expected: Decimal | None = None
    actual: Decimal | None = None
