"""Coherent order → DDT → invoice chains with injected discrepancies as ground truth.

Each chain depends only on `(seed, index)`, so chain k is identical whether 30 or 300 chains
are generated: the `ci` split is a prefix of the `full` split.
"""

import random
from datetime import date, timedelta
from decimal import ROUND_HALF_UP, Decimal

from pydantic import BaseModel, ConfigDict

from doc_extraction.schemas import (
    Discrepancy,
    DiscrepancyType,
    Document,
    DocumentKind,
    LineItem,
    Party,
    VatSummary,
)
from doc_extraction.synth.data import (
    CITIES,
    LEGAL_FORMS,
    NAME_FIRST,
    NAME_SECOND,
    PRODUCTS,
    STREETS,
    Product,
)
from doc_extraction.synth.piva import random_piva

CENT = Decimal("0.01")
FIRST_ORDER_DATE = date(2026, 1, 5)


class Chain(BaseModel):
    model_config = ConfigDict(frozen=True)

    chain_id: str
    order: Document
    ddt: Document
    invoice: Document
    discrepancies: tuple[Discrepancy, ...]


class _Line(BaseModel):
    """Mutable working line used while building the three documents."""

    product: Product
    quantity: Decimal
    unit_price: Decimal


def money(value: Decimal) -> Decimal:
    return value.quantize(CENT, rounding=ROUND_HALF_UP)


def generate_chain(seed: int, index: int) -> Chain:
    rng = random.Random(f"{seed}:{index}")  # str seeds are hashed with SHA-512: stable
    supplier, customer = _party(rng), _party(rng)
    order_date = FIRST_ORDER_DATE + timedelta(days=rng.randrange(0, 240))
    ddt_date = order_date + timedelta(days=rng.randint(2, 10))
    invoice_date = ddt_date + timedelta(days=rng.randint(0, 20))
    year = order_date.year
    order_no = rng.choice(
        [f"OA/{year}/{index + 1:04d}", f"ORD-{index + 101}", f"{index + 1}/{year}"]
    )
    ddt_no = rng.choice([f"{index + 17}/{year}", f"DDT-{index + 501}", f"B{index + 1:05d}"])
    invoice_no = rng.choice([f"FT{year}-{index + 1:05d}", f"{index + 33}/A", f"{year}/{index + 1}"])

    ordered = [
        _Line(product=p, quantity=_quantity(rng, p), unit_price=_price(rng, p))
        for p in rng.sample(PRODUCTS, rng.randint(2, 8))
    ]
    delivered = [line.model_copy() for line in ordered]
    invoiced = [line.model_copy() for line in ordered]
    discrepancies = _inject(rng, ordered, delivered, invoiced)

    order = Document(
        kind=DocumentKind.ORDER,
        number=order_no,
        date=order_date,
        supplier=supplier,
        customer=customer,
        lines=_items(ordered, prices=True, vat=False),
        taxable_amount=sum((money(x.quantity * x.unit_price) for x in ordered), Decimal("0")),
    )
    ddt = Document(
        kind=DocumentKind.DDT,
        number=ddt_no,
        date=ddt_date,
        supplier=supplier,
        customer=customer,
        lines=_items(delivered, prices=False, vat=False),
        order_refs=(order_no,),
    )
    summary = _vat_summary(invoiced)
    taxable = sum((s.taxable_amount for s in summary), Decimal("0"))
    vat = sum((s.vat_amount for s in summary), Decimal("0"))
    invoice = Document(
        kind=DocumentKind.INVOICE,
        number=invoice_no,
        date=invoice_date,
        supplier=supplier,
        customer=customer,
        lines=_items(invoiced, prices=True, vat=True),
        order_refs=(order_no,),
        ddt_refs=(ddt_no,),
        vat_summary=summary,
        taxable_amount=taxable,
        vat_amount=vat,
        total_amount=taxable + vat,
    )
    return Chain(
        chain_id=f"chain-{index:04d}",
        order=order,
        ddt=ddt,
        invoice=invoice,
        discrepancies=tuple(discrepancies),
    )


def _party(rng: random.Random) -> Party:
    city = rng.choice(CITIES)
    name = f"{rng.choice(NAME_FIRST)} {rng.choice(NAME_SECOND)} {rng.choice(LEGAL_FORMS)}"
    return Party(
        name=name,
        vat_number=random_piva(rng),
        address=f"{rng.choice(STREETS)}, {rng.randint(1, 180)}",
        postal_code=city.postal_code,
        city=city.name,
        province=city.province,
    )


def _quantity(rng: random.Random, product: Product) -> Decimal:
    if product.unit == "m":
        return Decimal(rng.choice([25, 50, 100, 150, 200, 500]))
    if product.unit == "conf":
        return Decimal(rng.randint(1, 12))
    return Decimal(rng.randint(1, 40))


def _price(rng: random.Random, product: Product) -> Decimal:
    cents = rng.randint(int(product.min_price * 100), int(product.max_price * 100))
    return Decimal(cents) / 100


def _inject(
    rng: random.Random, ordered: list[_Line], delivered: list[_Line], invoiced: list[_Line]
) -> list[Discrepancy]:
    """Mutate delivered/invoiced lines in place; return what was injected (the ground truth)."""
    if rng.random() < 0.5:
        return []
    kinds = rng.sample(list(DiscrepancyType), rng.randint(1, 2))
    # Each discrepancy hits a different ordered line, so the ground truth is unambiguous.
    targets = rng.sample(range(len(ordered)), min(len(kinds), len(ordered)))
    found: list[Discrepancy] = []
    removed: set[str] = set()
    for kind, idx in zip(kinds, targets, strict=False):
        line = ordered[idx]
        code = line.product.code
        match kind:
            case DiscrepancyType.MISSING_LINE if len(ordered) - len(removed) > 1:
                removed.add(code)
                found.append(Discrepancy(type=kind, item_code=code, expected=line.quantity))
            case DiscrepancyType.QTY_MISMATCH:
                inv = next(x for x in invoiced if x.product.code == code)
                delta = max(Decimal(1), (line.quantity * Decimal("0.1")).to_integral_value())
                sign = rng.choice([1, -1]) if line.quantity > delta else 1  # keep quantity > 0
                inv.quantity = line.quantity + sign * delta
                found.append(
                    Discrepancy(
                        type=kind, item_code=code, expected=line.quantity, actual=inv.quantity
                    )
                )
            case DiscrepancyType.PRICE_MISMATCH:
                inv = next(x for x in invoiced if x.product.code == code)
                factor = Decimal(1) + Decimal(rng.choice([5, 8, 10, -5])) / 100
                inv.unit_price = money(line.unit_price * factor)
                found.append(
                    Discrepancy(
                        type=kind, item_code=code, expected=line.unit_price, actual=inv.unit_price
                    )
                )
            case DiscrepancyType.EXTRA_LINE:
                used = {x.product.code for x in ordered}
                product = rng.choice([p for p in PRODUCTS if p.code not in used])
                extra = _Line(
                    product=product,
                    quantity=_quantity(rng, product),
                    unit_price=_price(rng, product),
                )
                invoiced.append(extra)
                found.append(Discrepancy(type=kind, item_code=product.code, actual=extra.quantity))
    delivered[:] = [x for x in delivered if x.product.code not in removed]
    invoiced[:] = [x for x in invoiced if x.product.code not in removed]
    return found


def _items(lines: list[_Line], *, prices: bool, vat: bool) -> tuple[LineItem, ...]:
    return tuple(
        LineItem(
            line_number=i,
            code=x.product.code,
            description=x.product.description,
            quantity=x.quantity,
            unit=x.product.unit,
            unit_price=x.unit_price if prices else None,
            total=money(x.quantity * x.unit_price) if prices else None,
            vat_rate=x.product.vat_rate if vat else None,
        )
        for i, x in enumerate(lines, start=1)
    )


def _vat_summary(lines: list[_Line]) -> tuple[VatSummary, ...]:
    buckets: dict[Decimal, Decimal] = {}
    for x in lines:
        rate = x.product.vat_rate
        buckets[rate] = buckets.get(rate, Decimal("0")) + money(x.quantity * x.unit_price)
    return tuple(
        VatSummary(
            vat_rate=rate,
            taxable_amount=taxable,
            vat_amount=money(taxable * rate / 100),
        )
        for rate, taxable in sorted(buckets.items(), reverse=True)
    )
