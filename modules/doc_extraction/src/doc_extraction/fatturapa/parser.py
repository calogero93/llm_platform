"""Deterministic FatturaPA XML → `Document`. No LLM: the data is already structured."""

from datetime import date
from decimal import Decimal

from lxml import etree
from lxml.etree import _Element

from doc_extraction.fatturapa.xml import SAFE_PARSER, InvalidFatturaPAError
from doc_extraction.schemas import Document, DocumentKind, LineItem, Party, VatSummary


def _text(el: _Element, path: str) -> str:
    found = el.findtext(path)
    if found is None:
        raise InvalidFatturaPAError(f"missing required element {path}")
    return found.strip()


def _opt(el: _Element, path: str) -> str | None:
    found = el.findtext(path)
    return found.strip() if found is not None else None


def _party(el: _Element) -> Party:
    anagrafica = "DatiAnagrafici/Anagrafica/"
    name = _opt(el, anagrafica + "Denominazione") or " ".join(
        filter(None, (_opt(el, anagrafica + "Nome"), _opt(el, anagrafica + "Cognome")))
    )
    street = _text(el, "Sede/Indirizzo")
    number = _opt(el, "Sede/NumeroCivico")
    return Party(
        name=name,
        vat_number=_text(el, "DatiAnagrafici/IdFiscaleIVA/IdCodice"),
        address=f"{street}, {number}" if number else street,
        postal_code=_text(el, "Sede/CAP"),
        city=_text(el, "Sede/Comune"),
        province=_opt(el, "Sede/Provincia") or "",
    )


def _line(el: _Element) -> LineItem:
    quantity = _opt(el, "Quantita")
    return LineItem(
        line_number=int(_text(el, "NumeroLinea")),
        code=_opt(el, "CodiceArticolo/CodiceValore"),
        description=_text(el, "Descrizione"),
        # Quantita is optional in FatturaPA (e.g. services): its absence means one unit.
        quantity=Decimal(quantity) if quantity is not None else Decimal(1),
        unit=_opt(el, "UnitaMisura"),
        unit_price=Decimal(_text(el, "PrezzoUnitario")),
        total=Decimal(_text(el, "PrezzoTotale")),
        vat_rate=Decimal(_text(el, "AliquotaIVA")),
    )


def parse_fatturapa(xml: bytes) -> list[Document]:
    """One `Document` per `FatturaElettronicaBody` (a file may batch several invoices)."""
    root = etree.fromstring(xml, SAFE_PARSER)
    if root.getroottree().docinfo.doctype:  # type: ignore[union-attr]  # lxml-stubs lag lxml 6
        # FatturaPA never declares a DTD; one is only useful to an attacker (XXE, entity bombs).
        raise InvalidFatturaPAError("DTD declarations are not allowed")
    header = root.find("FatturaElettronicaHeader")
    if header is None:
        raise InvalidFatturaPAError("missing FatturaElettronicaHeader")
    supplier = _party(_require(header, "CedentePrestatore"))
    customer = _party(_require(header, "CessionarioCommittente"))

    documents = []
    for body in root.iterfind("FatturaElettronicaBody"):
        generali = _require(body, "DatiGenerali/DatiGeneraliDocumento")
        summary = tuple(
            VatSummary(
                vat_rate=Decimal(_text(r, "AliquotaIVA")),
                taxable_amount=Decimal(_text(r, "ImponibileImporto")),
                vat_amount=Decimal(_text(r, "Imposta")),
            )
            for r in body.iterfind("DatiBeniServizi/DatiRiepilogo")
        )
        total = _opt(generali, "ImportoTotaleDocumento")
        documents.append(
            Document(
                kind=DocumentKind.INVOICE,
                number=_text(generali, "Numero"),
                date=date.fromisoformat(_text(generali, "Data")),
                supplier=supplier,
                customer=customer,
                lines=tuple(_line(x) for x in body.iterfind("DatiBeniServizi/DettaglioLinee")),
                order_refs=tuple(
                    _text(x, "IdDocumento")
                    for x in body.iterfind("DatiGenerali/DatiOrdineAcquisto")
                ),
                ddt_refs=tuple(
                    _text(x, "NumeroDDT") for x in body.iterfind("DatiGenerali/DatiDDT")
                ),
                vat_summary=summary,
                taxable_amount=sum((s.taxable_amount for s in summary), Decimal(0)),
                vat_amount=sum((s.vat_amount for s in summary), Decimal(0)),
                total_amount=Decimal(total) if total is not None else None,
            )
        )
    if not documents:
        raise InvalidFatturaPAError("no FatturaElettronicaBody")
    return documents


def _require(el: _Element, path: str) -> _Element:
    found = el.find(path)
    if found is None:
        raise InvalidFatturaPAError(f"missing required element {path}")
    return found
