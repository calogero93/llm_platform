"""Render a `Document` (invoice) as a FatturaPA FPR12 XML file."""

from datetime import date
from decimal import Decimal

from lxml import etree
from lxml.etree import _Element

from doc_extraction.fatturapa import NAMESPACE
from doc_extraction.schemas import Document, DocumentKind, Party


def _sub(parent: _Element, tag: str, text: str | None = None) -> _Element:
    el = etree.SubElement(parent, tag)
    if text is not None:
        el.text = text
    return el


def _amount(value: Decimal) -> str:
    return f"{value:.2f}"


def _party(parent: _Element, tag: str, party: Party, *, seller: bool) -> None:
    root = _sub(parent, tag)
    anagrafici = _sub(root, "DatiAnagrafici")
    id_iva = _sub(anagrafici, "IdFiscaleIVA")
    _sub(id_iva, "IdPaese", "IT")
    _sub(id_iva, "IdCodice", party.vat_number)
    _sub(_sub(anagrafici, "Anagrafica"), "Denominazione", party.name)
    if seller:
        _sub(anagrafici, "RegimeFiscale", "RF01")  # ordinary VAT regime
    sede = _sub(root, "Sede")
    _sub(sede, "Indirizzo", party.address)
    _sub(sede, "CAP", party.postal_code)
    _sub(sede, "Comune", party.city)
    _sub(sede, "Provincia", party.province)
    _sub(sede, "Nazione", "IT")


def to_fatturapa_xml(invoice: Document, progressive: str, ddt_dates: dict[str, date]) -> bytes:
    """`ddt_dates` maps each referenced DDT number to its date (required by the XSD)."""
    if invoice.kind is not DocumentKind.INVOICE:
        raise ValueError(f"only invoices can be FatturaPA, got {invoice.kind}")
    root = etree.Element(f"{{{NAMESPACE}}}FatturaElettronica", nsmap={"p": NAMESPACE})
    root.set("versione", "FPR12")

    header = _sub(root, "FatturaElettronicaHeader")
    trasmissione = _sub(header, "DatiTrasmissione")
    trasmittente = _sub(trasmissione, "IdTrasmittente")
    _sub(trasmittente, "IdPaese", "IT")
    _sub(trasmittente, "IdCodice", invoice.supplier.vat_number)
    _sub(trasmissione, "ProgressivoInvio", progressive)
    _sub(trasmissione, "FormatoTrasmissione", "FPR12")
    _sub(trasmissione, "CodiceDestinatario", "0000000")
    _party(header, "CedentePrestatore", invoice.supplier, seller=True)
    _party(header, "CessionarioCommittente", invoice.customer, seller=False)

    body = _sub(root, "FatturaElettronicaBody")
    generali = _sub(body, "DatiGenerali")
    documento = _sub(generali, "DatiGeneraliDocumento")
    _sub(documento, "TipoDocumento", "TD01")
    _sub(documento, "Divisa", "EUR")
    _sub(documento, "Data", invoice.date.isoformat())
    _sub(documento, "Numero", invoice.number)
    if invoice.total_amount is not None:
        _sub(documento, "ImportoTotaleDocumento", _amount(invoice.total_amount))
    for order_ref in invoice.order_refs:
        _sub(_sub(generali, "DatiOrdineAcquisto"), "IdDocumento", order_ref)
    for ddt_ref in invoice.ddt_refs:
        ddt = _sub(generali, "DatiDDT")
        _sub(ddt, "NumeroDDT", ddt_ref)
        _sub(ddt, "DataDDT", ddt_dates[ddt_ref].isoformat())

    beni = _sub(body, "DatiBeniServizi")
    for line in invoice.lines:
        if line.unit_price is None or line.total is None or line.vat_rate is None:
            raise ValueError(f"invoice line {line.line_number} lacks price, total or VAT")
        dettaglio = _sub(beni, "DettaglioLinee")
        _sub(dettaglio, "NumeroLinea", str(line.line_number))
        if line.code is not None:
            codice = _sub(dettaglio, "CodiceArticolo")
            _sub(codice, "CodiceTipo", "INTERNO")
            _sub(codice, "CodiceValore", line.code)
        _sub(dettaglio, "Descrizione", line.description)
        _sub(dettaglio, "Quantita", f"{line.quantity:.2f}")
        if line.unit is not None:
            _sub(dettaglio, "UnitaMisura", line.unit)
        _sub(dettaglio, "PrezzoUnitario", _amount(line.unit_price))
        _sub(dettaglio, "PrezzoTotale", _amount(line.total))
        _sub(dettaglio, "AliquotaIVA", _amount(line.vat_rate))
    for summary in invoice.vat_summary:
        riepilogo = _sub(beni, "DatiRiepilogo")
        _sub(riepilogo, "AliquotaIVA", _amount(summary.vat_rate))
        _sub(riepilogo, "ImponibileImporto", _amount(summary.taxable_amount))
        _sub(riepilogo, "Imposta", _amount(summary.vat_amount))
        _sub(riepilogo, "EsigibilitaIVA", "I")  # immediate

    return etree.tostring(root, xml_declaration=True, encoding="UTF-8", pretty_print=True)
