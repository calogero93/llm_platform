# Vendored XML schemas

| File | Source | SHA-256 of the original download | Local change |
|---|---|---|---|
| `Schema_VFPR12_v1.2.3.xsd` | https://www.fatturapa.gov.it/export/documenti/fatturapa/v1.4/Schema_VFPR12_v1.2.3.xsd (FatturaPA B2B, valid from 2025-04-01) | `152944f6eef9f5d69ef6e955ee173b32142b00a8c1c5222fc97dfab5910e8a8c` | `xs:import` of xmldsig points to the local file instead of w3.org |
| `xmldsig-core-schema.xsd` | https://www.w3.org/TR/2002/REC-xmldsig-core-20020212/xmldsig-core-schema.xsd | `35cf8197da812c85e40d57891b35c94187569ed474a2dac813ce5090dafcd35c` | none |

Vendored so validation works offline (no network at runtime).
