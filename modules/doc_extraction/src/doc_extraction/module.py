from fastapi import APIRouter

from doc_extraction.evals import InvoiceXmlSuite
from llmp.eval import EvalSuite
from llmp.modules import PlatformContext


class DocExtractionModule:
    name = "doc_extraction"

    def router(self, ctx: PlatformContext) -> APIRouter:
        # Endpoints arrive in Phase 3 (extraction) and Phase 4 (reconciliation).
        return APIRouter(tags=["doc-extraction"])

    def eval_suites(self) -> list[EvalSuite]:
        return [InvoiceXmlSuite()]


module = DocExtractionModule()
