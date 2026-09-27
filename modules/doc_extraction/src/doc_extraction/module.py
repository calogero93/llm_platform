from fastapi import APIRouter

from llmp.modules import PlatformContext


class DocExtractionModule:
    name = "doc_extraction"

    def router(self, ctx: PlatformContext) -> APIRouter:
        # Endpoints arrive in Phase 3 (extraction) and Phase 4 (reconciliation).
        return APIRouter(tags=["doc-extraction"])


module = DocExtractionModule()
