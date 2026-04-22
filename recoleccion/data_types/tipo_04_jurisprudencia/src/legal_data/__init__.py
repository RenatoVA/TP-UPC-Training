"""Herramientas para tipo de dato 04: jurisprudencia relevante."""

from .resumeai_client import (
    ResumeAILegalAnalysisRequest,
    ResumeAILegalEnrichedCase,
    ResumeAILegalRunMetadata,
    analyze_pdf_with_resumeai_legal,
    persist_enriched_case,
)

__all__ = [
    "ResumeAILegalAnalysisRequest",
    "ResumeAILegalEnrichedCase",
    "ResumeAILegalRunMetadata",
    "analyze_pdf_with_resumeai_legal",
    "persist_enriched_case",
]
