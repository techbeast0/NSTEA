"""Context builder — assembles grounded RAG context for the reasoning agent."""

from __future__ import annotations

from nstea.retrieval.vector_store import SearchResult


def build_context(
    patient_summary: str,
    search_results: list[SearchResult],
    max_context_chars: int = 4000,
) -> str:
    """Assemble final context string from patient summary + retrieved guidelines.

    Args:
        patient_summary: Pre-formatted patient clinical summary.
        search_results: Retrieved guideline chunks with relevance scores.
        max_context_chars: Approximate cap on retrieved context length.

    Returns:
        Combined context string ready for LLM prompt insertion.
    """
    sections = [patient_summary, ""]

    if search_results:
        sections.append("--- RETRIEVED CLINICAL GUIDELINES ---")
        char_budget = max_context_chars
        for i, sr in enumerate(search_results, 1):
            entry = (
                f"\n[Guideline {i}] (source: {sr.chunk.source}, relevance: {sr.score:.2f})\n"
                f"{sr.chunk.text}\n"
            )
            if len(entry) > char_budget:
                # Truncate this entry to fit
                entry = entry[:char_budget] + "\n... [truncated]"
                sections.append(entry)
                break
            sections.append(entry)
            char_budget -= len(entry)
        sections.append("--- END GUIDELINES ---")
    else:
        sections.append(
            "--- NO GUIDELINES RETRIEVED ---\n"
            "Note: No matching clinical guidelines were found. "
            "Base your analysis on general medical knowledge and explicitly "
            "state when you lack guideline support."
        )

    return "\n".join(sections)
