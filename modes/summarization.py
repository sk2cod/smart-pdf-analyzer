# ============================================================
# modes/summarization.py
# ============================================================
# Mode 2: Context-Constrained Summarization chain.
# Implements map-reduce pattern:
#   Map:     cheap model compresses each chunk independently
#   Reduce:  premium model synthesizes all compressed chunks
#
# Does NOT import from Streamlit.
# Does NOT render output — that is formatters.py's job.
# ============================================================

from langchain_core.documents import Document
from core.router import (
    get_model,
    TASK_SUMMARIZATION_MAP,
    TASK_SUMMARIZATION_SYNTHESIS,
)
from prompts.summarization_prompts import (
    MAP_PROMPT,
    SYNTHESIS_PROMPT,
    FORMAT_INSTRUCTIONS,
)
from config import (
    SUMMARIZATION_COST_GATE_CHUNKS,
    SUMMARY_FORMATS,
)


def _run_map_stage(chunks: list[Document]) -> list[str]:
    """
    Compresses each chunk independently using the cheap model.

    Args:
        chunks: Document chunks to compress.

    Returns:
        List of compressed bullet-point summaries.
    """
    model = get_model(TASK_SUMMARIZATION_MAP)
    chain = MAP_PROMPT | model
    compressed = []

    for chunk in chunks:
        result = chain.invoke({"text": chunk.page_content})
        compressed.append(result.content.strip())

    return compressed


def _run_synthesis_stage(
    compressed_summaries: list[str],
    max_words: int,
    output_format: str,
    focus_instruction: str = "",
) -> str:
    """
    Synthesizes all compressed summaries into final output
    using the premium model.

    Args:
        compressed_summaries: Output from _run_map_stage()
        max_words: Maximum word count for output
        output_format: Key from FORMAT_INSTRUCTIONS dict
        focus_instruction: Optional user-defined focus hint

    Returns:
        Final formatted summary string.
    """
    model = get_model(TASK_SUMMARIZATION_SYNTHESIS)
    chain = SYNTHESIS_PROMPT | model

    format_instruction = FORMAT_INSTRUCTIONS.get(
        output_format,
        FORMAT_INSTRUCTIONS["bullet_hierarchy"]
    )

    focus_text = ""
    if focus_instruction.strip():
        focus_text = (
            f"FOCUS INSTRUCTION: The user has requested special "
            f"attention to: {focus_instruction.strip()}\n"
            f"Prioritise this content in your output."
        )

    combined_text = "\n\n---\n\n".join(compressed_summaries)

    result = chain.invoke({
        "text": combined_text,
        "max_words": max_words,
        "output_format_instruction": format_instruction,
        "focus_instruction": focus_text,
    })

    return result.content.strip()


def run_summarization(
    chunks: list[Document],
    max_words: int = 500,
    output_format: str = "bullet_hierarchy",
    focus_instruction: str = "",
) -> dict:
    """
    Runs the full map-reduce summarization pipeline.

    Args:
        chunks: Document chunks filtered by selected source docs
        max_words: User-defined word limit from UI slider
        output_format: One of the keys in SUMMARY_FORMATS
        focus_instruction: Optional user focus hint from UI

    Returns:
        Result dict with summary, word count, and metadata.
    """
    warnings = []

    if not chunks:
        return {
            "summary": "",
            "word_count": 0,
            "chunks_processed": 0,
            "output_format": output_format,
            "needs_cost_confirmation": False,
            "estimated_map_calls": 0,
            "warnings": ["No content available to summarize."],
        }

    chunk_count = len(chunks)

    # Run map stage
    compressed = _run_map_stage(chunks)

    # Run synthesis stage
    summary = _run_synthesis_stage(
        compressed_summaries=compressed,
        max_words=max_words,
        output_format=output_format,
        focus_instruction=focus_instruction,
    )

    # Post-processing word count check
    word_count = len(summary.split())
    if word_count > max_words * 1.1:
        warnings.append(
            f"Summary is {word_count} words, slightly over the "
            f"{max_words} word limit."
        )
    # Estimate token usage for cost tracking
    estimated_input_tokens = chunk_count * 800
    estimated_output_tokens = chunk_count * 200 + 600
    return {
        "summary": summary,
        "word_count": word_count,
        "chunks_processed": chunk_count,
        "output_format": output_format,
        "needs_cost_confirmation": False,
        "estimated_map_calls": chunk_count,
        "estimated_input_tokens": estimated_input_tokens,
        "estimated_output_tokens": estimated_output_tokens,
        "warnings": warnings,
    }

def _run_map_stage(chunks: list) -> list[str]:
    """Public wrapper for map stage — used by UI cache system."""
    from core.router import get_model, TASK_SUMMARIZATION_MAP
    from prompts.summarization_prompts import MAP_PROMPT
    model = get_model(TASK_SUMMARIZATION_MAP)
    chain = MAP_PROMPT | model
    compressed = []
    for chunk in chunks:
        result = chain.invoke({"text": chunk.page_content})
        compressed.append(result.content.strip())
    return compressed


def _run_synthesis_stage(
    compressed_summaries: list[str],
    max_words: int,
    output_format: str,
    focus_instruction: str = "",
) -> str:
    """Public wrapper for synthesis stage — used by UI cache system."""
    from core.router import get_model, TASK_SUMMARIZATION_SYNTHESIS
    from prompts.summarization_prompts import (
        SYNTHESIS_PROMPT,
        FORMAT_INSTRUCTIONS,
    )
    model = get_model(TASK_SUMMARIZATION_SYNTHESIS)
    chain = SYNTHESIS_PROMPT | model
    format_instruction = FORMAT_INSTRUCTIONS.get(
        output_format,
        FORMAT_INSTRUCTIONS["bullet_hierarchy"]
    )
    focus_text = ""
    if focus_instruction.strip():
        focus_text = (
            f"FOCUS INSTRUCTION: {focus_instruction.strip()}"
        )
    combined_text = "\n\n---\n\n".join(compressed_summaries)
    result = chain.invoke({
        "text": combined_text,
        "max_words": max_words,
        "output_format_instruction": format_instruction,
        "focus_instruction": focus_text,
    })
    return result.content.strip()