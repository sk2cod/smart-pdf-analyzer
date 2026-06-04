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


# ------------------------------------------------------------
# Map Stage
# ------------------------------------------------------------

def _run_map_stage(chunks: list[Document]) -> list[str]:
    """
    Compresses each chunk independently using the cheap model.
    Runs sequentially — parallelisation can be added in v2.

    Args:
        chunks: Document chunks from the vector store or
                directly from ingestion for summarization.

    Returns:
        List of compressed bullet-point summaries,
        one string per input chunk.
    """
    model = get_model(TASK_SUMMARIZATION_MAP)
    chain = MAP_PROMPT | model
    compressed = []

    for chunk in chunks:
        result = chain.invoke({"text": chunk.page_content})
        compressed.append(result.content.strip())

    return compressed


# ------------------------------------------------------------
# Synthesis Stage
# ------------------------------------------------------------

def _run_synthesis_stage(
    compressed_summaries: list[str],
    max_words: int,
    output_format: str,
    focus_instruction: str = "",
) -> str:
    """
    Synthesizes all compressed summaries into the final output
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

    # Build format instruction from config
    format_instruction = FORMAT_INSTRUCTIONS.get(
        output_format,
        FORMAT_INSTRUCTIONS["bullet_hierarchy"]
    )

    # Build focus instruction
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


# ------------------------------------------------------------
# Main Entry Point
# ------------------------------------------------------------

def run_summarization(
    chunks: list[Document],
    max_words: int = 500,
    output_format: str = "bullet_hierarchy",
    focus_instruction: str = "",
) -> dict:
    """
    Runs the full map-reduce summarization pipeline.

    Args:
        chunks: Document chunks — filtered by selected source docs
        max_words: User-defined word limit from UI slider
        output_format: One of the keys in SUMMARY_FORMATS
        focus_instruction: Optional user focus hint from UI

    Returns:
        Result dict:
        {
            "summary": str,
            "word_count": int,
            "chunks_processed": int,
            "output_format": str,
            "needs_cost_confirmation": bool,
            "estimated_map_calls": int,
            "warnings": list
        }
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

    # Cost gate — warn user before large summarization jobs
    needs_confirmation = chunk_count > SUMMARIZATION_COST_GATE_CHUNKS

    if needs_confirmation:
        return {
            "summary": "",
            "word_count": 0,
            "chunks_processed": 0,
            "output_format": output_format,
            "needs_cost_confirmation": True,
            "estimated_map_calls": chunk_count,
            "warnings": [
                f"This document produces {chunk_count} chunks. "
                f"Summarization will make {chunk_count} cheap model "
                f"calls plus 1 premium model call. "
                f"Confirm to proceed."
            ],
        }

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
            f"{max_words} word limit. The model was asked to trim "
            f"but exceeded it marginally."
        )

    return {
        "summary": summary,
        "word_count": word_count,
        "chunks_processed": chunk_count,
        "output_format": output_format,
        "needs_cost_confirmation": False,
        "estimated_map_calls": chunk_count,
        "warnings": warnings,
    }