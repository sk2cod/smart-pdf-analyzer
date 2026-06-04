# ============================================================
# utils/formatters.py
# ============================================================
# Converts raw mode outputs into display-ready strings.
# Owns all output rendering logic.
# Does NOT call any LLM or import from Streamlit.
# ============================================================

from config import SUMMARY_FORMATS


# ------------------------------------------------------------
# Extraction Formatter
# ------------------------------------------------------------

def format_extraction_result(result: dict) -> dict:
    """
    Converts raw extraction dict into display-ready structure.

    Args:
        result: Raw dict from modes/extraction.run_extraction()

    Returns:
        Display dict:
        {
            "document_type": str,
            "rows": list[dict],   each: {field, value, confidence}
            "notes": str,
            "warnings": list,
            "pages_processed": int,
            "vision_pages": int,
            "has_low_confidence": bool,
            "copyable_text": str,
        }
    """
    fields = result.get("fields", {})
    confidence = result.get("confidence", {})

    rows = []
    has_low_confidence = False

    for field, value in fields.items():
        conf = confidence.get(field, "LOW")
        if conf in ("LOW", "NULL"):
            has_low_confidence = True

        rows.append({
            "field": _format_field_name(field),
            "value": value if value and value != "NULL" else "—",
            "confidence": conf,
            "is_null": value == "NULL" or not value,
        })

    # Build copy-pasteable plain text version
    copyable_lines = [
        f"Document Type: {result.get('document_type', 'Unknown')}",
        "-" * 40,
    ]
    for row in rows:
        flag = " ⚠️" if row["confidence"] in ("LOW", "NULL") else ""
        copyable_lines.append(
            f"{row['field']:<30} {row['value']}{flag}"
        )
    copyable_text = "\n".join(copyable_lines)

    return {
        "document_type": result.get("document_type", "Unknown"),
        "rows": rows,
        "notes": result.get("extraction_notes", ""),
        "warnings": result.get("warnings", []),
        "pages_processed": result.get("pages_processed", 0),
        "vision_pages": result.get("vision_pages", 0),
        "has_low_confidence": has_low_confidence,
        "copyable_text": copyable_text,
    }


def _format_field_name(field: str) -> str:
    """
    Converts snake_case or UPPER_CASE field names to
    Title Case with spaces for display.
    e.g. "invoice_number" → "Invoice Number"
    """
    return field.replace("_", " ").replace("-", " ").title()


# ------------------------------------------------------------
# Summary Formatter
# ------------------------------------------------------------

def format_summary_result(result: dict) -> dict:
    """
    Prepares summary result for display.

    Args:
        result: Raw dict from modes/summarization.run_summarization()

    Returns:
        Display dict with formatted metadata label.
    """
    fmt = result.get("output_format", "bullet_hierarchy")
    fmt_label = SUMMARY_FORMATS.get(fmt, fmt)

    word_count = result.get("word_count", 0)
    chunks = result.get("chunks_processed", 0)

    meta_label = (
        f"{word_count} words · "
        f"{chunks} chunks processed · "
        f"{fmt_label}"
    )

    return {
        "summary": result.get("summary", ""),
        "meta_label": meta_label,
        "word_count": word_count,
        "warnings": result.get("warnings", []),
        "output_format": fmt,
        "needs_cost_confirmation": result.get(
            "needs_cost_confirmation", False
        ),
        "estimated_map_calls": result.get("estimated_map_calls", 0),
    }


# ------------------------------------------------------------
# Chat Formatter
# ------------------------------------------------------------

def format_sources_display(sources: list[dict]) -> str:
    """
    Formats source citations list into a clean display string.

    Args:
        sources: List of {filename, page} dicts from chatbot.

    Returns:
        Formatted markdown string for source display.
    """
    if not sources:
        return ""

    lines = ["📎 **Sources:**"]
    for source in sources:
        lines.append(
            f"- {source['filename']} — Page {source['page']}"
        )
    return "\n".join(lines)