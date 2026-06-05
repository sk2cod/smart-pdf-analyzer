# ============================================================
# ui/tab_summarization.py
# ============================================================
# Renders the Summarization tab.
#
# Changes from original:
#   - When output format is mermaid_flowchart or mermaid_mindmap,
#     the raw code block is REPLACED by a live rendered diagram
#     via utils/mermaid_renderer.py.
#   - All other formats (bullet_hierarchy, executive_prose,
#     comparison_table) are unchanged.
#   - Export still exports the raw text/code as before.
# ============================================================

import streamlit as st

from config import (
    SUMMARY_FORMATS,
    SUMMARY_MIN_WORDS,
    SUMMARY_MAX_WORDS,
    SUMMARIZATION_COST_GATE_CHUNKS,
)
from modes.summarization import run_summarization
from utils.cost_tracker import log_usage
from utils.mermaid_renderer import render_mermaid, extract_mermaid_code, estimate_height

# ------------------------------------------------------------
# Format helpers
# ------------------------------------------------------------

MERMAID_FORMATS = {"mermaid_flowchart", "mermaid_mindmap"}


def _is_mermaid(format_key: str) -> bool:
    return format_key in MERMAID_FORMATS


# ------------------------------------------------------------
# Main render function
# ------------------------------------------------------------

def render_summarization_tab() -> None:
    """Render the Summarization tab UI."""

    st.header("📝 Summarization")
    st.caption("Generate structured summaries with custom length and format controls.")

    # Guard — documents must be processed first
    if not st.session_state.get("ingestion_complete"):
        st.info("Upload and process documents using the sidebar to get started.")
        return

    active_files: list = st.session_state.get("active_filenames", [])
    chunks: list = st.session_state.get("processed_chunks", [])

    # --------------------------------------------------------
    # Controls row
    # --------------------------------------------------------
    col_source, col_format = st.columns([1, 1])

    with col_source:
        st.markdown("**Source**")
        source_options = ["All documents"] + active_files
        source_choice = st.radio(
            label="Source",
            options=source_options,
            label_visibility="collapsed",
        )

    with col_format:
        st.markdown("**Output format**")
        format_key = st.selectbox(
            label="Output format",
            options=list(SUMMARY_FORMATS.keys()),
            format_func=lambda k: SUMMARY_FORMATS[k],
            key="summary_format",
            label_visibility="collapsed",
        )

    # Word-length slider (hidden for Mermaid — diagrams have no word limit)
    if not _is_mermaid(format_key):
        st.markdown("**Maximum length (words)**")
        max_words = st.slider(
            label="Maximum length (words)",
            min_value=SUMMARY_MIN_WORDS,
            max_value=SUMMARY_MAX_WORDS,
            value=st.session_state.get("summary_max_words", 500),
            step=50,
            key="summary_max_words",
            label_visibility="collapsed",
        )
    else:
        max_words = 500   # unused for Mermaid but required by run_summarization signature

    # Optional focus instruction
    focus = st.text_input(
        label="Focus instruction (optional)",
        placeholder="e.g. Focus on financial figures and risk factors only",
        key="summary_focus",
    )

    # --------------------------------------------------------
    # Cost gate for large documents
    # --------------------------------------------------------
    chunk_count = len(chunks)
    needs_confirmation = (
        chunk_count > SUMMARIZATION_COST_GATE_CHUNKS
        and not st.session_state.get("summary_confirmed_cost", False)
    )

    if needs_confirmation:
        st.warning(
            f"⚠️ This document has **{chunk_count} chunks**. "
            f"Summarization will use gpt-4o and may cost ~$0.05–$0.15. "
            f"Confirm to proceed."
        )
        if st.button("✅ Yes, proceed with summarization"):
            st.session_state["summary_confirmed_cost"] = True
            st.rerun()
        return

    # --------------------------------------------------------
    # Generate button
    # --------------------------------------------------------
    if st.button("⚡ Generate Summary", type="primary"):
        selected_chunks = (
            chunks
            if source_choice == "All documents"
            else [c for c in chunks if c.metadata.get("source") == source_choice]
        )

        if not selected_chunks:
            st.error("No chunks available for the selected source.")
            return

        with st.spinner("Generating summary…"):
            result = run_summarization(
                chunks=selected_chunks,
                format_key=format_key,
                max_words=max_words,
                focus=focus or None,
            )

        if result.get("error"):
            st.error(f"Summarization failed: {result['error']}")
            return

        # Store result in session state
        st.session_state["summary_result"] = result["summary"]
        st.session_state["summary_format_used"] = format_key
        st.session_state["summary_meta"] = {
            "words": result.get("word_count", 0),
            "chunks": len(selected_chunks),
            "format_label": SUMMARY_FORMATS[format_key],
        }

        # Log cost
        if result.get("token_usage"):
            log_usage(result["token_usage"])

    # --------------------------------------------------------
    # Display result
    # --------------------------------------------------------
    summary_text: str | None = st.session_state.get("summary_result")
    format_used: str = st.session_state.get("summary_format_used", format_key)
    meta: dict = st.session_state.get("summary_meta", {})

    if not summary_text:
        return

    st.divider()

    # Metadata footer line
    meta_parts = []
    if meta.get("words"):
        meta_parts.append(f"{meta['words']} words")
    if meta.get("chunks"):
        meta_parts.append(f"{meta['chunks']} chunks processed")
    if meta.get("format_label"):
        meta_parts.append(meta["format_label"])

    col_meta, col_export = st.columns([3, 1])

    with col_meta:
        if meta_parts:
            st.caption(" · ".join(meta_parts))

    with col_export:
        st.download_button(
            label="📥 Export",
            data=summary_text,
            file_name="summary.md",
            mime="text/markdown",
        )

    # --------------------------------------------------------
    # Render result — Mermaid vs text formats
    # --------------------------------------------------------
    if _is_mermaid(format_used):
        _render_mermaid_result(summary_text, format_used)
    else:
        _render_text_result(summary_text, format_used)


# ------------------------------------------------------------
# Mermaid rendering
# ------------------------------------------------------------

def _render_mermaid_result(raw_output: str, format_used: str) -> None:
    """
    Extract and render a live Mermaid diagram.
    Replaces the code block entirely — no raw code shown.
    """
    clean_code = extract_mermaid_code(raw_output)

    if not clean_code:
        st.warning("No diagram code found in the output. Try regenerating.")
        return

    diagram_type = (
        "Flowchart" if format_used == "mermaid_flowchart" else "Mindmap"
    )
    st.markdown(f"#### 🗺️ {diagram_type} Diagram")

    height = estimate_height(clean_code)
    render_mermaid(clean_code, height=height)


# ------------------------------------------------------------
# Text / table rendering (unchanged behaviour)
# ------------------------------------------------------------

def _render_text_result(summary_text: str, format_used: str) -> None:
    """
    Render non-Mermaid summary output.
    - comparison_table → st.markdown (renders the table)
    - all others       → st.markdown
    """
    st.markdown(summary_text)
