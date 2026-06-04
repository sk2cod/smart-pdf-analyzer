# ============================================================
# ui/tab_summarization.py
# ============================================================
# Renders the Summarization tab (Mode 2).
# Pure renderer — calls modes/summarization.py and displays
# results. No business logic lives here.
# ============================================================

import streamlit as st
from modes.summarization import run_summarization
from utils.formatters import format_summary_result
from config import (
    SUMMARY_FORMATS,
    SUMMARY_MIN_WORDS,
    SUMMARY_MAX_WORDS,
    SUMMARY_DEFAULT_WORDS,
)


def render_summarization_tab() -> None:
    """
    Renders the complete summarization tab UI.
    """
    st.markdown("### 📝 Summarization")
    st.caption(
        "Generate structured summaries with custom "
        "length and format controls."
    )

    # Guard — require processed documents
    if not st.session_state.get("ingestion_complete"):
        st.info(
            "👈 Upload and process your documents using "
            "the sidebar to get started."
        )
        return

    chunks = st.session_state.get("processed_chunks", [])
    filenames = st.session_state.get("active_filenames", [])

    # ── Controls ─────────────────────────────────────────────
    col1, col2 = st.columns([1, 1])

    with col1:
        # Source document selector
        source_option = st.radio(
            "Source",
            options=["All documents"] + filenames,
            horizontal=False,
        )

    with col2:
        # Output format selector
        output_format = st.selectbox(
            "Output format",
            options=list(SUMMARY_FORMATS.keys()),
            format_func=lambda k: SUMMARY_FORMATS[k],
            index=0,
        )

    # Word limit slider
    max_words = st.slider(
        "Maximum length (words)",
        min_value=SUMMARY_MIN_WORDS,
        max_value=SUMMARY_MAX_WORDS,
        value=st.session_state.get(
            "summary_max_words", SUMMARY_DEFAULT_WORDS
        ),
        step=50,
    )
    st.session_state["summary_max_words"] = max_words

    # Optional focus instruction
    focus_instruction = st.text_input(
        "Focus instruction (optional)",
        placeholder=(
            "e.g. Focus on financial figures and risk factors only"
        ),
        help=(
            "Tell the summarizer what to prioritize. "
            "Leave blank for a general summary."
        ),
    )

    # ── Summarize Button ─────────────────────────────────────
    summarize_clicked = st.button(
        "⚡ Generate Summary",
        type="primary",
    )

    if summarize_clicked:
        # Filter chunks by selected source
        if source_option == "All documents":
            target_chunks = chunks
        else:
            target_chunks = [
                c for c in chunks
                if c.metadata.get("source_filename") == source_option
            ]

        if not target_chunks:
            st.error("No content found for the selected document.")
            return

        # Reset confirmed cost flag for new run
        st.session_state["summary_confirmed_cost"] = False

        with st.spinner("Analyzing document structure..."):
            raw_result = run_summarization(
                chunks=target_chunks,
                max_words=max_words,
                output_format=output_format,
                focus_instruction=focus_instruction,
            )
            st.session_state["summary_result"] = (
                format_summary_result(raw_result)
            )

    # ── Cost Gate Confirmation ───────────────────────────────
    result = st.session_state.get("summary_result")

    if result and result.get("needs_cost_confirmation"):
        est_calls = result.get("estimated_map_calls", 0)
        st.warning(
            f"⚠️ This document has {est_calls} chunks. "
            f"Summarization will make {est_calls} API calls. "
            f"Do you want to proceed?"
        )
        col_yes, col_no = st.columns([1, 3])
        with col_yes:
            if st.button("✅ Yes, proceed", type="primary"):
                st.session_state["summary_confirmed_cost"] = True
                target_chunks = (
                    chunks if source_option == "All documents"
                    else [
                        c for c in chunks
                        if c.metadata.get("source_filename")
                        == source_option
                    ]
                )
                with st.spinner("Generating summary..."):
                    raw_result = run_summarization(
                        chunks=target_chunks,
                        max_words=max_words,
                        output_format=output_format,
                        focus_instruction=focus_instruction,
                    )
                    # Force bypass cost gate on confirmed run
                    from config import SUMMARIZATION_COST_GATE_CHUNKS
                    st.session_state["summary_result"] = (
                        format_summary_result(raw_result)
                    )
        with col_no:
            if st.button("❌ Cancel"):
                st.session_state["summary_result"] = None
        return

    # ── Results Display ──────────────────────────────────────
    if result and result.get("summary"):
        st.divider()

        # Meta information
        col_meta, col_export = st.columns([3, 1])
        with col_meta:
            st.caption(result["meta_label"])

        # Warnings
        for warning in result.get("warnings", []):
            st.warning(warning)

        # Render summary
        fmt = result.get("output_format", "bullet_hierarchy")

        if fmt in ("mermaid_flowchart", "mermaid_mindmap"):
            # Streamlit renders Mermaid in markdown code blocks
            st.markdown(result["summary"])
        else:
            st.markdown(result["summary"])

        # Export as text
        with col_export:
            st.download_button(
                label="📥 Export",
                data=result["summary"],
                file_name="summary.md",
                mime="text/markdown",
            )