# ============================================================
# ui/tab_summarization.py
# ============================================================

import streamlit as st
from modes.summarization import run_summarization
from utils.formatters import format_summary_result
from utils.mermaid_renderer import render_mermaid, extract_mermaid_code, estimate_height
from config import (
    SUMMARY_FORMATS,
    SUMMARY_MIN_WORDS,
    SUMMARY_MAX_WORDS,
    SUMMARY_DEFAULT_WORDS,
)

COST_GATE_THRESHOLD = 40
MERMAID_FORMATS = {"mermaid_flowchart", "mermaid_mindmap"}


def render_summarization_tab() -> None:
    st.markdown("### 📝 Summarization")
    st.caption(
        "Generate structured summaries with custom "
        "length and format controls."
    )

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
        source_option = st.radio(
            "Source",
            options=["All documents"] + filenames,
            horizontal=False,
        )

    with col2:
        output_format = st.selectbox(
            "Output format",
            options=list(SUMMARY_FORMATS.keys()),
            format_func=lambda k: SUMMARY_FORMATS[k],
            index=0,
        )

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

    focus_instruction = st.text_input(
        "Focus instruction (optional)",
        placeholder=(
            "e.g. Focus on financial figures and risk factors only"
        ),
    )

    # Filter chunks by source selection
    if source_option == "All documents":
        target_chunks = chunks
    else:
        target_chunks = [
            c for c in chunks
            if c.metadata.get("source_filename") == source_option
        ]

    chunk_count = len(target_chunks)
    needs_confirmation = chunk_count > COST_GATE_THRESHOLD

    # ── Generate Button ───────────────────────────────────────
    generate_clicked = st.button(
        "⚡ Generate Summary",
        type="primary",
    )

    # ── Cost Gate ─────────────────────────────────────────────
    if generate_clicked:
        if needs_confirmation:
            st.session_state["summary_awaiting_confirmation"] = True
            st.session_state["summary_pending_chunks"] = chunk_count
        else:
            _run_and_store(
                target_chunks=target_chunks,
                chunk_count=chunk_count,
                max_words=max_words,
                output_format=output_format,
                focus_instruction=focus_instruction,
            )

    if st.session_state.get("summary_awaiting_confirmation"):
        pending = st.session_state.get("summary_pending_chunks", 0)
        st.warning(
            f"⚠ This document has {pending} chunks. "
            f"Summarization will make {pending} cheap model calls "
            f"+ 1 premium model call. "
            f"Estimated cost: ${(pending * 0.0002) + 0.03:.3f}"
        )
        col_yes, col_no = st.columns([1, 3])
        with col_yes:
            if st.button("✅ Yes, proceed", type="primary"):
                st.session_state["summary_awaiting_confirmation"] = False
                _run_and_store(
                    target_chunks=target_chunks,
                    chunk_count=chunk_count,
                    max_words=max_words,
                    output_format=output_format,
                    focus_instruction=focus_instruction,
                )
        with col_no:
            if st.button("❌ Cancel"):
                st.session_state["summary_awaiting_confirmation"] = False
                st.rerun()
        return

    # ── Results Display ───────────────────────────────────────
    _render_result(st.session_state.get("summary_result"), output_format)


def _run_and_store(
    target_chunks: list,
    chunk_count: int,
    max_words: int,
    output_format: str,
    focus_instruction: str,
) -> None:
    """
    Runs summarization with map stage caching.
    If map cache exists for current document set,
    skips map stage and only re-runs synthesis.
    """
    current_fingerprint = st.session_state.get(
        "uploaded_files_fingerprint", ""
    )
    cached_fingerprint = st.session_state.get(
        "map_cache_fingerprint", ""
    )
    cached_map = st.session_state.get("map_cache_compressed", None)

    # Use cached map results if same document set
    use_cache = (
        cached_map is not None
        and cached_fingerprint == current_fingerprint
        and len(cached_map) == chunk_count
    )

    if use_cache:
        with st.spinner(
            "Re-synthesizing with new settings "
            "(map stage cached — synthesis only)..."
        ):
            from modes.summarization import (
                _run_synthesis_stage,
                FORMAT_INSTRUCTIONS,
            )
            from prompts.summarization_prompts import FORMAT_INSTRUCTIONS
            summary = _run_synthesis_stage(
                compressed_summaries=cached_map,
                max_words=max_words,
                output_format=output_format,
                focus_instruction=focus_instruction,
            )
            word_count = len(summary.split())
            raw_result = {
                "summary": summary,
                "word_count": word_count,
                "chunks_processed": chunk_count,
                "output_format": output_format,
                "needs_cost_confirmation": False,
                "estimated_map_calls": 0,
                "warnings": ["Map stage used cached results."],
            }
    else:
        with st.spinner(
            f"Summarizing {chunk_count} chunks... "
            f"This may take a few minutes."
        ):
            from modes.summarization import _run_map_stage
            compressed = _run_map_stage(target_chunks)

            # Cache map results for this document set
            st.session_state["map_cache_compressed"] = compressed
            st.session_state["map_cache_fingerprint"] = (
                current_fingerprint
            )

            from modes.summarization import _run_synthesis_stage
            summary = _run_synthesis_stage(
                compressed_summaries=compressed,
                max_words=max_words,
                output_format=output_format,
                focus_instruction=focus_instruction,
            )
            word_count = len(summary.split())
            raw_result = {
                "summary": summary,
                "word_count": word_count,
                "chunks_processed": chunk_count,
                "output_format": output_format,
                "needs_cost_confirmation": False,
                "estimated_map_calls": chunk_count,
                "warnings": [],
            }

    st.session_state["summary_result"] = (
        format_summary_result(raw_result)
    )

    # Update cost tracker
    from utils.cost_tracker import add_cost_entry
    token_log = st.session_state.get("session_token_log", [])
    if not use_cache:
        add_cost_entry(
            token_log=token_log,
            operation="summarization_map",
            model="gpt-4o-mini",
            input_tokens=chunk_count * 800,
            output_tokens=chunk_count * 200,
            actual_calls=chunk_count,
        )
    add_cost_entry(
        token_log=token_log,
        operation="summarization_synthesis",
        model="gpt-4o",
        input_tokens=8000,
        output_tokens=600,
    )
    st.session_state["session_token_log"] = token_log


def _render_result(result: dict, output_format: str = "") -> None:
    """Renders the summary result if available."""
    if not result or not result.get("summary"):
        return

    st.divider()

    col_meta, col_export = st.columns([3, 1])
    with col_meta:
        st.caption(result["meta_label"])

    for warning in result.get("warnings", []):
        st.warning(warning)

    with col_export:
        st.download_button(
            label="📥 Export",
            data=result["summary"],
            file_name="summary.md",
            mime="text/markdown",
        )

    # ── Mermaid: render live diagram instead of raw code ──────
    if output_format in MERMAID_FORMATS:
        clean_code = extract_mermaid_code(result["summary"])
        if clean_code:
            diagram_type = (
                "Flowchart" if output_format == "mermaid_flowchart"
                else "Mindmap"
            )
            st.markdown(f"#### 🗺️ {diagram_type} Diagram")
            render_mermaid(clean_code, height=estimate_height(clean_code))
            with st.expander("View Mermaid code"):
                st.code(clean_code, language="text")
        else:
            st.warning("No diagram code found in output. Try regenerating.")
    else:
        st.markdown(result["summary"])