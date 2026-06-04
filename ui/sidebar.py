# ============================================================
# ui/sidebar.py
# ============================================================
# Renders the sidebar and owns the document processing
# pipeline trigger.
#
# Responsibilities:
#   - File uploader widget
#   - Fingerprint guard (prevents re-embedding)
#   - Process Documents button and pipeline orchestration
#   - Processing status display
#   - Cost tracker display
#   - Session state writes for ingestion results
# ============================================================

import streamlit as st
from config import (
    load_secrets,
    MAX_UPLOAD_MB,
    APP_VERSION,
)
from core.ingestion import load_pdfs, fingerprint_files
from core.chunking import chunk_documents
from core.embeddings import build_vectorstore, clear_vectorstore_cache
from utils.validators import validate_uploaded_files
from utils.cost_tracker import format_cost_display


def render_sidebar() -> None:
    """
    Renders the complete sidebar UI and handles all
    document processing pipeline logic.
    Writes results directly to st.session_state.
    """
    load_secrets()

    with st.sidebar:
        # ── Header ──────────────────────────────────────────
        st.markdown("### 🧠 Smart PDF Analyzer")
        st.caption(f"v{APP_VERSION}")
        st.divider()

        # ── File Uploader ───────────────────────────────────
        st.markdown("#### 📂 Upload Documents")
        st.caption(f"PDF files only · Max {MAX_UPLOAD_MB}MB total")

        uploaded_files = st.file_uploader(
            label="Upload PDFs",
            type=["pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

        # ── Uploaded File List ──────────────────────────────
        if uploaded_files:
            st.markdown("**Uploaded files:**")
            for f in uploaded_files:
                size_mb = f.size / (1024 * 1024)
                st.caption(f"📄 {f.name} ({size_mb:.1f}MB)")

        st.divider()

        # ── Process Button ──────────────────────────────────
        process_clicked = st.button(
            "🚀 Process Documents",
            use_container_width=True,
            type="primary",
            disabled=not uploaded_files,
        )

        # ── Fingerprint Guard ───────────────────────────────
        if process_clicked and uploaded_files:
            is_valid, errors = validate_uploaded_files(uploaded_files)

            if not is_valid:
                for error in errors:
                    st.error(error)
            else:
                new_fingerprint = fingerprint_files(uploaded_files)
                existing_fingerprint = st.session_state.get(
                    "uploaded_files_fingerprint"
                )

                if new_fingerprint == existing_fingerprint:
                    # Same files — skip re-processing entirely
                    st.info("✅ Documents already processed.")
                else:
                    # New file set — clear stale state and reprocess
                    _clear_downstream_state()
                    clear_vectorstore_cache()
                    _run_processing_pipeline(
                        uploaded_files=uploaded_files,
                        fingerprint=new_fingerprint,
                    )

        # ── Processing Status ───────────────────────────────
        if st.session_state.get("ingestion_complete"):
            st.success(
                f"✅ {len(st.session_state.get('active_filenames', []))} "
                f"document(s) ready"
            )
            filenames = st.session_state.get("active_filenames", [])
            for name in filenames:
                st.caption(f"  · {name}")

        if st.session_state.get("ingestion_error"):
            st.error(st.session_state["ingestion_error"])

        st.divider()

        # ── Settings ────────────────────────────────────────
        st.markdown("#### ⚙️ Settings")

        rag_k = st.slider(
            "Retrieval depth (k)",
            min_value=2,
            max_value=12,
            value=st.session_state.get("chat_k", 6),
            help="Number of document chunks retrieved per question",
        )
        st.session_state["chat_k"] = rag_k

        st.divider()

        # ── Cost Tracker ────────────────────────────────────
        st.markdown("#### 💰 Session Cost")
        token_log = st.session_state.get("session_token_log", [])

        if token_log:
            display = format_cost_display(token_log)
            col1, col2 = st.columns(2)
            with col1:
                st.metric("API Calls", display["total_calls"])
            with col2:
                st.metric("Est. Cost", display["total_cost"])
            st.caption(
                f"Input tokens: {display['input_tokens']} · "
                f"Output tokens: {display['output_tokens']}"
            )
        else:
            st.caption("No API calls yet this session.")


# ------------------------------------------------------------
# Processing Pipeline
# ------------------------------------------------------------

def _run_processing_pipeline(
    uploaded_files: list,
    fingerprint: str,
) -> None:
    """
    Runs the full document processing pipeline with
    progress indicators. Writes all results to session state.

    Stages:
    1. Load PDFs → raw_documents
    2. Chunk documents → processed_chunks
    3. Build vector store → vectorstore
    4. Set ingestion_complete = True
    """
    progress = st.progress(0, text="Starting...")

    try:
        # Stage 1 — Load PDFs
        progress.progress(10, text="Loading PDFs...")
        documents, load_warnings = load_pdfs(uploaded_files)

        if not documents:
            st.session_state["ingestion_error"] = (
                "No content could be extracted from the uploaded files."
            )
            progress.empty()
            return

        st.session_state["raw_documents"] = documents

        # Stage 2 — Chunk documents
        progress.progress(40, text="Processing document structure...")
        chunks = chunk_documents(documents)
        st.session_state["processed_chunks"] = chunks

        # Stage 3 — Build vector store
        progress.progress(65, text="Building knowledge base...")

        if chunks:
            vectorstore = build_vectorstore(chunks)
            st.session_state["vectorstore"] = vectorstore
        else:
            st.session_state["vectorstore"] = None
            load_warnings.append(
                "No text chunks produced — "
                "chat mode unavailable for this document set."
            )

        # Stage 4 — Finalise session state
        progress.progress(95, text="Finalising...")

        st.session_state["uploaded_files_fingerprint"] = fingerprint
        st.session_state["active_filenames"] = [
            f.name for f in uploaded_files
        ]
        st.session_state["ingestion_complete"] = True
        st.session_state["ingestion_error"] = None

        # Reset mode-specific results for new document set
        _clear_mode_results()

        # Show any warnings from loading
        for warning in load_warnings:
            st.warning(warning)

        progress.progress(100, text="Complete!")
        progress.empty()

    except Exception as e:
        progress.empty()
        st.session_state["ingestion_error"] = (
            f"Processing failed: {str(e)}"
        )
        st.session_state["ingestion_complete"] = False


# ------------------------------------------------------------
# State Management Helpers
# ------------------------------------------------------------

def _clear_downstream_state() -> None:
    """
    Clears all session state when a new file set is uploaded.
    Prevents stale results from appearing with new documents.
    """
    keys_to_clear = [
        "uploaded_files_fingerprint",
        "raw_documents",
        "processed_chunks",
        "vectorstore",
        "ingestion_complete",
        "ingestion_error",
        "active_filenames",
    ]
    for key in keys_to_clear:
        st.session_state.pop(key, None)

    _clear_mode_results()


def _clear_mode_results() -> None:
    """
    Clears mode-specific results only.
    Called when switching document sets so old results
    do not persist alongside new documents.
    """
    mode_keys = [
        "extraction_result",
        "extraction_confidence",
        "summary_result",
        "summary_confirmed_cost",
        "chat_history",
        "chat_chain",
    ]
    for key in mode_keys:
        st.session_state.pop(key, None)