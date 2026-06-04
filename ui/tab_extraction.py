# ============================================================
# ui/tab_extraction.py
# ============================================================
# Renders the Data Extraction tab (Mode 1).
# Pure renderer — calls modes/extraction.py and displays
# results. No business logic lives here.
# ============================================================

import streamlit as st
from modes.extraction import run_extraction
from utils.formatters import format_extraction_result
from config import DOCUMENT_TYPES


def render_extraction_tab() -> None:
    """
    Renders the complete extraction tab UI.
    Reads from and writes to st.session_state only for
    extraction-specific keys.
    """
    st.markdown("### 📋 Data Extraction")
    st.caption(
        "Extract structured fields from invoices, "
        "identity documents, receipts, and more."
    )

    # Guard — require processed documents
    if not st.session_state.get("ingestion_complete"):
        st.info(
            "👈 Upload and process your documents using "
            "the sidebar to get started."
        )
        return

    documents = st.session_state.get("raw_documents", [])
    filenames = st.session_state.get("active_filenames", [])

    # ── Controls ─────────────────────────────────────────────
    col1, col2 = st.columns([1, 1])

    with col1:
        doc_type = st.selectbox(
            "Document type",
            options=list(DOCUMENT_TYPES.keys()),
            format_func=lambda k: DOCUMENT_TYPES[k],
            index=0,
            help="Select document type or let the app detect it automatically",
        )

    with col2:
        if len(filenames) > 1:
            selected_file = st.selectbox(
                "Source document",
                options=["All documents"] + filenames,
            )
        else:
            selected_file = filenames[0] if filenames else "All documents"
            st.selectbox(
                "Source document",
                options=[selected_file],
                disabled=True,
            )

    # Vision enhancement option
    # Only show if any image-based pages exist
    has_image_pages = any(
        d.metadata.get("needs_vision", False)
        for d in documents
    )

    use_enhanced = False
    if has_image_pages:
        use_enhanced = st.checkbox(
            "🔍 Use enhanced vision for scanned pages "
            "(uses premium model — slightly higher cost)",
            value=False,
        )

    # ── Extract Button ───────────────────────────────────────
    extract_clicked = st.button(
        "🔍 Extract Fields",
        type="primary",
        use_container_width=False,
    )

    if extract_clicked:
        # Filter documents by selected file if needed
        if selected_file == "All documents":
            target_docs = documents
        else:
            target_docs = [
                d for d in documents
                if d.metadata.get("source_filename") == selected_file
            ]

        with st.spinner("Extracting fields..."):
            raw_result = run_extraction(
                documents=target_docs,
                document_type=doc_type,
                use_enhanced_vision=use_enhanced,
            )
            st.session_state["extraction_result"] = (
                format_extraction_result(raw_result)
            )

    # ── Results Display ──────────────────────────────────────
    result = st.session_state.get("extraction_result")

    if result:
        st.divider()

        # Header row with document type and copy button
        col_title, col_copy = st.columns([3, 1])
        with col_title:
            st.markdown(
                f"**Extracted Data** · "
                f"Document type: `{result['document_type']}`"
            )
            st.caption(
                f"{result['pages_processed']} page(s) processed · "
                f"{result['vision_pages']} via vision"
            )
        with col_copy:
            st.code(result["copyable_text"], language=None)

        # Warnings
        for warning in result.get("warnings", []):
            st.warning(warning)

        # Low confidence notice
        if result.get("has_low_confidence"):
            st.info(
                "⚠️ Some fields have low confidence — "
                "marked below. Verify these manually."
            )

        # Field table
        if result.get("rows"):
            for row in result["rows"]:
                col_field, col_value, col_conf = st.columns(
                    [2, 3, 1]
                )

                with col_field:
                    st.markdown(f"**{row['field']}**")

                with col_value:
                    if row["is_null"]:
                        st.markdown(
                            "<span style='color: gray;'>—</span>",
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(row["value"])

                with col_conf:
                    conf = row["confidence"]
                    if conf == "HIGH":
                        st.success("HIGH", icon="✅")
                    elif conf == "MEDIUM":
                        st.warning("MED", icon="⚠️")
                    else:
                        st.error("LOW", icon="❌")

        # Extraction notes
        if result.get("notes"):
            with st.expander("📝 Extraction notes"):
                st.caption(result["notes"])