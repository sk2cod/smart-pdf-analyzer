# ============================================================
# app.py
# ============================================================
# Streamlit entry point.
# This file is intentionally thin — it only:
#   1. Configures the page
#   2. Initialises session state schema
#   3. Renders sidebar
#   4. Renders the three tabs
#
# Zero business logic lives here.
# All logic is delegated to ui/, modes/, core/, utils/.
# ============================================================

import streamlit as st
from config import (
    APP_TITLE,
    APP_SUBTITLE,
    APP_ICON,
    RAG_K,
    SUMMARY_DEFAULT_WORDS,
    load_secrets,
)
from ui.sidebar import render_sidebar
from ui.tab_extraction import render_extraction_tab
from ui.tab_summarization import render_summarization_tab
from ui.tab_chatbot import render_chatbot_tab

# ------------------------------------------------------------
# Page Configuration
# Must be the first Streamlit call in the script
# ------------------------------------------------------------

st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------
# Session State Initialisation
# All keys initialised here exactly once.
# Subsequent reruns skip keys that already exist.
# This is the canonical session state schema for the app.
# ------------------------------------------------------------

def _init_session_state() -> None:
    """
    Initialises all session state keys with default values.
    Only runs for keys that do not yet exist — safe to call
    on every rerun without overwriting existing state.
    """
    defaults = {
        # Upload & ingestion state
        "uploaded_files_fingerprint": None,
        "raw_documents": [],
        "processed_chunks": [],
        "vectorstore": None,
        "ingestion_complete": False,
        "ingestion_error": None,
        "active_filenames": [],

        # Mode 1: Extraction state
        "extraction_result": None,
        "extraction_confidence": None,

        # Mode 2: Summarization state
        "summary_max_words": SUMMARY_DEFAULT_WORDS,
        "summary_format": "bullet_hierarchy",
        "summary_result": None,
        "summary_confirmed_cost": False,

        # Mode 3: Chat state
        "chat_history": [],
        "chat_k": RAG_K,

        # Cross-cutting
        "session_token_log": [],
    }

    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


# ------------------------------------------------------------
# App Entry Point
# ------------------------------------------------------------

def main() -> None:
    """
    Main application entry point.
    Initialises state, renders sidebar, renders tabs.
    """
    load_secrets()
    _init_session_state()

    # Page header
    st.title(f"{APP_ICON} {APP_TITLE}")
    st.caption(APP_SUBTITLE)

    # Sidebar — handles upload and processing
    render_sidebar()

    # Main panel — three operational tabs
    tab1, tab2, tab3 = st.tabs([
        "📋 Extract",
        "📝 Summarize",
        "💬 Chat",
    ])

    with tab1:
        render_extraction_tab()

    with tab2:
        render_summarization_tab()

    with tab3:
        render_chatbot_tab()


if __name__ == "__main__":
    main()