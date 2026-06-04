# ============================================================
# ui/tab_chatbot.py
# ============================================================
# Renders the Multi-Document Chat tab (Mode 3).
# Pure renderer — calls modes/chatbot.py and manages
# chat history in session state.
# ============================================================

import streamlit as st
from modes.chatbot import run_chat
from utils.formatters import format_sources_display
from config import RAG_K


def render_chatbot_tab() -> None:
    """
    Renders the complete chatbot tab UI.
    Chat history persists in st.session_state["chat_history"]
    and survives tab switches and reruns.
    """
    st.markdown("### 💬 Document Chat")
    st.caption(
        "Ask questions across all your uploaded documents. "
        "Answers include exact source citations."
    )

    # Guard — require processed documents with vector store
    if not st.session_state.get("ingestion_complete"):
        st.info(
            "👈 Upload and process your documents using "
            "the sidebar to get started."
        )
        return

    vectorstore = st.session_state.get("vectorstore")
    if vectorstore is None:
        st.warning(
            "⚠️ No searchable content found in the uploaded "
            "documents. The chat mode requires text-based PDFs."
        )
        return

    filenames = st.session_state.get("active_filenames", [])

    # ── Active Sources Display ───────────────────────────────
    sources_text = " · ".join(filenames)
    st.caption(f"🔍 Searching across: {sources_text}")

    # ── Chat History Controls ────────────────────────────────
    col_space, col_clear = st.columns([4, 1])
    with col_clear:
        if st.button("🗑️ Clear", help="Clear chat history"):
            st.session_state["chat_history"] = []
            st.rerun()

    # Initialise chat history if needed
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    chat_history = st.session_state["chat_history"]

    # ── Chat History Display ─────────────────────────────────
    for message in chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Show sources for assistant messages
            if (
                message["role"] == "assistant"
                and message.get("sources")
            ):
                sources_display = format_sources_display(
                    message["sources"]
                )
                if sources_display:
                    st.caption(sources_display)

    # ── Chat Input ───────────────────────────────────────────
    question = st.chat_input(
        "Ask anything about your documents..."
    )

    if question:
        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(question)

        # Add to history
        chat_history.append({
            "role": "user",
            "content": question,
            "sources": [],
        })

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Searching documents..."):
                k = st.session_state.get("chat_k", RAG_K)
                response = run_chat(
                    question=question,
                    vectorstore=vectorstore,
                    chat_history=chat_history[:-1],
                    k=k,
                )

            # Display answer
            st.markdown(response["answer"])

            # Display sources
            sources_display = format_sources_display(
                response["sources"]
            )
            if sources_display:
                st.caption(sources_display)

            # Debug info in expander
            with st.expander("🔍 Retrieval details", expanded=False):
                st.caption(
                    f"Chunks retrieved: {response['chunks_used']} · "
                    f"Question as searched: "
                    f"_{response['rewritten_question']}_"
                )

        # Add assistant response to history
        chat_history.append({
            "role": "assistant",
            "content": response["answer"],
            "sources": response["sources"],
        })

        # Update session state
        st.session_state["chat_history"] = chat_history