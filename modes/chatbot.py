# ============================================================
# modes/chatbot.py
# ============================================================
# Mode 3: Multi-Document RAG Chatbot chain.
# Responsibilities:
#   - Build conversational retrieval chain from vector store
#   - Rewrite follow-up questions for better retrieval
#   - Retrieve and rerank chunks per query
#   - Generate grounded answers with source citations
#
# Does NOT import from Streamlit.
# Chat history is managed by ui/tab_chatbot.py via session state.
# ============================================================

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, AIMessage
from core.router import get_model, TASK_RAG, TASK_CLASSIFICATION
from core.embeddings import get_vectorstore_retriever
from prompts.rag_prompts import RAG_PROMPT, QUESTION_REWRITER_PROMPT
from config import RAG_K, CHAT_MEMORY_K


# ------------------------------------------------------------
# Question Rewriter
# Makes follow-up questions self-contained for retrieval
# ------------------------------------------------------------

def rewrite_question(
    question: str,
    chat_history: list[dict],
) -> str:
    """
    Rewrites a follow-up question to be self-contained.
    Critical for accurate retrieval when questions reference
    earlier conversation (e.g. "what about the second one?")

    Args:
        question: Current user question.
        chat_history: List of {role, content} dicts.

    Returns:
        Rewritten standalone question string.
    """
    if not chat_history:
        return question

    model = get_model(TASK_CLASSIFICATION)
    chain = QUESTION_REWRITER_PROMPT | model

    history_text = "\n".join([
        f"{msg['role'].title()}: {msg['content']}"
        for msg in chat_history[-4:]
    ])

    result = chain.invoke({
        "chat_history": history_text,
        "question": question,
    })

    return result.content.strip()


# ------------------------------------------------------------
# Context Retrieval and Reranking
# ------------------------------------------------------------

def retrieve_and_rerank(
    question: str,
    vectorstore: FAISS,
    k: int = RAG_K,
) -> list[Document]:
    """
    Retrieves top-k chunks from vector store and reranks
    using cross-encoder for improved relevance.

    Args:
        question: Standalone question (after rewriting).
        vectorstore: Built FAISS vector store.
        k: Number of chunks to retrieve.

    Returns:
        Reranked list of Document chunks.
    """
    retriever = get_vectorstore_retriever(vectorstore, k=k)
    chunks = retriever.invoke(question)

    # Rerank using cross-encoder if available
    try:
        from sentence_transformers import CrossEncoder
        reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

        pairs = [[question, chunk.page_content] for chunk in chunks]
        scores = reranker.predict(pairs)

        scored_chunks = list(zip(scores, chunks))
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        chunks = [chunk for _, chunk in scored_chunks]

    except Exception:
        # Reranker unavailable — use vector similarity order
        pass

    return chunks


# ------------------------------------------------------------
# Context Formatter
# Formats retrieved chunks for injection into RAG prompt
# ------------------------------------------------------------

def format_context(chunks: list[Document]) -> str:
    """
    Formats retrieved chunks into a structured context string
    for injection into the RAG system prompt.

    Includes source metadata with each chunk so the model
    can produce accurate citations.

    Args:
        chunks: Retrieved and reranked Document chunks.

    Returns:
        Formatted context string with source labels.
    """
    context_parts = []

    for i, chunk in enumerate(chunks):
        filename = chunk.metadata.get("source_filename", "unknown")
        page = chunk.metadata.get("page_number", "?")
        block_type = chunk.metadata.get("block_type", "PROSE")

        header = (
            f"[Chunk {i+1} | Source: {filename}, "
            f"Page {page} | Type: {block_type}]"
        )
        context_parts.append(f"{header}\n{chunk.page_content}")

    return "\n\n".join(context_parts)


# ------------------------------------------------------------
# Chat History Formatter
# Converts session state history to LangChain message format
# ------------------------------------------------------------

def format_chat_history(chat_history: list[dict]) -> list:
    """
    Converts the session state chat history list into
    LangChain message objects for the prompt template.

    Args:
        chat_history: List of {role, content} dicts from
                      st.session_state["chat_history"]

    Returns:
        List of HumanMessage and AIMessage objects.
    """
    messages = []
    for msg in chat_history[-CHAT_MEMORY_K * 2:]:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))
    return messages


# ------------------------------------------------------------
# Main Entry Point
# ------------------------------------------------------------

def run_chat(
    question: str,
    vectorstore: FAISS,
    chat_history: list[dict],
    k: int = RAG_K,
) -> dict:
    """
    Runs a single RAG chat turn.

    Pipeline:
    1. Rewrite question for self-contained retrieval
    2. Retrieve and rerank relevant chunks
    3. Format context with source metadata
    4. Generate grounded answer with citations

    Args:
        question: User's current question.
        vectorstore: Built FAISS vector store.
        chat_history: Full conversation history from session state.
        k: Number of chunks to retrieve.

    Returns:
        Response dict:
        {
            "answer": str,
            "sources": list[dict],
            "rewritten_question": str,
            "chunks_used": int,
        }
    """
    # Step 1 — Rewrite question for better retrieval
    standalone_question = rewrite_question(question, chat_history)

    # Step 2 — Retrieve and rerank
    chunks = retrieve_and_rerank(
        question=standalone_question,
        vectorstore=vectorstore,
        k=k,
    )

    if not chunks:
        return {
            "answer": (
                "No relevant content was found in the uploaded documents "
                "for your question. Please try rephrasing or uploading "
                "additional documents."
            ),
            "sources": [],
            "rewritten_question": standalone_question,
            "chunks_used": 0,
        }

    # Step 3 — Format context
    context = format_context(chunks)

    # Step 4 — Generate answer
    model = get_model(TASK_RAG)
    chain = RAG_PROMPT | model

    formatted_history = format_chat_history(chat_history)

    result = chain.invoke({
        "context": context,
        "chat_history": formatted_history,
        "question": question,
    })

    # Extract unique sources from retrieved chunks
    sources = []
    seen = set()
    for chunk in chunks:
        filename = chunk.metadata.get("source_filename", "unknown")
        page = chunk.metadata.get("page_number", "?")
        key = f"{filename}:{page}"
        if key not in seen:
            seen.add(key)
            sources.append({
                "filename": filename,
                "page": page,
            })

    return {
        "answer": result.content.strip(),
        "sources": sources,
        "rewritten_question": standalone_question,
        "chunks_used": len(chunks),
    }