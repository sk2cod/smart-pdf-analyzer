# ============================================================
# core/embeddings.py
# ============================================================
# Manages the FAISS vector store lifecycle.
# Responsibilities:
#   - Build vector store from chunks (called once per file set)
#   - Cache vector store locally during development
#   - Load cached vector store to avoid re-embedding
#   - Expose fingerprint utility for session state guard
#
# Does NOT chunk documents or call chat models.
# Does NOT import from Streamlit.
# ============================================================

import os
import shutil
from pathlib import Path
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from config import (
    EMBEDDING_MODEL,
    VECTORSTORE_CACHE_PATH,
)


# ------------------------------------------------------------
# Embeddings Instance
# Single instance reused across all operations
# ------------------------------------------------------------

def _get_embeddings() -> OpenAIEmbeddings:
    """
    Returns a configured OpenAIEmbeddings instance.
    Uses text-embedding-3-small — cheapest embedding model,
    sufficient quality for document retrieval tasks.
    """
    return OpenAIEmbeddings(model=EMBEDDING_MODEL)


# ------------------------------------------------------------
# Vector Store Builder
# ------------------------------------------------------------

def build_vectorstore(chunks: list[Document]) -> FAISS:
    """
    Builds a FAISS vector store from a list of chunk Documents.
    This is the only function that calls the embeddings API
    and incurs embedding costs.

    Called exactly once per unique file set upload,
    protected by the fingerprint guard in ui/sidebar.py.

    Args:
        chunks: List of Documents from chunking.chunk_documents()

    Returns:
        FAISS vector store loaded in memory.

    Raises:
        ValueError: If chunks list is empty.
    """
    if not chunks:
        raise ValueError(
            "Cannot build vector store from empty chunk list. "
            "Ensure PDFs were loaded and chunked successfully."
        )

    embeddings = _get_embeddings()

    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings,
    )

    # Cache to disk during local development
    # This allows reloading without re-embedding on dev server restart
    # On Streamlit Cloud this write is ephemeral and session-scoped
    try:
        cache_path = Path(VECTORSTORE_CACHE_PATH)
        cache_path.mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(str(cache_path))
    except Exception:
        # Cache write failure is non-fatal
        # Vector store still lives in memory and works normally
        pass

    return vectorstore


# ------------------------------------------------------------
# Vector Store Loader
# ------------------------------------------------------------

def load_cached_vectorstore() -> FAISS | None:
    """
    Attempts to load a cached vector store from disk.
    Used during local development to skip re-embedding
    after a dev server restart.

    Returns:
        FAISS vector store if cache exists, None otherwise.
    """
    cache_path = Path(VECTORSTORE_CACHE_PATH)

    if not cache_path.exists():
        return None

    try:
        embeddings = _get_embeddings()
        vectorstore = FAISS.load_local(
            folder_path=str(cache_path),
            embeddings=embeddings,
            allow_dangerous_deserialization=True,
        )
        return vectorstore
    except Exception:
        # Cache is corrupted or incompatible — ignore and rebuild
        return None


# ------------------------------------------------------------
# Cache Management
# ------------------------------------------------------------

def clear_vectorstore_cache() -> None:
    """
    Deletes the local vector store cache directory.
    Called when a new file set is uploaded to ensure
    stale embeddings do not persist across sessions.
    """
    cache_path = Path(VECTORSTORE_CACHE_PATH)
    if cache_path.exists():
        shutil.rmtree(cache_path)


def get_vectorstore_retriever(
    vectorstore: FAISS,
    k: int,
) -> object:
    """
    Returns a configured retriever from the vector store.

    Args:
        vectorstore: Built FAISS vector store.
        k: Number of chunks to retrieve per query.

    Returns:
        LangChain retriever object ready for use in RAG chain.
    """
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )