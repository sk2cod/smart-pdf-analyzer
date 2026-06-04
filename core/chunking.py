# ============================================================
# core/chunking.py
# ============================================================
# Table-aware text chunking strategy.
# Responsibilities:
#   - Split prose blocks using RecursiveCharacterTextSplitter
#   - Split table blocks on row boundaries only (never mid-row)
#   - Stamp every chunk with full source metadata
#   - Preserve block_type tag for downstream rendering
#
# Does NOT load PDFs, embed, or call any LLM.
# Does NOT import from Streamlit.
# ============================================================

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP, TABLE_CHUNK_SIZE


# ------------------------------------------------------------
# Prose Splitter
# Standard recursive splitter for regular text content
# ------------------------------------------------------------

def _get_prose_splitter() -> RecursiveCharacterTextSplitter:
    """
    Returns a configured splitter for prose/regular text.
    Splits on paragraph breaks first, then sentences,
    then words — preserving as much semantic coherence
    as possible at each level.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
        is_separator_regex=False,
    )


# ------------------------------------------------------------
# Table-Aware Splitter
# Splits only on row boundaries to preserve table structure
# ------------------------------------------------------------

def _split_table_block(text: str, metadata: dict) -> list[Document]:
    """
    Splits a table block on row boundaries only.
    Never splits mid-row to preserve data relationships.

    Strategy:
    - Split on newline (each row is a new line in extracted text)
    - Group rows into chunks not exceeding TABLE_CHUNK_SIZE
    - Each chunk gets the full source metadata plus row range

    Args:
        text: Raw table text extracted from the PDF page.
        metadata: Source metadata dict to attach to chunks.

    Returns:
        List of Document objects, one per table chunk.
    """
    rows = [r for r in text.split("\n") if r.strip()]
    chunks = []
    current_chunk_rows = []
    current_size = 0
    start_row = 1

    for row_index, row in enumerate(rows):
        row_size = len(row)

        # If adding this row would exceed limit and we have content,
        # save current chunk and start fresh
        if current_size + row_size > TABLE_CHUNK_SIZE and current_chunk_rows:
            chunk_text = "\n".join(current_chunk_rows)
            chunk_metadata = {
                **metadata,
                "block_type": "TABLE",
                "table_row_start": start_row,
                "table_row_end": row_index,
            }
            chunks.append(Document(
                page_content=chunk_text,
                metadata=chunk_metadata,
            ))
            current_chunk_rows = []
            current_size = 0
            start_row = row_index + 1

        current_chunk_rows.append(row)
        current_size += row_size

    # Flush remaining rows
    if current_chunk_rows:
        chunk_text = "\n".join(current_chunk_rows)
        chunk_metadata = {
            **metadata,
            "block_type": "TABLE",
            "table_row_start": start_row,
            "table_row_end": len(rows),
        }
        chunks.append(Document(
            page_content=chunk_text,
            metadata=chunk_metadata,
        ))

    return chunks


# ------------------------------------------------------------
# Main Entry Point
# ------------------------------------------------------------

def chunk_documents(documents: list[Document]) -> list[Document]:
    """
    Applies table-aware chunking to a list of Documents.
    Each Document represents one PDF page from ingestion.py.

    Pipeline:
    1. Skip image-based pages (handled by vision extraction)
    2. Route table pages to table-aware splitter
    3. Route prose pages to recursive character splitter
    4. Stamp every chunk with full metadata + chunk_index

    Args:
        documents: List of Documents from ingestion.load_pdfs()

    Returns:
        Flat list of chunk Documents ready for embedding.
    """
    prose_splitter = _get_prose_splitter()
    all_chunks: list[Document] = []
    chunk_index = 0

    for doc in documents:

        # Skip image pages — they go through vision extraction
        # in modes/extraction.py, not through RAG chunking
        if doc.metadata.get("needs_vision", False):
            continue

        # Skip empty pages (common in PDFs with blank separators)
        if not doc.page_content.strip():
            continue

        base_metadata = {
            "source_filename": doc.metadata.get("source_filename", "unknown"),
            "page_number": doc.metadata.get("page_number", 0),
            "doc_index": doc.metadata.get("doc_index", 0),
            "total_pages": doc.metadata.get("total_pages", 0),
        }

        # Route based on whether page contains table content
        if doc.metadata.get("has_table", False):
            page_chunks = _split_table_block(
                text=doc.page_content,
                metadata=base_metadata,
            )
        else:
            # Standard prose splitting
            split_docs = prose_splitter.split_documents([doc])
            page_chunks = []
            for split_doc in split_docs:
                page_chunks.append(Document(
                    page_content=split_doc.page_content,
                    metadata={
                        **base_metadata,
                        "block_type": "PROSE",
                    },
                ))

        # Stamp every chunk with a global chunk index
        for chunk in page_chunks:
            chunk.metadata["chunk_index"] = chunk_index
            chunk_index += 1
            all_chunks.append(chunk)

    return all_chunks