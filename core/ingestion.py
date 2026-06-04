# ============================================================
# core/ingestion.py
# ============================================================
# Owns all PDF loading logic.
# Responsibilities:
#   - Load PDFs using PyMuPDF (primary) or pdfplumber (fallback)
#   - Tag every page with source metadata
#   - Detect whether a page is text-based or photo/scanned
#   - Assess image quality for vision routing decisions
#   - Return a flat list of LangChain Document objects
#
# Does NOT chunk, embed, or call any LLM.
# Does NOT import from Streamlit.
# ============================================================

import hashlib
import io
from pathlib import Path
from typing import Optional
import fitz  # PyMuPDF
import pdfplumber
from PIL import Image, ImageStat
from langchain_core.documents import Document
from config import (
    TEXT_CONTENT_THRESHOLD,
    IMAGE_MIN_BRIGHTNESS,
    IMAGE_MAX_BRIGHTNESS,
    IMAGE_MIN_CONTRAST,
    IMAGE_MIN_DPI,
    MAX_UPLOAD_MB,
    VISION_MAX_PAGES,
)


# ------------------------------------------------------------
# Image Quality Assessment
# ------------------------------------------------------------

def assess_image_quality(pix: fitz.Pixmap) -> dict:
    """
    Assesses quality of a PDF page rendered as an image.
    Used to decide whether to route to cheap or premium
    vision model.

    Args:
        pix: PyMuPDF Pixmap object of the rendered page.

    Returns:
        Dictionary with quality metrics and routing decision:
        {
            "brightness": float,
            "contrast": float,
            "dpi_equivalent": int,
            "passes_quality_gate": bool,
            "failure_reasons": list[str]
        }
    """
    # Convert PyMuPDF pixmap to PIL Image for analysis
    img_bytes = pix.tobytes("png")
    img = Image.open(io.BytesIO(img_bytes)).convert("L")  # Greyscale

    stat = ImageStat.Stat(img)
    brightness = stat.mean[0]
    contrast = stat.stddev[0]

    # Estimate DPI from pixmap dimensions vs standard A4
    # A4 at 150 DPI = 1240 x 1754 pixels approximately
    dpi_equivalent = int((pix.width / 8.27))  # width / A4 inches

    failure_reasons = []

    if brightness < IMAGE_MIN_BRIGHTNESS:
        failure_reasons.append(f"Too dark (brightness: {brightness:.1f})")
    if brightness > IMAGE_MAX_BRIGHTNESS:
        failure_reasons.append(f"Too washed out (brightness: {brightness:.1f})")
    if contrast < IMAGE_MIN_CONTRAST:
        failure_reasons.append(f"Low contrast (score: {contrast:.1f})")
    if dpi_equivalent < IMAGE_MIN_DPI:
        failure_reasons.append(f"Low resolution (est. DPI: {dpi_equivalent})")

    return {
        "brightness": brightness,
        "contrast": contrast,
        "dpi_equivalent": dpi_equivalent,
        "passes_quality_gate": len(failure_reasons) == 0,
        "failure_reasons": failure_reasons,
    }


# ------------------------------------------------------------
# Page Type Detection
# ------------------------------------------------------------

def is_text_based_page(page: fitz.Page) -> bool:
    """
    Determines whether a PDF page has extractable text content
    or should be treated as an image/scan.

    Args:
        page: PyMuPDF Page object.

    Returns:
        True if page has sufficient text, False if image-based.
    """
    text = page.get_text("text").strip()
    return len(text) >= TEXT_CONTENT_THRESHOLD


# ------------------------------------------------------------
# Single Page Processors
# ------------------------------------------------------------

def process_text_page(
    page: fitz.Page,
    filename: str,
    page_number: int,
    doc_index: int,
    total_pages: int,
) -> Document:
    """
    Extracts text from a standard text-based PDF page.
    Detects table blocks for downstream table-aware chunking.

    Returns:
        LangChain Document with text content and full metadata.
    """
    # Extract raw text
    text = page.get_text("text").strip()

    # Detect table blocks using PyMuPDF's block analysis
    blocks = page.get_text("blocks")
    has_table = False
    for block in blocks:
        # block[6] is block type: 0=text, 1=image
        # Heuristic: blocks with tab characters suggest tables
        if block[6] == 0 and "\t" in block[4]:
            has_table = True
            break

    return Document(
        page_content=text,
        metadata={
            "source_filename": filename,
            "page_number": page_number,
            "doc_index": doc_index,
            "total_pages": total_pages,
            "page_type": "text",
            "has_table": has_table,
            "needs_vision": False,
            "vision_quality": None,
        },
    )


def process_image_page(
    page: fitz.Page,
    filename: str,
    page_number: int,
    doc_index: int,
    total_pages: int,
    page_vision_count: int,
) -> Document:
    """
    Processes a scanned or photo-based PDF page.
    Renders the page to an image, assesses quality,
    and stores the image bytes for vision extraction.

    Returns:
        LangChain Document with image bytes and quality metadata.
        page_content is empty — filled later by vision extraction.
    """
    # Render page to image at 2x scale for better quality
    mat = fitz.Matrix(2.0, 2.0)
    pix = page.get_pixmap(matrix=mat)

    # Assess image quality for routing decision
    quality = assess_image_quality(pix)

    # Store image as PNG bytes in metadata for vision extraction
    img_bytes = pix.tobytes("png")

    # Check vision page limit
    exceeds_limit = page_vision_count > VISION_MAX_PAGES

    return Document(
        page_content="",  # Filled by vision extraction in modes/extraction.py
        metadata={
            "source_filename": filename,
            "page_number": page_number,
            "doc_index": doc_index,
            "total_pages": total_pages,
            "page_type": "image",
            "has_table": False,
            "needs_vision": True,
            "vision_quality": quality,
            "image_bytes": img_bytes,
            "exceeds_vision_limit": exceeds_limit,
        },
    )


# ------------------------------------------------------------
# Main Entry Point
# ------------------------------------------------------------

def load_pdfs(uploaded_files: list) -> tuple[list[Document], list[str]]:
    """
    Primary ingestion function. Loads one or more uploaded
    PDF files and returns a flat list of Document objects,
    one per page, with full metadata attached.

    Args:
        uploaded_files: List of Streamlit UploadedFile objects.

    Returns:
        Tuple of:
        - List of Document objects (one per page)
        - List of warning messages (e.g. vision page limit exceeded)
    """
    all_documents: list[Document] = []
    warnings: list[str] = []

    for doc_index, uploaded_file in enumerate(uploaded_files):

        filename = uploaded_file.name
        file_bytes = uploaded_file.read()

        # Validate file size
        file_size_mb = len(file_bytes) / (1024 * 1024)
        if file_size_mb > MAX_UPLOAD_MB:
            warnings.append(
                f"{filename} exceeds {MAX_UPLOAD_MB}MB limit "
                f"({file_size_mb:.1f}MB) — skipped."
            )
            continue

        try:
            # Open with PyMuPDF
            pdf_document = fitz.open(stream=file_bytes, filetype="pdf")
            total_pages = len(pdf_document)
            vision_page_count = 0

            for page_number in range(total_pages):
                page = pdf_document[page_number]
                display_page = page_number + 1  # 1-indexed for display

                if is_text_based_page(page):
                    doc = process_text_page(
                        page=page,
                        filename=filename,
                        page_number=display_page,
                        doc_index=doc_index,
                        total_pages=total_pages,
                    )
                else:
                    vision_page_count += 1
                    doc = process_image_page(
                        page=page,
                        filename=filename,
                        page_number=display_page,
                        doc_index=doc_index,
                        total_pages=total_pages,
                        page_vision_count=vision_page_count,
                    )

                    if vision_page_count > VISION_MAX_PAGES:
                        warnings.append(
                            f"{filename} has more than {VISION_MAX_PAGES} "
                            f"scanned pages. Only first {VISION_MAX_PAGES} "
                            f"will be processed via vision extraction."
                        )

                all_documents.append(doc)

            pdf_document.close()

        except Exception as e:
            # Fallback to pdfplumber for problematic PDFs
            try:
                all_documents.extend(
                    _fallback_load(
                        file_bytes=file_bytes,
                        filename=filename,
                        doc_index=doc_index,
                    )
                )
            except Exception as fallback_error:
                warnings.append(
                    f"Could not load {filename}: {str(e)}. "
                    f"Fallback also failed: {str(fallback_error)}"
                )

    return all_documents, warnings


def _fallback_load(
    file_bytes: bytes,
    filename: str,
    doc_index: int,
) -> list[Document]:
    """
    Fallback PDF loader using pdfplumber.
    Used when PyMuPDF fails on a particular PDF.

    Returns:
        List of Document objects with basic metadata.
    """
    documents = []

    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        total_pages = len(pdf.pages)

        for page_number, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            documents.append(
                Document(
                    page_content=text.strip(),
                    metadata={
                        "source_filename": filename,
                        "page_number": page_number + 1,
                        "doc_index": doc_index,
                        "total_pages": total_pages,
                        "page_type": "text",
                        "has_table": False,
                        "needs_vision": False,
                        "vision_quality": None,
                        "loader": "pdfplumber_fallback",
                    },
                )
            )

    return documents


# ------------------------------------------------------------
# File Fingerprinting
# ------------------------------------------------------------

def fingerprint_files(uploaded_files: list) -> str:
    """
    Produces a deterministic hash from the uploaded file set.
    Used by session state to detect whether re-embedding
    is necessary.

    The hash is based on sorted filenames and file sizes.
    Same files in any order → same fingerprint.
    Any file added, removed, or replaced → different fingerprint.

    Args:
        uploaded_files: List of Streamlit UploadedFile objects.

    Returns:
        SHA256 hex digest string.
    """
    file_signatures = sorted([
        f"{f.name}:{f.size}"
        for f in uploaded_files
    ])
    combined = "|".join(file_signatures)
    return hashlib.sha256(combined.encode()).hexdigest()