# ============================================================
# modes/extraction.py
# ============================================================
# Mode 1: Data Extraction chain.
# Responsibilities:
#   - Classify document type
#   - Extract fields from text-based pages
#   - Extract fields from image-based pages via vision
#   - Return structured dict with fields + confidence scores
#
# Does NOT import from Streamlit.
# Does NOT render output — that is formatters.py's job.
# ============================================================

import base64
import json
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from core.router import (
    get_model,
    TASK_CLASSIFICATION,
    TASK_EXTRACTION,
    TASK_EXTRACTION_VISION,
)
from prompts.extraction_prompts import (
    CLASSIFICATION_PROMPT,
    EXTRACTION_PROMPT,
    VISION_EXTRACTION_PROMPT,
)
from config import DOCUMENT_TYPES


# ------------------------------------------------------------
# Document Classification
# ------------------------------------------------------------

def classify_document(page_text: str) -> str:
    """
    Classifies the document type from the first page text.
    Uses cheap model — simple single-word classification task.

    Args:
        page_text: Text content of the first page.

    Returns:
        Document type string: IDENTITY, FINANCIAL, LEGAL,
        MEDICAL, REPORT, or UNKNOWN.
    """
    model = get_model(TASK_CLASSIFICATION)
    chain = CLASSIFICATION_PROMPT | model
    result = chain.invoke({"page_text": page_text[:2000]})
    return result.content.strip().upper()


# ------------------------------------------------------------
# Text-Based Extraction
# ------------------------------------------------------------

def extract_from_text(
    page_text: str,
    document_type: str,
) -> dict:
    """
    Extracts structured fields from text-based PDF pages.
    Uses cheap model with anti-hallucination prompt.

    Args:
        page_text: Extracted text from PDF page.
        document_type: Classified or user-selected type.

    Returns:
        Parsed extraction dict with fields and confidence scores.
    """
    model = get_model(TASK_EXTRACTION)
    chain = EXTRACTION_PROMPT | model

    result = chain.invoke({
        "document_type": document_type,
        "page_text": page_text,
    })

    return _parse_extraction_response(result.content)


# ------------------------------------------------------------
# Vision-Based Extraction
# ------------------------------------------------------------

def extract_from_image(
    image_bytes: bytes,
    document_type: str,
    use_premium: bool = False,
) -> dict:
    """
    Extracts structured fields from a scanned/photo PDF page
    using vision capability.

    Args:
        image_bytes: PNG bytes of the rendered PDF page.
        document_type: Classified or user-selected type.
        use_premium: If True, uses gpt-4o regardless of quality.

    Returns:
        Parsed extraction dict with fields and confidence scores.
    """
    task = TASK_EXTRACTION_VISION if use_premium else TASK_EXTRACTION
    model = get_model(task)

    # Build vision message with base64 encoded image
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    # Build system prompt as plain string
    from langchain_core.messages import SystemMessage

    system_text = VISION_EXTRACTION_PROMPT.messages[0].prompt.template

    # Build human text as plain string
    human_text = (
        f"Extract all visible data fields from this document image.\n\n"
        f"Document type context: {document_type}\n\n"
        f"Identify every discrete field you can see. "
        f"Do not use a fixed template. "
        f"Discover the schema from what is visible in the image."
    )

    # Build multimodal message with image
    vision_message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": human_text,
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{image_b64}",
                    "detail": "high",
                },
            },
        ]
    )

    result = model.invoke([
        SystemMessage(content=system_text),
        vision_message,
    ])

    return _parse_extraction_response(result.content)


# ------------------------------------------------------------
# Main Entry Point
# ------------------------------------------------------------

def run_extraction(
    documents: list[Document],
    document_type: str = "auto",
    use_enhanced_vision: bool = False,
) -> dict:
    """
    Runs the full extraction pipeline on uploaded documents.
    Handles both text-based and image-based pages.

    Pipeline:
    1. Classify document type (if auto)
    2. Extract from text pages
    3. Extract from image pages (vision path)
    4. Merge results across pages

    Args:
        documents: List of Documents from ingestion.load_pdfs()
        document_type: User-selected type or 'auto' for detection
        use_enhanced_vision: If True, uses premium model for all
                             image pages regardless of quality

    Returns:
        Combined extraction result dict:
        {
            "document_type": str,
            "fields": dict,
            "confidence": dict,
            "extraction_notes": str,
            "pages_processed": int,
            "vision_pages": int,
            "warnings": list
        }
    """
    if not documents:
        return _empty_result("No documents provided.")

    warnings = []
    all_fields = {}
    all_confidence = {}
    all_notes = []
    pages_processed = 0
    vision_pages = 0

    # Use first available text page for classification
    detected_type = document_type
    if document_type == "auto":
        first_text = next(
            (d.page_content for d in documents
             if not d.metadata.get("needs_vision") and d.page_content.strip()),
            ""
        )
        if first_text:
            detected_type = classify_document(first_text)
        else:
            detected_type = "UNKNOWN"

    # Process each page
    for doc in documents:
        if doc.metadata.get("needs_vision", False):

            # Skip if exceeds vision page limit
            if doc.metadata.get("exceeds_vision_limit", False):
                warnings.append(
                    f"Page {doc.metadata.get('page_number')} skipped "
                    f"(vision page limit reached)."
                )
                continue

            # Determine whether to use premium model
            quality = doc.metadata.get("vision_quality", {})
            passes_gate = quality.get("passes_quality_gate", True)
            use_premium = use_enhanced_vision or not passes_gate

            if not passes_gate and not use_enhanced_vision:
                reasons = quality.get("failure_reasons", [])
                warnings.append(
                    f"Page {doc.metadata.get('page_number')} low quality "
                    f"({', '.join(reasons)}) — using standard model."
                )

            image_bytes = doc.metadata.get("image_bytes", b"")
            if image_bytes:
                result = extract_from_image(
                    image_bytes=image_bytes,
                    document_type=detected_type,
                    use_premium=use_premium,
                )
                _merge_results(all_fields, all_confidence, result)
                vision_pages += 1
                pages_processed += 1

        else:
            if doc.page_content.strip():
                result = extract_from_text(
                    page_text=doc.page_content,
                    document_type=detected_type,
                )
                _merge_results(all_fields, all_confidence, result)
                if result.get("extraction_notes"):
                    all_notes.append(result["extraction_notes"])
                pages_processed += 1

    return {
        "document_type": detected_type,
        "fields": all_fields,
        "confidence": all_confidence,
        "extraction_notes": " | ".join(all_notes),
        "pages_processed": pages_processed,
        "vision_pages": vision_pages,
        "warnings": warnings,
    }


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def _parse_extraction_response(response_text: str) -> dict:
    """
    Parses the LLM's JSON extraction response.
    Handles common formatting issues gracefully.
    """
    # Strip markdown fences if model added them
    text = response_text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1])

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Return safe fallback on parse failure
        return {
            "document_type": "UNKNOWN",
            "fields": {"raw_response": response_text},
            "confidence": {"raw_response": "LOW"},
            "extraction_notes": "JSON parsing failed — raw response preserved.",
        }


def _merge_results(
    all_fields: dict,
    all_confidence: dict,
    result: dict,
) -> None:
    """
    Merges fields and confidence from a page result
    into the running totals. Later pages do not overwrite
    earlier pages for the same field — first occurrence wins.
    """
    fields = result.get("fields", {})
    confidence = result.get("confidence", {})

    for key, value in fields.items():
        if key not in all_fields or all_fields[key] == "NULL":
            all_fields[key] = value
            all_confidence[key] = confidence.get(key, "LOW")


def _empty_result(reason: str) -> dict:
    return {
        "document_type": "UNKNOWN",
        "fields": {},
        "confidence": {},
        "extraction_notes": reason,
        "pages_processed": 0,
        "vision_pages": 0,
        "warnings": [reason],
    }