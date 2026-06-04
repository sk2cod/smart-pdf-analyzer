# ============================================================
# core/router.py
# ============================================================
# THE single point of model assignment for the entire app.
# Every mode (extraction, summarization, RAG) calls
# get_model(task_type) from here — never instantiates
# models directly.
#
# To switch providers or models:
#   1. Update model names in config.py
#   2. Swap ChatOpenAI for ChatAnthropic/ChatGoogleGenerativeAI
#      in this file only
#   3. Nothing else in the codebase changes
# ============================================================

from langchain_openai import ChatOpenAI
from config import (
    CHEAP_MODEL,
    PREMIUM_MODEL,
    EXTRACTION_TEMPERATURE,
    RAG_TEMPERATURE,
    SUMMARIZATION_MAP_TEMPERATURE,
    SUMMARIZATION_SYNTHESIS_TEMPERATURE,
)

# ------------------------------------------------------------
# Task type constants
# Use these strings when calling get_model() to avoid typos
# ------------------------------------------------------------
TASK_EXTRACTION: str = "extraction"
TASK_EXTRACTION_VISION: str = "extraction_vision"
TASK_SUMMARIZATION_MAP: str = "summarization_map"
TASK_SUMMARIZATION_SYNTHESIS: str = "summarization_synthesis"
TASK_RAG: str = "rag"
TASK_CLASSIFICATION: str = "classification"


def get_model(task_type: str) -> ChatOpenAI:
    """
    Returns the appropriate ChatOpenAI instance for the given task.

    Routing rules:
    - CHEAP_MODEL  → extraction, classification, RAG, map-stage summarization
    - PREMIUM_MODEL → synthesis-stage summarization, vision on low quality images

    Args:
        task_type: One of the TASK_* constants defined above.

    Returns:
        Configured ChatOpenAI instance.

    Raises:
        ValueError: If task_type is not recognised.
    """

    routing_map = {

        # Extraction from text PDFs — cheap model fully capable
        TASK_EXTRACTION: ChatOpenAI(
            model=CHEAP_MODEL,
            temperature=EXTRACTION_TEMPERATURE,
        ),

        # Extraction from photo/scanned PDFs — premium model
        # only when user confirms enhanced path or image
        # quality check fails thresholds in ingestion.py
        TASK_EXTRACTION_VISION: ChatOpenAI(
            model=PREMIUM_MODEL,
            temperature=EXTRACTION_TEMPERATURE,
        ),

        # Map stage — compressing individual chunks
        # Cheap model handles compression well
        TASK_SUMMARIZATION_MAP: ChatOpenAI(
            model=CHEAP_MODEL,
            temperature=SUMMARIZATION_MAP_TEMPERATURE,
        ),

        # Synthesis stage — final reasoning and generation
        # Premium model only call in summarization pipeline
        TASK_SUMMARIZATION_SYNTHESIS: ChatOpenAI(
            model=PREMIUM_MODEL,
            temperature=SUMMARIZATION_SYNTHESIS_TEMPERATURE,
        ),

        # RAG answer generation — grounded by retrieved chunks
        # Cheap model sufficient as it reads, not reasons freely
        TASK_RAG: ChatOpenAI(
            model=CHEAP_MODEL,
            temperature=RAG_TEMPERATURE,
        ),

        # Document type classification — single classification call
        # Cheap model fully capable of this simple task
        TASK_CLASSIFICATION: ChatOpenAI(
            model=CHEAP_MODEL,
            temperature=0.0,
        ),
    }

    if task_type not in routing_map:
        raise ValueError(
            f"Unknown task_type: '{task_type}'. "
            f"Must be one of: {list(routing_map.keys())}"
        )

    return routing_map[task_type]


def get_model_name(task_type: str) -> str:
    """
    Returns the model name string for a given task type.
    Used by cost_tracker.py to log which model was called.

    Args:
        task_type: One of the TASK_* constants.

    Returns:
        Model name string (e.g. 'gpt-4o-mini').
    """
    premium_tasks = {
        TASK_EXTRACTION_VISION,
        TASK_SUMMARIZATION_SYNTHESIS,
    }
    return PREMIUM_MODEL if task_type in premium_tasks else CHEAP_MODEL