# ============================================================
# config.py
# ============================================================
# SINGLE SOURCE OF TRUTH for all application constants.
# Every other file imports from here.
# To change a model, chunk size, or threshold —
# edit this file only. Nothing else needs to change.
# ============================================================

import os
import streamlit as st
from dotenv import load_dotenv

# ------------------------------------------------------------
# Environment Loading
# Tries Streamlit Cloud secrets first (production),
# falls back to .env file (local development).
# This single loader works in both environments
# without any code changes.
# ------------------------------------------------------------

def load_secrets() -> None:
    """
    Unified secret loader.
    - Locally: loads from .env via python-dotenv
    - Streamlit Cloud: reads from st.secrets
    Both paths set the same os.environ keys so the
    rest of the application never needs to know which
    environment it is running in.
    """
    try:
        # Streamlit Cloud — secrets injected via dashboard
        if hasattr(st, "secrets") and "OPENAI_API_KEY" in st.secrets:
            os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
            if "LANGCHAIN_API_KEY" in st.secrets:
                os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]
                os.environ["LANGCHAIN_TRACING_V2"] = st.secrets.get(
                    "LANGCHAIN_TRACING_V2", "false"
                )
                os.environ["LANGCHAIN_PROJECT"] = st.secrets.get(
                    "LANGCHAIN_PROJECT", "smart-pdf-analyzer"
                )
    except Exception:
        pass

    # Local development — loads from .env file
    load_dotenv()


# ------------------------------------------------------------
# Model Configuration
# All model assignments live here.
# Changing provider = update these strings + router.py only.
# ------------------------------------------------------------

# Cheap model — used for extraction, RAG, map-stage summarization,
# document classification, and vision on clean images
CHEAP_MODEL: str = "gpt-4o-mini"

# Premium model — used for summarization synthesis and
# vision on low-quality / user-confirmed enhanced images
PREMIUM_MODEL: str = "gpt-4o"

# Embedding model — used once at ingestion time
EMBEDDING_MODEL: str = "text-embedding-3-small"

# Reranking model — runs locally, no API cost
RERANKING_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ------------------------------------------------------------
# Model Temperature Settings
# Lower = more deterministic (better for extraction/RAG)
# Slightly higher = more natural prose (better for summaries)
# ------------------------------------------------------------

EXTRACTION_TEMPERATURE: float = 0.0
RAG_TEMPERATURE: float = 0.0
SUMMARIZATION_MAP_TEMPERATURE: float = 0.1
SUMMARIZATION_SYNTHESIS_TEMPERATURE: float = 0.3

# ------------------------------------------------------------
# Chunking Configuration
# ------------------------------------------------------------

# Standard prose chunk size in characters (~600 words)
CHUNK_SIZE: int = 800

# Overlap between consecutive chunks (~15% of chunk size)
# Prevents key sentences at boundaries from being lost
CHUNK_OVERLAP: int = 120

# Larger chunk size for table blocks
# Keeps entire small tables as single chunks
TABLE_CHUNK_SIZE: int = 1200

# Minimum character count to consider a page as text-based
# Below this threshold → route to vision extraction path
TEXT_CONTENT_THRESHOLD: int = 50

# ------------------------------------------------------------
# Vision / Image Quality Thresholds
# Used by ingestion.py to decide which model handles photo PDFs
# ------------------------------------------------------------

# Minimum mean brightness (0-255) — below = too dark
IMAGE_MIN_BRIGHTNESS: float = 40.0

# Maximum mean brightness (0-255) — above = too washed out
IMAGE_MAX_BRIGHTNESS: float = 220.0

# Minimum contrast score (standard deviation of pixel values)
IMAGE_MIN_CONTRAST: float = 30.0

# Minimum DPI equivalent for acceptable resolution
IMAGE_MIN_DPI: int = 150

# Maximum pages to process via vision extraction
VISION_MAX_PAGES: int = 3

# ------------------------------------------------------------
# RAG Configuration
# ------------------------------------------------------------

# Number of chunks retrieved per query
RAG_K: int = 6

# Number of conversation turns kept in memory
CHAT_MEMORY_K: int = 6

# Maximum chunks before showing cost warning for summarization
SUMMARIZATION_COST_GATE_CHUNKS: int = 40

# ------------------------------------------------------------
# Upload & Processing Limits
# ------------------------------------------------------------

# Maximum total upload size in megabytes
MAX_UPLOAD_MB: int = 50

# Maximum pages for vision extraction before warning
MAX_VISION_PAGES: int = 3

# Supported file types
SUPPORTED_FILE_TYPES: list = ["pdf"]

# ------------------------------------------------------------
# Vector Store
# ------------------------------------------------------------

# Local cache path (development only — gitignored)
# On Streamlit Cloud this path is ephemeral and session-scoped
VECTORSTORE_CACHE_PATH: str = ".vectorstore_cache/"

# ------------------------------------------------------------
# Cost Tracking
# Approximate costs in USD per 1000 tokens
# Update if OpenAI changes pricing
# ------------------------------------------------------------

COST_PER_1K_INPUT = {
    "gpt-4o-mini":              0.000150,
    "gpt-4o":                   0.002500,
    "text-embedding-3-small":   0.000020,
}

COST_PER_1K_OUTPUT = {
    "gpt-4o-mini":              0.000600,
    "gpt-4o":                   0.010000,
    "text-embedding-3-small":   0.000000,
}

# ------------------------------------------------------------
# Summarization Output Formats
# Keys match what the UI dropdown returns.
# Values are passed into the summarization prompt template.
# Adding a new format = add one entry here + one in the prompt.
# ------------------------------------------------------------

SUMMARY_FORMATS: dict = {
    "bullet_hierarchy":  "structured bullet points with hierarchy",
    "executive_prose":   "executive summary in flowing prose",
    "mermaid_flowchart": "Mermaid flowchart diagram",
    "mermaid_mindmap":   "Mermaid mindmap diagram",
    "comparison_table":  "markdown comparison table",
    "analyst_insight":   "Analyst Insight",
}

# ------------------------------------------------------------
# Document Type Schemas for Extraction
# Keys match the UI dropdown.
# Values are descriptive hints passed to the classification prompt.
# The model infers the actual field schema dynamically.
# ------------------------------------------------------------

DOCUMENT_TYPES: dict = {
    "auto":     "Detect document type automatically",
    "invoice":  "Invoice, bill, or purchase order",
    "identity": "Passport, driver licence, or national ID",
    "receipt":  "Physical or digital payment receipt",
    "contract": "Legal contract or agreement",
    "report":   "Business or research report",
    "custom":   "Custom — I will describe the document",
}

# ------------------------------------------------------------
# UI Display Constants
# ------------------------------------------------------------

APP_TITLE: str = "Smart PDF Analyzer"
APP_SUBTITLE: str = "Extract · Summarize · Chat"
APP_ICON: str = "🧠"
APP_VERSION: str = "1.0.0-MVP"

SUMMARY_MIN_WORDS: int = 100
SUMMARY_MAX_WORDS: int = 1500
SUMMARY_DEFAULT_WORDS: int = 500