# 🧠 Smart PDF Analyzer & Workplace Assistant

A production-grade document intelligence application built with 
Python, LangChain, and Streamlit. Upload one or multiple PDFs 
and interact with them through three powerful modes.

---

## ✨ Features

### 📋 Mode 1 — Data Extraction
- Automatically detects document type (invoice, identity, receipt, contract)
- Extracts structured fields with confidence scoring (HIGH/MEDIUM/LOW)
- Handles both text-based and scanned/photo PDFs via vision extraction
- Anti-hallucination prompt engineering — outputs NULL rather than guessing

### 📝 Mode 2 — Summarization
- Map-reduce pipeline for documents of any length
- User-controlled word limit (100—1500 words)
- Multiple output formats: bullet hierarchy, executive prose, 
  Mermaid flowchart, Mermaid mindmap, comparison table
- Map stage caching — changing format/length only re-runs synthesis
- Cost gate protection before large summarization jobs

### 💬 Mode 3 — Multi-Document RAG Chat
- Chat across one or many PDFs simultaneously
- Exact source citations with filename and page number
- Cross-encoder reranking for improved retrieval quality
- Conversation memory with follow-up question rewriting

---

## 🏗️ Architecture

    smart-pdf-analyzer/
    ├── app.py                  ← Streamlit entry point
    ├── config.py               ← All constants and model config
    ├── core/
    │   ├── ingestion.py        ← PDF loading and metadata tagging
    │   ├── chunking.py         ← Table-aware text splitting
    │   ├── embeddings.py       ← FAISS vector store management
    │   └── router.py           ← Model routing engine
    ├── modes/
    │   ├── extraction.py       ← Extraction chain
    │   ├── summarization.py    ← Map-reduce summarization chain
    │   └── chatbot.py          ← RAG conversational chain
    ├── prompts/
    │   ├── extraction_prompts.py
    │   ├── summarization_prompts.py
    │   └── rag_prompts.py
    ├── ui/
    │   ├── sidebar.py
    │   ├── tab_extraction.py
    │   ├── tab_summarization.py
    │   └── tab_chatbot.py
    └── utils/
        ├── formatters.py
        ├── validators.py
        └── cost_tracker.py

---

## 🤖 Model Routing

| Task | Model | Reason |
|------|-------|--------|
| Document classification | gpt-4o-mini | Simple classification |
| Text extraction | gpt-4o-mini | Deterministic field matching |
| Vision extraction (clean image) | gpt-4o-mini | Sufficient for quality images |
| Vision extraction (low quality) | gpt-4o | Better accuracy on degraded images |
| Summarization map stage | gpt-4o-mini | Compression task |
| Summarization synthesis | gpt-4o | Final reasoning and generation |
| RAG chat | gpt-4o-mini | Grounded answer from context |
| Embeddings | text-embedding-3-small | Cost efficient |

---

## 🚀 Getting Started

### Prerequisites
- Python 3.11
- OpenAI API key (Prototype plan minimum)
- LangSmith API key (optional, for tracing)

### Installation

    git clone https://github.com/sk2cod/smart-pdf-analyzer.git
    cd smart-pdf-analyzer
    python -m venv .venv
    .venv\Scripts\activate
    pip install -r requirements.txt

### Configuration

Create a `.env` file in the project root:

    OPENAI_API_KEY=sk-proj-your-key-here
    LANGCHAIN_TRACING_V2=true
    LANGCHAIN_API_KEY=ls__your-key-here
    LANGCHAIN_PROJECT=smart-pdf-analyzer

### Run

    streamlit run app.py

---

## 💰 Estimated Costs

| Operation | Approx Cost |
|-----------|-------------|
| Process 3 PDFs (embedding) | ~$0.001 |
| Extract fields from invoice | ~$0.001 |
| Summarize 10-page document | ~$0.008 |
| Summarize full book (258 chunks) | ~$0.088 |
| 10 chat questions | ~$0.003 |

---

## 🔒 Security

- API keys stored in `.env` locally — never committed to GitHub
- `.gitignore` excludes all secrets and cache files
- Restricted OpenAI API key recommended for production

---

## 🛠️ Built With

- [Streamlit](https://streamlit.io) — UI framework
- [LangChain](https://langchain.com) — LLM orchestration
- [OpenAI](https://openai.com) — Language models
- [FAISS](https://github.com/facebookresearch/faiss) — Vector store
- [PyMuPDF](https://pymupdf.readthedocs.io) — PDF processing
- [sentence-transformers](https://sbert.net) — Cross-encoder reranking
- [LangSmith](https://smith.langchain.com) — LLM observability

