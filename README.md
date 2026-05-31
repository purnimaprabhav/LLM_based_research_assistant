# LLM-Based Research Assistant

> A RAG pipeline for semantic querying of large document corpora — built to go beyond keyword search and return synthesised, context-aware answers from technical knowledge bases.

---

## Overview

This project is a personal prototype exploring how Retrieval-Augmented Generation (RAG) can make large, unstructured document collections actually queryable. The problem it solves is straightforward: keyword search fails on technical documents because the answer you need is rarely phrased the way you asked the question.

The system ingests PDFs, extracts and chunks text, embeds it into a FAISS vector store, and uses GPT-4o-mini via LangChain's `RetrievalQA` chain to synthesise answers grounded in retrieved document chunks. The filtering layer enables automatic analysis and structured summarisation of technical content — useful in decision-support contexts where you need insight, not raw excerpts.

This started as a hands-on exploration of where RAG works well and where it breaks. The architecture is intentionally kept simple to make those failure modes visible.

---

## Architecture

```
PDF Documents
     │
     ▼
[ PDF Parser ]          ← pdfplumber + PyPDF2 + pdfminer.six (fallback chain)
     │
     ▼
[ Text Chunker ]        ← tiktoken-aware chunking to respect GPT context limits
     │
     ▼
[ OpenAI Embeddings ]   ← text-embedding-ada-002 via langchain-openai
     │
     ▼
[ FAISS Index ]         ← faiss-cpu, local vector store (FAISSStore wrapper)
     │
     ▼
[ Retriever ]           ← similarity search, top-k chunks returned
     │
     ▼
[ RetrievalQA Chain ]   ← LangChain, chain_type="stuff", GPT-4o-mini
     │
     ▼
[ Synthesised Answer ]
```

**Core class:** `ResearchAssistant` orchestrates the full pipeline:
- `ingest_pdfs(pdf_paths)` — parses PDFs and builds the FAISS index
- `setup_qa()` — initialises the `RetrievalQA` chain with the retriever and LLM
- `ask(question)` — runs a natural language query end-to-end

---

## Key Technical Choices

- **PDF parsing fallback chain:** `pdfplumber` → `PyPDF2` → `pdfminer.six` — handles scanned, malformed, and text-native PDFs without crashing on edge cases.
- **tiktoken-aware chunking:** chunks are sized to respect GPT-4o-mini's context window, avoiding silent truncation during generation.
- **FAISS (CPU):** local vector store — no external service dependency, fast enough for corpora of hundreds of documents on a standard machine.
- **`chain_type="stuff"`:** passes all retrieved chunks directly into the prompt. Simple and transparent — easier to debug retrieval quality than map-reduce or refine chains.
- **`temperature=0`:** deterministic outputs, important for technical content where consistency matters.
- **NLP preprocessing:** `spacy` and `nltk` used for text cleaning and sentence boundary detection before chunking.
- **Async support:** `aiohttp` included for future async ingestion of web-sourced documents alongside PDFs.

---

## Stack

| Layer | Library / Tool |
|---|---|
| LLM | GPT-4o-mini (`langchain-openai`) |
| Embeddings | OpenAI `text-embedding-ada-002` |
| Vector Store | FAISS (`faiss-cpu`) |
| Orchestration | LangChain (`RetrievalQA`, `langchain-core`, `langchain-community`) |
| PDF Parsing | `pdfplumber`, `PyPDF2`, `pdfminer.six` |
| Tokenisation | `tiktoken` |
| NLP | `spacy`, `nltk` |
| Web scraping (planned) | `beautifulsoup4`, `aiohttp`, `requests` |
| Config | `python-dotenv` |

---

## Project Structure

```
├── research_assistant/
│   ├── vectorstore/
│   │   └── faiss_store.py       # FAISSStore: build_index(), as_retriever()
│   ├── utils/
│   │   └── pdf_parser.py        # extract_text_from_pdf() with fallback chain
│   └── core.py                  # ResearchAssistant class
├── scripts/                     # Ingestion and query entry points
├── requirements.txt
├── .env.example
└── README.md
```

---

## Setup

1. **Clone the repo**

```bash
git clone https://github.com/purnimaprabhav/LLM_based_research_assistant.git
cd LLM_based_research_assistant
```

2. **Create a virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm  # spaCy language model
```

4. **Configure your API key**

```bash
cp .env.example .env
# Add your key to .env:
# OPENAI_API_KEY=sk-...
```

5. **Ingest your documents**

```python
from research_assistant.core import ResearchAssistant

assistant = ResearchAssistant()
assistant.ingest_pdfs(["doc1.pdf", "doc2.pdf"])
assistant.setup_qa()
```

6. **Ask a question**

```python
answer = assistant.ask("What are the key findings on X?")
print(answer)
```

---

## Current Limitations / What's Next

- **Chunking is static** — fixed-size, tiktoken-aware but not semantically aware. Sentence-boundary chunking via `spacy` is the natural next step.
- **`chain_type="stuff"`** breaks on very large result sets where retrieved chunks exceed the context window. Map-reduce or refine chains would handle this better.
- **No reranking** — retrieved chunks are returned purely by cosine similarity. A cross-encoder reranker (e.g. `sentence-transformers`) would improve answer quality on ambiguous queries.
- **No evaluation framework** — retrieval precision and answer faithfulness are not yet measured. RAGAS integration is planned.
- **Web ingestion** (`beautifulsoup4`, `aiohttp`) is scaffolded but not yet wired into the main pipeline.
- **No UI** — CLI only for now. Streamlit frontend is a natural next step.

---

## What I Learned

RAG is deceptively simple to prototype and genuinely hard to do well. The failure modes are almost never in the LLM — they're in the retrieval step: chunks that lose context when split, embeddings that retrieve topically related but semantically wrong passages, and prompts that don't tell the model what to do when the retrieved context doesn't contain the answer.

Building this made those problems concrete, which is the point.

---

## Author

**Venkata Purnima PRABHA**
[GitHub](https://github.com/purnimaprabhav) 
---

## Contributing

Pull requests and issues are welcome. Please open an issue to discuss major changes before submitting a PR.
