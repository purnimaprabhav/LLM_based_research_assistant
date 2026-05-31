* LLM-Based Research Assistant *
** Overview **
This project is a prototype of an AI-powered research assistant designed to tackle a practical problem: navigating large, dense knowledge bases efficiently. Instead of relying on keyword search, the system enables semantic querying — understanding the intent behind a question and retrieving contextually relevant information from documents.
The core idea is straightforward: ingest a knowledge base, embed it into a vector store, and use an LLM to synthesise answers rather than just returning raw chunks. The added filtering layer allows automatic analysis and structured summarisation of technical content, making it useful for decision-support contexts where you need insight, not just search results.
This started as a personal exploration into Generative AI applied to knowledge management. The goal was to understand, hands-on, how RAG pipelines actually behave at scale — where they work well and where they break.

Main Architecture

Document Ingestion Pipeline: Loads and chunks documents from the knowledge base, preparing them for embedding.
Vector Store: Embeds document chunks using sentence-level models and stores them for semantic retrieval.
Retrieval Module: Runs similarity search against the vector store based on the user's natural language query.
LLM Synthesis Layer: Passes retrieved chunks to an LLM to generate a coherent, context-aware answer.
Intelligent Filters: Post-retrieval filtering logic for automatic analysis and structured synthesis of technical reports.


Features

Semantic querying over large document collections — no keyword matching required.
Modular separation between ingestion, retrieval, and generation concerns.
Intelligent filtering for technical report analysis and synthesis.
Designed with decision-support use cases in mind: outputs are structured summaries, not raw document excerpts.
Clean separation between the research_assistant core and scripts layer for flexibility and extensibility.


Methodology and Tools

Language Models: OpenAI / compatible LLM backends
Frameworks: Python, LangChain, Sentence-Transformers
AI/ML Techniques: RAG (Retrieval-Augmented Generation), vector embeddings, semantic similarity search, chunking strategies
Vector Store: FAISS (local) — adaptable to other stores
Approach: Prototype-first, iterative — built to understand the failure modes of RAG in practice, not just the happy path


Use Cases

Semantic search over internal technical documentation or research corpora
Automatic synthesis of lengthy technical reports for faster decision-making
Knowledge base querying for teams dealing with high volumes of unstructured documents
Foundation for a production-ready document intelligence assistant


Project Structure
├── research_assistant/     # Core RAG pipeline: ingestion, retrieval, generation, filtering
├── scripts/                # Utility scripts for data prep, indexing, and running queries
├── requirements.txt        # Python dependencies
├── .gitignore
├── README.md

Setup Instructions

Clone the repository

bashgit clone https://github.com/purnimaprabhav/LLM_based_research_assistant.git
cd LLM_based_research_assistant

Create a virtual environment

bashpython -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

Install dependencies

bashpip install -r requirements.txt

Set up your API key

bashexport OPENAI_API_KEY="your-api-key-here"

Ingest your documents

Place your documents in the designated data folder and run the ingestion script:
bashpython scripts/ingest.py

Run a query

bashpython scripts/query.py --question "Your question here"

Current Limitations / Future Work

Feedback loop and query history are not yet implemented — planned for a future iteration.
Chunking strategy is static; adaptive chunking based on document type would improve retrieval quality.
UI layer (e.g. Streamlit) not yet built — currently CLI only.
Evaluation framework (retrieval precision, answer faithfulness) is a natural next step before any production use.


Summary
This project demonstrates that semantic querying meaningfully outperforms keyword search for navigating complex knowledge bases — particularly for technical documents where the answer rarely lives in a single sentence. The synthesis layer adds real value by reducing cognitive load for the end user.
It also surfaced some honest lessons: RAG works well when documents are well-structured and chunked thoughtfully; it degrades quickly when chunks lose context or when the retrieval step returns marginally relevant material. Those are the problems worth solving next.

Author
Venkata Purnima PRABHA

Contributing
Pull requests and issues are welcome. Please open an issue to discuss major changes before submitting a PR.
