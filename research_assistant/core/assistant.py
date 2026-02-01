import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env 
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not found in environment")

# LangChain imports
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

# Import your FAISS store
from research_assistant.vectorstore.faiss_store import FAISSStore

# PDF parsing
from research_assistant.utils.pdf_parser import extract_text_from_pdf

class ResearchAssistant:
    def __init__(self):
        # Initialize vector store
        self.vectorstore = FAISSStore()

        # QA chain placeholder
        self.qa_chain = None

    def ingest_pdfs(self, pdf_paths):
        """
        Read PDFs, extract text, and build FAISS embeddings.
        """
        texts = []
        for pdf in pdf_paths:
            texts.append(extract_text_from_pdf(pdf))

        self.vectorstore.build_index(texts)

    def setup_qa(self):
        """
        Setup RetrievalQA chain using vectorstore and LLM.
        """
        retriever = self.vectorstore.as_retriever()
        llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=False
        )

    def ask(self, question):
        """
        Ask a question using the QA chain.
        """
        if self.qa_chain is None:
            raise RuntimeError("QA chain not setup. Call setup_qa() first.")
        return self.qa_chain.run(question)
