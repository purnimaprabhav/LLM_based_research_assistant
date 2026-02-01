from pathlib import Path
from research_assistant.core.assistant import ResearchAssistant

BASE_DIR = Path(__file__).resolve().parent.parent

assistant = ResearchAssistant()

# PDF paths
pdf_files = [
    BASE_DIR / "data/pdfs/JMLR.pdf"
]

# Ingest PDFs
assistant.ingest_pdfs(pdf_files)

# Setup QA
assistant.setup_qa()

# Ask question
question = "Summarize the main points of the document."
answer = assistant.ask(question)

print("Q:", question)
print("A:", answer)
