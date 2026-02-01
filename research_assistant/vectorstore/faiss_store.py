from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

class FAISSStore:
    def __init__(self, embedding_model="text-embedding-3-small", index_path=None):
        """
        Initialize FAISS vector store for storing document embeddings.
        """
        # Embeddings object
        self.embeddings = OpenAIEmbeddings(model=embedding_model)

        # FAISS index placeholder
        self.index_path = index_path
        self.vectorstore = None

    def build_index(self, texts, metadatas=None):
        """
        Build FAISS index from a list of texts.
        """
        self.vectorstore = FAISS.from_texts(texts, self.embeddings, metadatas=metadatas)
        if self.index_path:
            self.vectorstore.save_local(self.index_path)

    def load_index(self):
        """
        Load FAISS index from disk.
        """
        if self.index_path:
            self.vectorstore = FAISS.load_local(self.index_path, self.embeddings)
        else:
            raise ValueError("No index_path provided to load FAISS index.")

    def as_retriever(self, search_kwargs=None):
        """
        Return a retriever object for RetrievalQA.
        """
        if self.vectorstore is None:
            raise ValueError("FAISS index not built or loaded.")
        return self.vectorstore.as_retriever(search_kwargs=search_kwargs or {"k": 3})
