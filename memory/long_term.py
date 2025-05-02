# long_term.py

import pathlib          # ← add this line
from typing import List
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Set up embedding model
embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")


BASE_DIR = pathlib.Path(__file__).parent                # folder where long_term.py lives
vectorstore_path = str(BASE_DIR / "faiss_memory_index") # absolute path
vectorstore = None

def _build_empty_index() -> FAISS:
    """Return a FAISS store with a single throw-away vector."""
    placeholder = Document(page_content="__PLACEHOLDER__",
                           metadata={"npc_id": "__system__", "sender": "system"})
    vs = FAISS.from_documents([placeholder], embedding)
    vs.save_local(vectorstore_path)
    return vs


def init_vectorstore() -> None:
    """Load the on-disk index, or create a minimal placeholder index."""
    global vectorstore
    if os.path.exists(vectorstore_path):
        try:
            vectorstore = FAISS.load_local(
                vectorstore_path,
                embedding,
                allow_dangerous_deserialization=True,
            )
            return
        except Exception as exc:
            print(f"[long_term] failed to load index ({exc}); rebuilding...")

    # Folder missing or load failed → build a minimal index
    vectorstore = _build_empty_index()

def add_to_long_term(npc_id: str, messages: List[str]):
    global vectorstore
    docs = [Document(page_content=msg, metadata={"npc_id": npc_id}) for msg in messages]

    if vectorstore is None:
        vectorstore = FAISS.from_documents(docs, embedding)
    else:
        vectorstore.add_documents(docs)

    vectorstore.save_local(vectorstore_path)

def search_long_term(npc_id: str, query: str, k: int = 3) -> List[str]:
    if vectorstore is None:
        return []
    results = vectorstore.similarity_search(query, k=k)
    return [doc.page_content for doc in results if doc.metadata.get("npc_id") == npc_id]
