"""
RAG Engine - Zemra e aplikacionit
Mbështet: PDF, TXT, Markdown, DOCX
Vector DB: ChromaDB
LLM: Ollama (lokal)
"""

import os
import uuid
from pathlib import Path
from typing import Optional

# Document loaders
import chromadb
from chromadb.utils import embedding_functions

# Text splitter
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Document parsers
import fitz  # PyMuPDF për PDF
import docx  # python-docx për DOCX
import requests
import json


# ──────────────────────────────────────────────
# 1. DOCUMENT PARSERS
# ──────────────────────────────────────────────

def load_pdf(filepath: str) -> str:
    """Lexon PDF dhe kthen tekstin."""
    doc = fitz.open(filepath)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text


def load_docx(filepath: str) -> str:
    """Lexon DOCX dhe kthen tekstin."""
    doc = docx.Document(filepath)
    return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])


def load_txt_or_md(filepath: str) -> str:
    """Lexon TXT ose Markdown."""
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def load_document(filepath: str) -> str:
    """Ngarkon çdo lloj dokumenti bazuar në shtesën e skedarit."""
    ext = Path(filepath).suffix.lower()
    if ext == ".pdf":
        return load_pdf(filepath)
    elif ext == ".docx":
        return load_docx(filepath)
    elif ext in [".txt", ".md", ".markdown"]:
        return load_txt_or_md(filepath)
    else:
        raise ValueError(f"Lloji i skedarit '{ext}' nuk mbështetet.")


# ──────────────────────────────────────────────
# 2. TEXT CHUNKER
# ──────────────────────────────────────────────

def split_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[str]:
    """Ndaj tekstin në copa të vogla për indexim."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return splitter.split_text(text)


# ──────────────────────────────────────────────
# 3. VECTOR STORE (ChromaDB)
# ──────────────────────────────────────────────

class VectorStore:
    def __init__(self, persist_dir: str = "./chroma_db", collection_name: str = "rag_docs"):
        self.client = chromadb.PersistentClient(path=persist_dir)
        # Përdor embedding falas nga sentence-transformers
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn
        )

    def add_document(self, chunks: list[str], source_name: str) -> int:
        """Shton copa teksti në vector store."""
        if not chunks:
            return 0
        ids = [f"{source_name}_{i}_{uuid.uuid4().hex[:8]}" for i in range(len(chunks))]
        metadatas = [{"source": source_name, "chunk_index": i} for i in range(len(chunks))]
        self.collection.add(documents=chunks, ids=ids, metadatas=metadatas)
        return len(chunks)

    def search(self, query: str, n_results: int = 5) -> list[dict]:
        """Kërkon dokumentet më relevante."""
        results = self.collection.query(
            query_texts=[query],
            n_results=min(n_results, self.collection.count() or 1)
        )
        docs = []
        if results["documents"]:
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            ):
                docs.append({
                    "text": doc,
                    "source": meta.get("source", "unknown"),
                    "score": round(1 - dist, 3)
                })
        return docs

    def get_stats(self) -> dict:
        """Kthen statistika të collection-it."""
        count = self.collection.count()
        return {"total_chunks": count}

    def delete_source(self, source_name: str):
        """Fshin të gjitha chunks nga një burim."""
        results = self.collection.get(where={"source": source_name})
        if results["ids"]:
            self.collection.delete(ids=results["ids"])
            return len(results["ids"])
        return 0

    def list_sources(self) -> list[str]:
        """Liston të gjitha burimet e indeksuara."""
        results = self.collection.get()
        sources = set()
        for meta in results.get("metadatas", []):
            if meta:
                sources.add(meta.get("source", "unknown"))
        return sorted(list(sources))


# ──────────────────────────────────────────────
# 4. OLLAMA LLM
# ──────────────────────────────────────────────

class OllamaLLM:
    def __init__(self, model: str = "llama3", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url

    def list_models(self) -> list[str]:
        """Liston modelet e disponueshme në Ollama."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if resp.status_code == 200:
                return [m["name"] for m in resp.json().get("models", [])]
        except Exception:
            pass
        return []

    def is_available(self) -> bool:
        """Kontrollon nëse Ollama është duke punuar."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=3)
            return resp.status_code == 200
        except Exception:
            return False

    def generate(self, prompt: str, stream: bool = False) -> str:
        """Gjeneron përgjigje nga Ollama."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": stream
        }
        try:
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=120
            )
            if resp.status_code == 200:
                return resp.json().get("response", "")
            else:
                return f"Gabim nga Ollama: {resp.status_code}"
        except requests.exceptions.ConnectionError:
            return "❌ Ollama nuk është duke punuar. Nis: `ollama serve`"
        except Exception as e:
            return f"❌ Gabim: {str(e)}"


# ──────────────────────────────────────────────
# 5. RAG PIPELINE
# ──────────────────────────────────────────────

class RAGApp:
    def __init__(
        self,
        ollama_model: str = "llama3",
        persist_dir: str = "./chroma_db",
        ollama_url: str = "http://localhost:11434"
    ):
        self.vector_store = VectorStore(persist_dir=persist_dir)
        self.llm = OllamaLLM(model=ollama_model, base_url=ollama_url)

    def ingest(self, filepath: str) -> dict:
        """Ngarkon dhe indekson një dokument."""
        source_name = Path(filepath).name
        try:
            text = load_document(filepath)
            if not text.strip():
                return {"success": False, "error": "Dokumenti është bosh."}
            chunks = split_text(text)
            added = self.vector_store.add_document(chunks, source_name)
            return {
                "success": True,
                "source": source_name,
                "chunks": added,
                "chars": len(text)
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def query(self, question: str, n_results: int = 5) -> dict:
        """Bën query RAG: kërkon kontekst + pyet LLM."""
        # 1. Kërko kontekst relevant
        docs = self.vector_store.search(question, n_results=n_results)
        if not docs:
            return {
                "answer": "Nuk u gjet asnjë dokument i indeksuar. Ngarko dokumenta së pari.",
                "sources": [],
                "context_used": ""
            }

        # 2. Ndërto kontekstin
        context_parts = []
        for i, doc in enumerate(docs):
            context_parts.append(f"[{i+1}] (nga: {doc['source']})\n{doc['text']}")
        context = "\n\n".join(context_parts)

        # 3. Ndërto promptin
        prompt = f"""Bazuar në kontekstin e mëposhtëm, përgjigju pyetjes qartë dhe saktë.
Nëse informacioni nuk gjendet në kontekst, thuaj "Nuk e di bazuar në dokumentat e ngarkuara."

KONTEKST:
{context}

PYETJA: {question}

PËRGJIGJA:"""

        # 4. Merr përgjigjen nga LLM
        answer = self.llm.generate(prompt)

        return {
            "answer": answer,
            "sources": [{"source": d["source"], "score": d["score"]} for d in docs],
            "context_used": context
        }

    def get_stats(self) -> dict:
        stats = self.vector_store.get_stats()
        stats["ollama_available"] = self.llm.is_available()
        stats["ollama_model"] = self.llm.model
        stats["sources"] = self.vector_store.list_sources()
        return stats

    def delete_source(self, source_name: str) -> int:
        return self.vector_store.delete_source(source_name)
