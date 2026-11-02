"""
Shared configuration for the Document Q&A (RAG) Agent.
Edit these values to tune retrieval quality, chunking, and storage locations.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

# --- Paths ------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
FAISS_DIR = BASE_DIR / "vectorstore_faiss"
CHROMA_DIR = BASE_DIR / "vectorstore_chroma"
CHROMA_COLLECTION_NAME = "rag_qa_knowledge_base"

# --- Chunking -----------------------------------------------------------
CHUNK_SIZE = 500        # characters per chunk
CHUNK_OVERLAP = 80      # characters shared between consecutive chunks
# Smaller chunks -> more precise retrieval but less surrounding context per chunk.
# Larger chunks -> more context per chunk but noisier retrieval. Overlap keeps
# sentences that straddle a chunk boundary from being split away from their meaning.

# --- Embeddings -----------------------------------------------------------
# Local, free sentence-transformer model -> no API cost, runs on CPU,
# downloaded and cached automatically on first use.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# --- Retrieval ------------------------------------------------------------
DEFAULT_K = 3            # number of chunks retrieved per query (override with --k)

# --- LLM (Groq) ----------------------------------------------------------
# Groq deprecated llama-3.3-70b-versatile / llama-3.1-8b-instant in mid-2026.
# openai/gpt-oss-120b is the current recommended general-purpose replacement.
# Full, current model list: https://console.groq.com/docs/models
# Override without touching code: set GROQ_MODEL_NAME in your .env file.
GROQ_MODEL_NAME = os.getenv("GROQ_MODEL_NAME", "openai/gpt-oss-120b")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_TEMPERATURE = 0  # low temperature -> deterministic, non-creative answers

# --- Prompt (grounding rule lives here — do not remove the "I don't know" line) ---
QA_PROMPT_TEMPLATE = """You are a helpful assistant answering questions using ONLY the context below. If the answer is not contained in the context, reply: "I don't know." Do not invent or guess. Provide a short answer and then include which document chunk(s) you used by quoting the first ~120 characters of each chunk.

Context:
{context}

Question:
{question}

Answer:"""
