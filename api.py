"""
api.py
FastAPI REST API for the Document Q&A (RAG) Agent.

Endpoints:
    GET  /api/health          – Health check & readiness probe
    POST /api/ask             – Ask a question (RAG pipeline)
    GET  /api/config          – Current agent configuration
    GET  /api/documents       – List indexed documents
    POST /api/ask-guardrailed – Ask with input/output guardrails

Run:
    uvicorn api:app --reload --port 8000

The React frontend is served from frontend/ at the root URL.
"""

import os
import sys
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

import config

# ---------------------------------------------------------------------------
# Pydantic models for request / response
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    """Request body for /api/ask."""
    question: str = Field(..., min_length=1, max_length=2000, description="The question to ask")
    store: str = Field(default="faiss", description="Vector store: 'faiss' or 'chroma'")
    k: int = Field(default=config.DEFAULT_K, ge=1, le=20, description="Number of chunks to retrieve")


class Citation(BaseModel):
    """A single source citation."""
    tag: str
    snippet: str


class AskResponse(BaseModel):
    """Response body for /api/ask."""
    answer: str
    citations: list[Citation]
    query_time_ms: float
    store_used: str
    chunks_retrieved: int


class HealthResponse(BaseModel):
    """Response body for /api/health."""
    status: str
    faiss_ready: bool
    chroma_ready: bool
    groq_configured: bool
    model: str


class ConfigResponse(BaseModel):
    """Response body for /api/config."""
    embedding_model: str
    llm_model: str
    chunk_size: int
    chunk_overlap: int
    default_k: int
    groq_configured: bool


class DocumentInfo(BaseModel):
    """Metadata about an indexed document."""
    name: str
    size_bytes: int
    extension: str


class DocumentsResponse(BaseModel):
    """Response body for /api/documents."""
    documents: list[DocumentInfo]
    total: int


# ---------------------------------------------------------------------------
# Agent initialisation (lazy-loaded singletons)
# ---------------------------------------------------------------------------

_components: dict = {}


def _get_components(store: str = "faiss", k: int = 3):
    """Load and cache embeddings, vector store, retriever, and LLM."""
    cache_key = f"{store}_{k}"
    if cache_key not in _components:
        try:
            from pydantic import SecretStr
            from langchain_huggingface import HuggingFaceEmbeddings
            from langchain_community.vectorstores import FAISS, Chroma
            from langchain_groq import ChatGroq

            embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_NAME)

            if store == "faiss":
                if not config.FAISS_DIR.exists():
                    raise FileNotFoundError(f"No FAISS index at {config.FAISS_DIR}. Run build_index.py first.")
                db = FAISS.load_local(
                    str(config.FAISS_DIR), embeddings, allow_dangerous_deserialization=True
                )
            elif store == "chroma":
                if not config.CHROMA_DIR.exists():
                    raise FileNotFoundError(f"No Chroma index at {config.CHROMA_DIR}. Run build_index.py first.")
                db = Chroma(
                    persist_directory=str(config.CHROMA_DIR),
                    embedding_function=embeddings,
                    collection_name=config.CHROMA_COLLECTION_NAME,
                )
            else:
                raise ValueError(f"Unknown store '{store}'")

            retriever = db.as_retriever(search_kwargs={"k": k})

            llm = ChatGroq(
                model=config.GROQ_MODEL_NAME,
                temperature=config.LLM_TEMPERATURE,
                api_key=SecretStr(config.GROQ_API_KEY),
            )
            _components[cache_key] = (retriever, llm)
        except Exception as e:
            raise HTTPException(status_code=503, detail=str(e))

    return _components[cache_key]


def _format_context(chunks):
    """Build context block and citations from retrieved chunks."""
    context_parts = []
    citations = []
    for i, doc in enumerate(chunks, start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page")
        tag = f"[{i}] {os.path.basename(source)}"
        if page is not None:
            tag += f" (page {page + 1})"
        context_parts.append(f"{tag}\n{doc.page_content}")
        snippet = doc.page_content[:150].replace("\n", " ").strip()
        citations.append(Citation(tag=tag, snippet=snippet))
    return "\n\n".join(context_parts), citations


# ---------------------------------------------------------------------------
# Guardrails (imported from existing practical5 code)
# ---------------------------------------------------------------------------

def _run_guardrails(question: str, raw_answer: str) -> tuple[str, bool]:
    """Run input + tool + output guardrails. Returns (answer, was_blocked)."""
    try:
        from practical5_with_guardrails import (
            run_input_guardrails,
            check_tool_action,
            run_output_guardrails,
        )

        input_block = run_input_guardrails(question)
        if input_block:
            return input_block, True

        tool_block = check_tool_action(question, raw_answer)
        if tool_block:
            return f"BLOCKED by tool/action guardrail: {tool_block}.", True

        guarded = run_output_guardrails(raw_answer)
        if guarded.startswith("BLOCKED"):
            return guarded, True

        return guarded, False
    except ImportError:
        return raw_answer, False


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="RAG QA Agent API",
    description="REST API for the Document Q&A (RAG) Agent — Practical 9",
    version="1.0.0",
)

# CORS for local React dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# REST API endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check system readiness."""
    return HealthResponse(
        status="ok",
        faiss_ready=config.FAISS_DIR.exists(),
        chroma_ready=config.CHROMA_DIR.exists(),
        groq_configured=bool(config.GROQ_API_KEY),
        model=config.GROQ_MODEL_NAME,
    )


@app.get("/api/config", response_model=ConfigResponse, tags=["System"])
async def get_config():
    """Return current agent configuration."""
    return ConfigResponse(
        embedding_model=config.EMBEDDING_MODEL_NAME,
        llm_model=config.GROQ_MODEL_NAME,
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        default_k=config.DEFAULT_K,
        groq_configured=bool(config.GROQ_API_KEY),
    )


@app.get("/api/documents", response_model=DocumentsResponse, tags=["Documents"])
async def list_documents():
    """List all documents in the data directory."""
    docs = []
    if config.DATA_DIR.exists():
        for path in sorted(config.DATA_DIR.glob("*")):
            if path.is_file():
                docs.append(DocumentInfo(
                    name=path.name,
                    size_bytes=path.stat().st_size,
                    extension=path.suffix,
                ))
    return DocumentsResponse(documents=docs, total=len(docs))


@app.post("/api/ask", response_model=AskResponse, tags=["Q&A"])
async def ask_question(req: AskRequest):
    """Ask a question using the RAG pipeline."""
    if req.store not in ("faiss", "chroma"):
        raise HTTPException(status_code=400, detail="store must be 'faiss' or 'chroma'")

    retriever, llm = _get_components(req.store, req.k)

    start = time.perf_counter()
    chunks = retriever.invoke(req.question)
    if not chunks:
        return AskResponse(
            answer="I don't know.",
            citations=[],
            query_time_ms=round((time.perf_counter() - start) * 1000, 2),
            store_used=req.store,
            chunks_retrieved=0,
        )

    context, citations = _format_context(chunks)
    prompt = config.QA_PROMPT_TEMPLATE.format(context=context, question=req.question)
    response = llm.invoke(prompt)
    answer = response.content if hasattr(response, "content") else str(response)
    elapsed = round((time.perf_counter() - start) * 1000, 2)

    return AskResponse(
        answer=answer,
        citations=citations,
        query_time_ms=elapsed,
        store_used=req.store,
        chunks_retrieved=len(chunks),
    )


@app.post("/api/ask-guardrailed", response_model=AskResponse, tags=["Q&A"])
async def ask_question_guardrailed(req: AskRequest):
    """Ask a question with input/output guardrails applied."""
    if req.store not in ("faiss", "chroma"):
        raise HTTPException(status_code=400, detail="store must be 'faiss' or 'chroma'")

    retriever, llm = _get_components(req.store, req.k)

    start = time.perf_counter()
    chunks = retriever.invoke(req.question)

    if not chunks:
        raw_answer = "I don't know."
        citations = []
    else:
        context, citations = _format_context(chunks)
        prompt = config.QA_PROMPT_TEMPLATE.format(context=context, question=req.question)
        response = llm.invoke(prompt)
        raw_answer = response.content if hasattr(response, "content") else str(response)

    guarded_answer, was_blocked = _run_guardrails(req.question, raw_answer)
    elapsed = round((time.perf_counter() - start) * 1000, 2)

    return AskResponse(
        answer=guarded_answer,
        citations=[] if was_blocked else citations,
        query_time_ms=elapsed,
        store_used=req.store,
        chunks_retrieved=0 if was_blocked else len(chunks),
    )


# ---------------------------------------------------------------------------
# Serve React frontend
# ---------------------------------------------------------------------------

FRONTEND_DIR = Path(__file__).resolve().parent / "frontend"

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/", tags=["Frontend"])
    async def serve_frontend():
        """Serve the React frontend."""
        return FileResponse(str(FRONTEND_DIR / "index.html"))
