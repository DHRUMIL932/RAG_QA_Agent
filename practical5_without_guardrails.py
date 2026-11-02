"""Practical 5 RAG agent demonstration without guardrails.

This keeps the original retrieval -> prompt -> LLM -> response flow.  The
batch runner is added so the same five demonstration prompts can be compared
with the guarded version.
"""

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

import config
from test_prompts import TEST_PROMPTS

load_dotenv(Path(__file__).resolve().parent / ".env")


class MockResponse:
    def __init__(self, content: str):
        self.content = content


class MockLLM:
    """Clearly marked fallback used only when the configured stack is unavailable."""

    def invoke(self, prompt: str) -> MockResponse:
        question = prompt.rsplit("Question:", 1)[-1].split("Answer:", 1)[0].strip()
        return MockResponse(
            "MOCK LLM FALLBACK: The retrieved Nimbus handbook context was provided. "
            f"Response to the user question: {question}"
        )


class SimpleDocument:
    def __init__(self, page_content: str, source: str):
        self.page_content = page_content
        self.metadata = {"source": source}


class SimpleRetriever:
    def __init__(self, documents: list[SimpleDocument], k: int):
        self.documents = documents
        self.k = k

    def invoke(self, question: str) -> list[SimpleDocument]:
        terms = set(question.lower().split())
        ranked = sorted(
            self.documents,
            key=lambda doc: sum(term in doc.page_content.lower() for term in terms),
            reverse=True,
        )
        return ranked[: self.k]


def load_fallback_retriever(k: int) -> SimpleRetriever:
    """Load handbook sections when optional LangChain packages are unavailable."""
    handbook = (config.DATA_DIR / "nimbus_handbook.txt").read_text(encoding="utf-8")
    sections = [section.strip() for section in handbook.split("\n\n") if section.strip()]
    documents = [
        SimpleDocument(section, str(config.DATA_DIR / "nimbus_handbook.txt"))
        for section in sections
    ]
    return SimpleRetriever(documents, k)


def load_agent_components(k: int) -> tuple[Any, Any]:
    """Reuse the project's configured embeddings, vector store, and Groq model."""
    if not config.GROQ_API_KEY:
        return load_fallback_retriever(k), MockLLM()

    try:
        from pydantic import SecretStr
        from langchain_community.vectorstores import FAISS
        from langchain_groq import ChatGroq
        from langchain_huggingface import HuggingFaceEmbeddings

        embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_NAME)
        db = FAISS.load_local(
            str(config.FAISS_DIR),
            embeddings,
            allow_dangerous_deserialization=True,
        )
        retriever = db.as_retriever(search_kwargs={"k": k})
        llm = ChatGroq(
            model=config.GROQ_MODEL_NAME,
            temperature=config.LLM_TEMPERATURE,
            api_key=SecretStr(config.GROQ_API_KEY),
        )
        return retriever, llm
    except (ImportError, OSError, ValueError) as error:
        print(f"WARNING: Using MOCK LLM FALLBACK because the configured stack is unavailable: {error}")
        return load_fallback_retriever(k), MockLLM()


def format_context(chunks: list[Any]) -> tuple[str, list[str]]:
    context_parts = []
    citations = []
    for i, doc in enumerate(chunks, start=1):
        source = doc.metadata.get("source", "unknown")
        tag = f"[{i}] {os.path.basename(source)}"
        context_parts.append(f"{tag}\n{doc.page_content}")
        snippet = doc.page_content[:150].replace("\n", " ").strip()
        citations.append(f'{tag}: "{snippet}..."')
    return "\n\n".join(context_parts), citations


def answer_question(question: str, retriever: Any, llm: Any) -> str:
    chunks = retriever.invoke(question)
    if not chunks:
        return "I don't know."
    context, citations = format_context(chunks)
    prompt = config.QA_PROMPT_TEMPLATE.format(context=context, question=question)
    response = llm.invoke(prompt)
    answer = response.content if hasattr(response, "content") else str(response)
    citations_text = "\nUsed chunks:\n" + "\n".join(f"  - {citation}" for citation in citations)
    return f"{answer}{citations_text}"


def main() -> None:
    retriever, llm = load_agent_components(config.DEFAULT_K)
    print("PRACTICAL 5 - WITHOUT GUARDRAILS")
    for test_name, user_input in TEST_PROMPTS:
        response = answer_question(user_input, retriever, llm)
        print(f"\nTest: {test_name}\nUser input: {user_input}\nAgent response: {response}\n")


if __name__ == "__main__":
    main()
