"""
app.py
Document Q&A (RAG) agent CLI.

Interactive mode:
    python app.py

One-shot mode:
    python app.py --query "What is the refund policy?"

Flags:
    --store   faiss | chroma   (default: faiss)
    --k       number of chunks to retrieve (default: 3)
    --query   run a single question and exit instead of entering interactive mode
"""

import argparse
import os
import sys

from dotenv import load_dotenv
from pydantic import SecretStr

load_dotenv()

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS, Chroma
from langchain_groq import ChatGroq

import config


def load_vectorstore(store: str, embeddings):
    if store == "faiss":
        if not config.FAISS_DIR.exists():
            sys.exit(f"No FAISS index found at {config.FAISS_DIR}/. Run build_index.py first.")
        # allow_dangerous_deserialization is safe here because this index was
        # built locally by build_index.py in this same project. Never set this
        # to True for a FAISS index you did not create yourself / don't trust.
        return FAISS.load_local(
            str(config.FAISS_DIR), embeddings, allow_dangerous_deserialization=True
        )
    elif store == "chroma":
        if not config.CHROMA_DIR.exists():
            sys.exit(f"No Chroma index found at {config.CHROMA_DIR}/. Run build_index.py first.")
        return Chroma(
            persist_directory=str(config.CHROMA_DIR),
            embedding_function=embeddings,
            collection_name=config.CHROMA_COLLECTION_NAME,
        )
    sys.exit(f"Unknown store '{store}'. Choose 'faiss' or 'chroma'.")


def format_context(chunks):
    """Build the {context} block fed to the LLM and a parallel, code-verified
    citation list (document name + page if available + snippet)."""
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
        citations.append(f'{tag}: "{snippet}..."')
    return "\n\n".join(context_parts), citations


def answer_question(question: str, retriever, llm):
    chunks = retriever.invoke(question)
    if not chunks:
        return "I don't know.", []

    context, citations = format_context(chunks)
    prompt = config.QA_PROMPT_TEMPLATE.format(context=context, question=question)
    response = llm.invoke(prompt)
    answer = response.content if hasattr(response, "content") else str(response)
    return answer, citations


def main():
    parser = argparse.ArgumentParser(description="Document Q&A (RAG) agent")
    parser.add_argument("--store", choices=["faiss", "chroma"], default="faiss")
    parser.add_argument("--k", type=int, default=config.DEFAULT_K)
    parser.add_argument("--query", type=str, default=None)
    args = parser.parse_args()

    if not config.GROQ_API_KEY:
        sys.exit("GROQ_API_KEY not set. Add it to a .env file in this folder (see .env.example).")

    embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_NAME)
    db = load_vectorstore(args.store, embeddings)
    retriever = db.as_retriever(search_kwargs={"k": args.k})

    llm = ChatGroq(
        model=config.GROQ_MODEL_NAME,
        temperature=config.LLM_TEMPERATURE,
        api_key=SecretStr(config.GROQ_API_KEY),
    )

    def run(question):
        answer, citations = answer_question(question, retriever, llm)
        print(f"\nAnswer: {answer}\n")
        if citations:
            print("Used chunks:")
            for c in citations:
                print(f"  - {c}")
        print()

    if args.query:
        run(args.query)
        return

    print(f"Document Q&A agent ready (store={args.store}, k={args.k}). Type 'exit' to quit.\n")
    while True:
        try:
            question = input("Ask a question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue
        run(question)


if __name__ == "__main__":
    main()
