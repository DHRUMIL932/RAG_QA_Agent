"""
build_index.py
Loads every .txt / .pdf file in data/, splits them into overlapping chunks,
embeds them with a local sentence-transformer model, and builds both a
FAISS index and a Chroma index from the same embeddings.

Usage:
    python build_index.py
"""

import sys

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS, Chroma

import config


def load_documents():
    """Load every .txt and .pdf file in DATA_DIR into LangChain Document objects."""
    files = sorted(p for p in config.DATA_DIR.glob("*") if p.is_file())

    if not files:
        print(f"No files found in {config.DATA_DIR}/. Add .txt or .pdf files and re-run.")
        sys.exit(1)

    docs = []
    for path in files:
        suffix = path.suffix.lower()
        if suffix == ".txt":
            docs.extend(TextLoader(str(path), encoding="utf-8").load())
        elif suffix == ".pdf":
            docs.extend(PyPDFLoader(str(path)).load())
        # unsupported extensions are skipped silently

    if not docs:
        print(f"No .txt or .pdf files found in {config.DATA_DIR}/ (only these extensions are supported).")
        sys.exit(1)

    return docs


def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    return splitter.split_documents(docs)


def main():
    print(f"Loading documents from {config.DATA_DIR}/ ...")
    docs = load_documents()
    print(f"Loaded {len(docs)} docs from data/")

    chunks = split_documents(docs)
    print(f"Split into {len(chunks)} chunks")

    print(
        f"Loading local embedding model '{config.EMBEDDING_MODEL_NAME}' "
        f"(first run downloads and caches it, may take a minute) ..."
    )
    embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_NAME)

    # --- FAISS: fast, simple, good for local read-heavy setups ---
    print("Building FAISS index ...")
    faiss_db = FAISS.from_documents(chunks, embeddings)
    faiss_db.save_local(str(config.FAISS_DIR))
    print(f"FAISS index saved to '{config.FAISS_DIR.name}/'")

    # --- Chroma: persistent, supports metadata filters and easier updates ---
    print("Building Chroma index ...")
    Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=str(config.CHROMA_DIR),
        collection_name=config.CHROMA_COLLECTION_NAME,
    )
    print(f"Chroma index saved to '{config.CHROMA_DIR.name}/'")

    print('\nIndex build complete. Try: python app.py --query "What is the refund policy?"')


if __name__ == "__main__":
    main()
