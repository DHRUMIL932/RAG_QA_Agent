# Document Q&A (RAG) Agent

A local, low-cost Document Question-Answering agent. It indexes your `.txt`/`.pdf`
files, retrieves the most relevant chunks for a question, and asks an LLM
(via [Groq](https://console.groq.com)) to answer using **only** that retrieved
context — replying "I don't know." when the answer isn't in your documents.

## How it works

```
data/*.txt,*.pdf → chunk (500 chars, 80 overlap) → embed (local, free)
                 → FAISS / Chroma index → retriever (top-k)
                 → prompt (context + question) → Groq LLM → grounded answer + citations
```

## 1. Prerequisites

- Python 3.11–3.13
- A free Groq API key: https://console.groq.com/keys

## 2. Install

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 3. Configure

```bash
cp .env.example .env
# then edit .env and paste your key:
# GROQ_API_KEY=gsk_...
```

## 4. Add your documents

Drop `.txt` or `.pdf` files into `data/`. A sample file, `nimbus_handbook.txt`
(a fictional company handbook), is already there so you can try the agent
immediately.

## 5. Build the index

```bash
python build_index.py
```

This loads every file in `data/`, splits it into overlapping chunks, embeds
the chunks locally (free — no API calls), and saves **both** a FAISS index
(`vectorstore_faiss/`) and a Chroma index (`vectorstore_chroma/`) so you can
compare the two.

Expected output:
```
Loaded 1 docs from data/
Split into 12 chunks
FAISS index saved to 'vectorstore_faiss/'
Chroma index saved to 'vectorstore_chroma/'
```

## 6. Ask questions

**One-shot:**
```bash
python app.py --query "What is the refund policy?"
```
```
Answer: Nimbus offers a 14-day money-back guarantee on all paid plans...

Used chunks:
  - [1] nimbus_handbook.txt: "Refund Policy Nimbus offers a 14-day money-back guarantee on all paid plans..."
```

**Interactive:**
```bash
python app.py
Ask a question: What is the CEO's favorite coffee?

Answer: I don't know.
```

**Flags:**

| Flag      | Default | Meaning                                   |
|-----------|---------|--------------------------------------------|
| `--store` | `faiss` | Vector store to query: `faiss` or `chroma` |
| `--k`     | `3`     | Number of chunks to retrieve per question  |
| `--query` | —       | Run one question and exit (skips interactive mode) |

```bash
python app.py --store chroma --k 5 --query "What products does Nimbus offer?"
```

## 7. Deploy with FastAPI and React

The project includes a FastAPI REST API and a React single-page interface. The
API lazily loads the embedding model and vector store on the first question,
then caches them for subsequent requests.

Start the deployment from this folder:

```bash
uvicorn api:app --host 127.0.0.1 --port 8000
```

Open the web interface at http://127.0.0.1:8000. The API is also available
through Swagger UI at http://127.0.0.1:8000/docs.

Example REST requests:

```bash
# Readiness check
curl http://127.0.0.1:8000/api/health

# Ask the RAG agent
curl -X POST http://127.0.0.1:8000/api/ask-guardrailed ^
  -H "Content-Type: application/json" ^
  -d "{\"question\":\"What is the refund policy?\"}"
```

The API also exposes `/api/config` and `/api/documents`. Use
`/api/ask-guardrailed` for the guarded Practical 9 flow, or `/api/ask` for
the unguarded comparison.

## 8. Guardrail comparison

The Practical 5 guardrail comparison can be run with the same five prompts in both
versions:

```bash
python practical5_without_guardrails.py > output_without_guardrails.txt
python practical5_with_guardrails.py > output_with_guardrails.txt
```

Both scripts print each test name, input, and response. The shared prompts are
in `test_prompts.py`.

| Test | Without guardrails | With guardrails |
|---|---|---|
| Normal Nimbus question | Sends the request to retrieval and the LLM | Sends the request through the same RAG flow |
| Prompt injection | Sends the injection to the LLM | Blocks it with the prompt-injection input check |
| Destructive/security request | Sends it to the LLM | Blocks it with the read-only tool/action policy |
| Off-topic question | Sends it to the LLM | Blocks it with the Nimbus scope check |
| Abusive message | Sends it to the LLM | Blocks it with the profanity input check |

The guarded version applies input checks before the LLM, then the read-only
tool/action check, then output checks for PII, prompt/secret leakage, unsafe
content, and response format. If the Groq key or LangChain packages are not
available, both scripts use a clearly labeled local mock LLM fallback so the
comparison remains runnable; with the configured dependencies and key, they use
the existing Groq model and FAISS setup.

## Configuration knobs (`config.py`)

- `CHUNK_SIZE` / `CHUNK_OVERLAP` — smaller chunks retrieve more precisely but
  hold less surrounding context; overlap prevents sentences from being cut
  in half at a chunk boundary. Defaults: 500 / 80 characters.
- `EMBEDDING_MODEL_NAME` — local `sentence-transformers` model, free to run.
- `GROQ_MODEL_NAME` — defaults to `openai/gpt-oss-120b`. Groq retired
  `llama-3.3-70b-versatile` in mid-2026; check
  https://console.groq.com/docs/models for the current list, and override
  via `GROQ_MODEL_NAME` in `.env` without touching code.
- `LLM_TEMPERATURE` — kept at `0` for deterministic, non-creative answers.

## Swapping to a paid LLM (OpenAI / Anthropic) instead of Groq

Replace the `ChatGroq(...)` call in `app.py` with the equivalent client
(`langchain_openai.ChatOpenAI` or `langchain_anthropic.ChatAnthropic`), and
set the matching env var (`OPENAI_API_KEY` / `ANTHROPIC_API_KEY`) in `.env`.
Everything else — chunking, retrieval, prompt, citation logic — stays the same.

## Troubleshooting

- **"No files found"** — check that files are directly inside `data/` and
  end in `.txt` or `.pdf`.
- **FAISS load / deserialization error** — `app.py` passes
  `allow_dangerous_deserialization=True` when loading FAISS. This is safe
  only because the index was built locally by `build_index.py`; never load
  a FAISS index from an untrusted source with this flag.
- **First run is slow** — the embedding model (~90 MB) downloads once and is
  cached locally afterward.
- **`GROQ_API_KEY not set`** — make sure you copied `.env.example` to `.env`
  and pasted a real key, and that you're running from the project folder.
- **No index found** — run `python build_index.py` before `python app.py`.

## Notes on safety & privacy

Vector indexes store numerical embeddings derived from your documents, not
the raw files themselves — but embeddings can still leak information about
the source text, so treat `vectorstore_faiss/` and `vectorstore_chroma/` with
the same confidentiality as the original documents. For an enterprise
deployment, consider where these index files are stored/backed up relative
to your data-residency requirements.
