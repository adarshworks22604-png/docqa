# DocQA RAG — backend

Document Q&A over your own PDFs with **hybrid search**, **source citations** and a built-in
**retrieval evaluation harness**. FastAPI + SQLite, no external services needed.

```
upload PDF ─► parse (keep page numbers) ─► chunk (keep char offsets) ─► embed ─► SQLite
question  ─► dense search ┐
                          ├─► Reciprocal Rank Fusion ─► (optional cross-encoder rerank) ─► LLM ─► answer with [1][2] citations
             BM25 search  ┘
```

## 1. Run it (VS Code)

Needs **Python 3.10+**. Open the `backend/` folder in VS Code, then in the integrated terminal:

```bash
# create + activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # macOS / Linux
.venv\Scripts\Activate.ps1         # Windows PowerShell

pip install -r requirements.txt    # first install is big: sentence-transformers pulls in PyTorch
cp .env.example .env               # Windows: copy .env.example .env

uvicorn app.main:app --reload
```

Open **http://localhost:8000/docs** for the interactive API (upload a file, ask a question).
In VS Code, pick the `.venv` interpreter (Ctrl/Cmd+Shift+P → _Python: Select Interpreter_); the
**Run and Debug** panel has an _API: uvicorn (reload)_ launch config, and `requests.http` lets you
fire requests with the REST Client extension.

The first start downloads the embedding model (~90 MB) — needs internet once.

**Want to try it in 2 minutes without PyTorch?** Remove the `sentence-transformers` line from
`requirements.txt` before installing and set `EMBEDDING_PROVIDER=hash` in `.env`. Search then works
on word overlap only (no synonyms) — fine for a smoke test, not for real quality.

### Try it with the sample document

```bash
python -m scripts.ingest_folder data/samples      # indexes data/samples/rag_overview.md
python -m evaluation.run_eval --k 1 3 5           # compares dense / sparse / hybrid
```

## 2. Answers: with or without an LLM

| `LLM_PROVIDER`   | What `/chat` returns                                                                                                                           |
| ---------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| `none` (default) | No key needed. Quotes the best-matching sentences, each cited `[n]`. A basic fallback — it does not write fluent answers.                      |
| `anthropic`      | Generated answer from Claude. Set `LLM_API_KEY` and `LLM_MODEL`.                                                                               |
| `openai`         | Any OpenAI-compatible API (OpenAI, Groq, Together, local Ollama…). Set `LLM_BASE_URL`, `LLM_MODEL`, and `LLM_API_KEY` (not needed for Ollama). |

See `.env.example` for ready-to-copy examples.

## 3. API

All routes are under `/api`.

| Method & path                                                       | What it does                                                                                     |
| ------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| `POST /documents`                                                   | Upload `.pdf .docx .txt .md` (multipart `file`). Re-uploading identical bytes is detected.       |
| `GET /documents` · `GET /documents/{id}` · `DELETE /documents/{id}` | List / inspect / delete (also removes chunks + file).                                            |
| `GET /documents/{id}/file`                                          | The original file. Open a PDF at a cited page with `…/file#page=7`.                              |
| `GET /documents/{id}/chunks?page=3` · `GET /chunks/{chunk_id}`      | Raw chunks, for building a source viewer.                                                        |
| `POST /chat`                                                        | `{question, document_ids?, strategy?, top_k?, rerank?, session_id?}` → answer + citations.       |
| `GET /chat/history/{session_id}`                                    | Stored conversation. Follow-ups are rewritten into standalone queries when an LLM is configured. |
| `POST /evaluation/run`                                              | Run the golden-set evaluation (see §5).                                                          |
| `GET /evaluation/reports` · `GET /evaluation/reports/{name}`        | Saved reports.                                                                                   |
| `GET /health`                                                       | Counts + active configuration.                                                                   |

**How a frontend shows citations.** The answer text contains `[1]`, `[2]`… `sources[n-1]` gives
the `filename`, `page`, full chunk `text`, `snippet` and the retrieval `scores` (dense/sparse ranks,
RRF score, rerank score) for marker _n_. `citations` is the subset the answer actually cites;
`sources[i].cited` tells you which ones. `warnings` explains fallbacks (e.g. no LLM, answer without
citations, reranker unavailable).

## 4. Project layout

```
app/
  main.py                    FastAPI app, CORS, error handling, startup
  core/                      config (.env), errors, logging, dependency singletons
  api/routes/                documents, chat, evaluation, health   (thin: validate -> call a service)
  schemas/                   Pydantic request/response models
  services/
    ingestion/               loader (PDF/DOCX/TXT) · chunker (page + offset aware) · pipeline
    retrieval/               embeddings · dense · bm25 + sparse · hybrid (RRF) · reranker · retriever
    generation/              prompts · llm clients (httpx) · citations
    rag_service.py           retrieve -> generate -> cite
  db/                        SQLite schema, repository (all SQL), in-memory vector store


evaluation/
  datasets/golden_qa.jsonl   questions + evidence phrases
  metrics.py                 hit@k, recall@k, MRR, nDCG@k
  run_eval.py                compares dense / sparse / hybrid (/ hybrid+rerank) -> reports/
  answer_quality.py          cited-evidence rate + optional LLM-as-judge


scripts/                     ingest_folder · reindex · generate_eval_set


tests/                       pytest suite (offline, no model downloads)


data/                        SQLite DB + uploaded files (created at runtime) · samples/
```

`evaluation/` calls the same `Retriever` the API uses, so what you measure is what users get.

## 5. Evaluation

Retrieval is evaluated **separately from generation**, so you can tell whether a bad answer came
from missing context or from the LLM.

1. Put questions in `evaluation/datasets/golden_qa.jsonl`, one JSON object per line:
   ```json
   {
     "id": "q1",
     "question": "What is the usual value of k in RRF?",
     "document": "rag_overview.md",
     "evidence": ["commonly set to 60"],
     "reference_answer": "60"
   }
   ```
   A retrieved chunk is **relevant** if it contains an `evidence` phrase (case/punctuation
   insensitive) and — if given — comes from `document`. Evidence phrases that match nothing in the
   index are reported under `skipped` instead of silently counting as failures.
2. Run it:
   ```bash
   python -m evaluation.run_eval --k 1 3 5              # dense vs sparse vs hybrid
   python -m evaluation.run_eval --rerank --answers     # + reranker, + answer checks
   ```
   Reports land in `evaluation/reports/` as `.json` and `.md`. The same thing is available at
   `POST /api/evaluation/run`.
3. To draft questions from your own PDFs (needs an LLM key):
   `python -m scripts.generate_eval_set --n 25`. **Review the output** — generated questions copy the
   chunk's wording, which makes them easier than real user questions.

**Read the numbers honestly.** `data/samples/rag_overview.md` is only ~5 chunks, so every strategy
scores near 1.0 at k=3: it is a smoke test, not a benchmark. To see real differences between dense,
sparse and hybrid you need your own documents (dozens of pages) and 30+ questions that include
paraphrased wording and exact identifiers (names, codes), where the strategies actually disagree.

## 6. Configuration

Everything is in `.env` (see `.env.example`). The ones you will touch most:

| Variable                                 | Default                                      | Notes                                                                                                    |
| ---------------------------------------- | -------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| `EMBEDDING_PROVIDER` / `EMBEDDING_MODEL` | `sentence-transformers` / `all-MiniLM-L6-v2` | After changing, run `python -m scripts.reindex` (the API refuses to mix models, with a clear 409 error). |
| `CHUNK_SIZE` / `CHUNK_OVERLAP`           | `800` / `120`                                | Characters. Re-upload documents after changing.                                                          |
| `RETRIEVAL_STRATEGY`                     | `hybrid`                                     | `dense`, `sparse` or `hybrid`; also settable per request.                                                |
| `TOP_K` / `CANDIDATE_K` / `RRF_K`        | `5` / `20` / `60`                            | Chunks sent to the LLM / depth of each ranked list / RRF constant.                                       |
| `RERANK_ENABLED`                         | `false`                                      | Cross-encoder rerank of the top candidates (needs sentence-transformers).                                |

## 7. Tests

```bash
pip install pytest
pytest
```

The suite runs offline (hash embeddings, fake LLM server) and covers chunking offsets, BM25, RRF,
citation parsing, PDF page citations, the API end to end, the LLM HTTP clients, and the metrics.

## 8. Docker

```bash
cp .env.example .env
docker compose up --build        # API on http://localhost:8000
```

## Known limitations

- **Search index lives in memory** (loaded from SQLite on first query). Comfortable for thousands
  of chunks; for much larger corpora swap `app/db/vector_store.py` for pgvector, Qdrant or FAISS.
- **No OCR.** Scanned PDFs have no text layer; run `ocrmypdf` on them first.
- **Chunks do not cross page boundaries**, and `.docx/.txt/.md` files are treated as a single
  "page 1" (the chunk's character offsets still locate the text).
- **Ingestion is synchronous** — a large PDF blocks the upload request until it is indexed.
- **No authentication or per-user isolation.** Fine for local use; add auth before deploying.
- Answers are not streamed token by token.
#   d o c q a  
 