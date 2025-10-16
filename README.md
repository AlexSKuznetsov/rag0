# RAG0 Document Processing Workflow

RAG0 orchestrates an end-to-end ingestion and retrieval experience on top of Temporal. Temporal supplies durable execution so long-running ingestion, grading, and retrieval steps survive restarts, capture auditable event history, and run across scalable worker pools. Documents flow from the CLI through ingestion activities into a Chroma-backed index, while an ask workflow—built with LangGraph and an Ollama model—answers questions with graded reflection and citations.

## Project Layout

```
rag0/
├── src/
│   ├── app.py                 # CLI entry point used by Makefile targets
│   ├── activities/            # Temporal activities (ingest, parse, ask, stats)
│   ├── agents/
│   │   └── ask/               # LangGraph agent (graph, nodes, state, config)
│   ├── ingestion/             # Detection, parsers, chunking, storage, vector store
│   ├── temporal/              # Worker bootstrap wiring workflows + activities
│   ├── utils/                 # Shared helpers for logging, settings, tracing
│   └── workflows/             # Temporal workflows (main, ingestion, question)
├── parsed/                    # Structured artifacts emitted by ingestion
├── storage/
│   └── index/                 # ChromaDB embeddings and metadata cache
├── dev.sh                     # Helper script to launch Temporal + worker + CLI
├── Makefile                   # Automation entrypoints (install, worker, workflows)
└── README.md
```

## Stack

- [Python 3.11+](https://www.python.org/) with [`uv`](https://github.com/astral-sh/uv)-powered dependency management (Makefile fallback to `pip`).
- [Temporal Server](https://temporal.io/) and the [Temporal Python SDK](https://docs.temporal.io/dev-guide/python/) for workflow orchestration and scheduling.
- [Docling](https://github.com/DS4SD/docling) for text-first PDF ingestion, paired with optional [LlamaIndex](https://www.llamaindex.ai/) normalization.
- Vision-capable LLMs (e.g., [Qwen 3-VL](https://qwenlm.github.io/en/) via [Ollama](https://ollama.com/)) to OCR scanned PDFs and images.
- [LangGraph](https://langchain-ai.github.io/langgraph/) + [LangChain](https://www.langchain.com/) to build the ask agent, including grading nodes and query rewriting.
- [ChromaDB](https://www.trychroma.com/) as the embedded vector database backed by `storage/index/`.
- [Granite Embedding](https://ollama.com/library/granite-embedding) model served via [Ollama](https://ollama.com/) for dense vector representations.
- [`python-dotenv`](https://github.com/theskumar/python-dotenv) for environment configuration, [`pytest`](https://docs.pytest.org/) for automated testing, and [`logging`](https://docs.python.org/3/library/logging.html) for observability.

## Workflow Overview

```mermaid
flowchart LR
    CLI[/CLI or Makefile<br/>commands/] --> MainWF{MainWorkflow<br/>src/workflows/main_workflow.py}

    subgraph Temporal Worker
        MainWF -->|/ingest| IngestWF[IngestionWorkflow<br/>src/workflows/ingestion_workflow.py]
        MainWF -->|/ask| AskWF[QuestionWorkflow<br/>src/workflows/question_workflow.py]
    end

    subgraph Ingestion Pipeline
        IngestWF --> Detect[Document detection<br/>src/ingestion/detector.py]
        Detect --> Parse[Parser selection<br/>src/ingestion/doc_parser.py<br/>src/ingestion/vision_parser.py]
        Parse --> Normalize[Normalization & chunking<br/>src/ingestion/chunking.py]
        Normalize --> Persist[Persist artifacts<br/>src/activities/store.py -> parsed/]
        Persist --> Index[Update embeddings<br/>src/activities/vector_index.py -> storage/index/]
    end

    subgraph Ask Pipeline
        AskWF --> Agent[Ask agent graph<br/>src/agents/ask/graph.py]
        Agent --> Retrieve[Retriever + grading<br/>LangGraph nodes]
        Retrieve --> Respond[Answer synthesis<br/>Ollama + reflection]
    end

    Respond --> CLI
    Index --> Retrieve
```

## Event Loop

![Event Loop Diagram](docs/event-loop.png)

Temporal acts as the event loop for the system. Commands issued through `src/app.py` are translated into workflow invocations that run inside the worker (`src/temporal/worker.py`). The worker pulls tasks from Temporal, executes the appropriate activities, and emits status events back to the CLI. This feedback loop keeps ingestion, indexing, and ask operations responsive even when long-running parsing or retrieval jobs are in flight.

## Ingestion Workflow

![Ingestion Workflow Diagram](docs/ingestion-workflow.png)

1. **Detect** – `detect_document_type_activity` inspects the payload (PDF, image, text) using `src/ingestion/detector.py` and selects the right parser strategy.
2. **Parse** – Docling handles text-first PDFs (`src/ingestion/doc_parser.py`), while the vision parser (`src/ingestion/vision_parser.py`) routes pages through a multimodal LLM for OCR.
3. **Normalize & Chunk** – `src/ingestion/chunking.py` standardizes metadata, deduplicates content, and generates retrieval-ready chunks.
4. **Persist** – Structured artifacts are written to `parsed/` via `store_parsed_document_activity`, preserving the raw extraction output for debugging.
5. **Index** – `update_vector_index_activity` leverages `src/ingestion/vector_store.py` to embed chunks, update the Chroma collection under `storage/index/`, and register document fingerprints for future refreshes.

## Ask Workflow

![Ask Workflow Diagram](docs/ask-workflow.png)

1. **Dispatch** – The CLI issues `/ask` commands that enqueue `QuestionWorkflow` executions with an `AskAgentConfig` payload.
2. **Agent Loop** – `src/agents/ask/graph.py` builds a LangGraph with `grade_documents`, `rewrite_query`, and `grade_answer` nodes interleaved between reasoning and response.
3. **Retrieval** – Multi-query retrieval pulls from the Chroma index (`storage/index/`), deduplicating chunks and expanding neighborhoods when reflection requires new context.
4. **Synthesis** – Responses are generated through Ollama, enriched with citations, and backed by reasoning traces so operators can inspect decision points.
5. **Fallbacks & Stats** – If the LLM is unavailable, the fallback responder surfaces multiple context snippets. Activity metrics are sent through `src/activities/stats.py` to power future observability dashboards.

## Setup

### Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
make install  # installs rag0 (editable) with dev tooling via uv, fallback to pip
```

### Configure environment variables

```bash
cp .env.example .env
# edit .env with your preferred editor to adjust defaults
```

### Install Temporal CLI (if needed)

```bash
# macOS
brew install temporal

# Linux
curl -sSf https://temporal.download/cli.sh | bash
```

Once installed, run `temporal server --help` to confirm the CLI is available.

### Prepare Ollama models (optional)

```bash
ollama pull qwen3:4b
ollama pull granite-embedding
```

## Usage

### Launch local stack

```bash
./dev.sh
```

The script ensures the Temporal dev server is running, starts the RAG0 worker, and drops you into the interactive CLI session. Use the prompt to run `/ingest` and `/ask` commands; press `Ctrl+C` to exit. Background processes shut down automatically when the session ends.

### Direct CLI access (optional)

If you need to run the CLI without the helper script (for example, when targeting a remote Temporal cluster), activate your virtualenv and invoke:

```bash
python -m src.app --ollama-model qwen3:4b --ask-top-k 4
```

### Configuration

The CLI and workflows read from flags or environment variables (via `.env`):

- `--ollama-model` / `RAG0_OLLAMA_MODEL`
- `--ollama-base-url` / `RAG0_OLLAMA_BASE_URL`
- `--ask-top-k` / `RAG0_ASK_TOP_K`
- `--ask-max-reflections` / `RAG0_ASK_MAX_REFLECTIONS`
- `--ask-min-citations` / `RAG0_ASK_MIN_CITATIONS`
- `--ask-reflection-enabled` / `RAG0_ASK_REFLECTION_ENABLED`
- `--ask-temperature` / `RAG0_ASK_TEMPERATURE`

Set `RAG0_ASK_REFLECTION_ENABLED=0` in `.env` (or pass `--ask-reflection-disabled`) to turn off the reflective grading loop. These values flow into `MainWorkflowConfig` and the `AskAgentConfig`, letting you tailor retrieval depth, reflection behavior, and model parameters per run.

## Developer Tooling

- `make lint` checks the tree with Ruff; `make format` applies Ruff formatting.
- `make typecheck` runs MyPy against `src/` and `tests/`.
- `make test` executes the pytest suite (CI captures coverage with `--cov=src`).
- Run `pre-commit install` after `make install` to enable the Ruff, MyPy, and pytest hooks.
- GitHub Actions (`.github/workflows/ci.yml`) mirrors these steps with uv-powered environments and uploads `coverage.xml`.

## Future Features

- API server to expose ingestion and ask workflows programmatically.
- Extended telemetry and stats dashboards sourced from workflow and activity metrics.
- Settings manager (CLI + file-based) for sharing configuration presets across teams.
- Scheduled re-ingestion and drift detection for long-lived document collections.
- Plug-in retrievers for external knowledge bases alongside the local Chroma index.

## Q&A

**Q:** Why use Temporal?  
**A:** Temporal makes the ingestion and ask orchestration more reliable by handling retries, state tracking, and long-running execution without custom infrastructure.

**Q:** Why do I need LangGraph if I can write logic with a Temporal workflow alone?  
**A:** You certainly can build everything inside Temporal, but this project demonstrates how familiar tools like LangGraph and LlamaIndex can plug into Temporal to add agentic retrieval logic without rebuilding it from scratch.
