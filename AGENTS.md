# Repository Guidelines

## Project Structure & Module Organization
Core logic lives under `src/`. `src/app.py` exposes the CLI used by the Makefile targets. `src/ingestion/` handles document detection, parsing, normalization, and the ChromaDB-backed vector manager; `src/workflows/` contains Temporal workflow definitions; `src/temporal/` houses the worker entry point; and `src/agents/` groups LangGraph-based agents such as the ask workflow. Parsed artifacts land in `parsed/`, raw inputs belong in `data/input/`, and embeddings persist under `storage/index/`. Keep experimental notebooks or scripts out of `src/` to avoid polluting the import graph.

## Build, Test, and Development Commands
Create a virtualenv (`python -m venv .venv && source .venv/bin/activate`) and install dependencies with `make install` (editable install using `uv` when available). Run `./dev.sh` to launch the Temporal dev server, worker, and interactive CLI; alternatively, use `make interactive` once the worker is running. Quality gates live in the Makefile: `make lint`, `make format`, `make typecheck`, and `make test`. Use `make clean` to wipe generated `parsed/`, `storage/`, cache directories, and logs after experiments. Install git hooks with `pre-commit install` to mirror the CI checks locally.

## Coding Style & Naming Conventions
Follow PEP 8 with 4-space indentation, snake_case functions, and CapWords classes/enums as seen in `src/ingestion/models.py`. Type hints and docstrings are standard; maintain them for new modules. Prefer pydantic models or dataclasses when exchanging structured payloads between activities. Keep logging via `logging.getLogger(__name__)`; avoid print statements outside CLI feedback.

## Testing Guidelines
Adopt `pytest` for new coverage. Place tests under `tests/` mirroring the package layout (e.g., `tests/ingestion/test_pipeline.py`). Name test functions `test_<behavior>`. Target meaningful integration tests around Temporal-compatible activities by isolating them into pure functions where possible. Until CI is wired, run `pytest` locally before opening a PR and include fixtures that exercise sample documents in `data/input/` without committing large binaries.

## Commit & Pull Request Guidelines
The repo has no formal history yet—use Conventional Commits (`feat:`, `fix:`, `chore:`) to establish consistency. Keep commit bodies concise and reference issue IDs when relevant. Pull requests should outline scope, manual test evidence (commands run, sample inputs), and any follow-up work. Attach screenshots or log excerpts if behavior changes ingest outputs or worker runtime.

## Temporal & Configuration Notes
Temporal endpoints, namespaces, and task queues default to the values wired in the Makefile; override them via `make ... ADDRESS=host:port` or CLI flags when targeting remote clusters. Store secrets (API keys for OCR models, etc.) in a `.env` file loaded via `python-dotenv`, but never commit it. Clean up stale `storage/` indices when schema changes to avoid inconsistent embeddings.

## Ask Agent Reflection Flow
- The LangGraph ask agent now mirrors the tutorial’s agentic RAG loop with `grade_answer`, `grade_documents`, and `rewrite_query` nodes inserted between reasoning and response. Each grading pass decides whether to route back through retrieval or finish the response.
- Reflection is controlled through `AskAgentConfig` (`reflection_enabled`, `max_reflections`, `min_citations`) and can be tuned via CLI flags (`--ask-max-reflections`, `--ask-min-citations`, `--ask-reflection-disabled`). Neighbor expansion and deduplicated chunk metadata ensure revised queries surface diverse context.
- Reasoning traces record the grading outcomes and generated follow-up queries so operators can inspect why additional retrieval occurred. The fallback responder also surfaces multiple snippets when the LLM is unavailable, aligning with the LangGraph agentic tutorial guidance.
