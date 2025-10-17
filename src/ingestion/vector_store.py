"""ChromaDB-backed local vector store manager using LlamaIndex."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, cast

import chromadb
from llama_index.core import Document, VectorStoreIndex
from llama_index.core.schema import QueryBundle
from llama_index.core.storage import StorageContext
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

# Reduce verbose logs from dependencies
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("ollama").setLevel(logging.WARNING)
logging.getLogger("llama_index").setLevel(logging.WARNING)
logging.getLogger("chromadb").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """Wrapper around a ChromaDB collection using LlamaIndex."""

    def __init__(
        self,
        storage_dir: Path,
        collection_name: str = "rag0",
        reset_collection: bool = False,
        embed_model_name: str = "granite-embedding",
    ) -> None:
        self._storage_dir = storage_dir
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        self._collection_name = collection_name
        self._embed_model_name = embed_model_name
        self._client = chromadb.PersistentClient(path=str(self._storage_dir))
        self._index: Optional[VectorStoreIndex] = None
        self._embed_model: Optional[OllamaEmbedding] = None
        self._chunk_sidecar_cache: Dict[str, List[Dict[str, Any]]] = {}

        if reset_collection:
            self._reset_collection()

        self._build_index()

    def _reset_collection(self) -> None:
        try:
            self._client.delete_collection(self._collection_name)
        except Exception as exc:  # pragma: no cover - chromadb cleanup best effort
            logger.debug("Could not delete collection %s: %s", self._collection_name, exc)

    def _build_index(self) -> None:
        collection = self._client.get_or_create_collection(name=self._collection_name)
        chroma_vector_store = ChromaVectorStore(chroma_collection=collection)
        storage_context = StorageContext.from_defaults(vector_store=chroma_vector_store)
        embed_model = OllamaEmbedding(model_name=self._embed_model_name)
        self._embed_model = embed_model

        if collection.count() > 0:
            self._index = VectorStoreIndex.from_vector_store(
                chroma_vector_store,
                embed_model=embed_model,
                storage_context=storage_context,
            )
        else:
            self._index = VectorStoreIndex(
                [],
                storage_context=storage_context,
                embed_model=embed_model,
            )

    def reset(self) -> None:
        """Reset the underlying Chroma collection and rebuild the index."""

        self._reset_collection()
        self._build_index()

    def is_available(self) -> bool:
        return self._index is not None

    def upsert_documents(self, documents: List[Dict[str, Any]]) -> None:
        if not documents or not self.is_available():
            return

        index = self._index
        if index is None:
            return

        for doc in documents:
            text = doc.get("text", "")
            if not text:
                continue
            metadata = self._clean_metadata(doc.get("metadata", {}))
            li_doc = Document(text=text, metadata=metadata)
            index.insert(li_doc)

    def _create_query_bundle(self, prompt: str) -> Optional[QueryBundle]:
        if not prompt.strip():
            return None
        embedding = None
        if self._embed_model is not None:
            try:
                embedding = self._embed_model.get_query_embedding(prompt)
            except Exception as exc:  # pragma: no cover - embedding failures should not break flow
                logger.warning("Falling back to raw query due to embedding error: %s", exc)
        return QueryBundle(query_str=prompt, embedding=embedding)

    def _retriever_input(self, prompt: str) -> Union[str, QueryBundle]:
        bundle = self._create_query_bundle(prompt)
        if bundle is None or bundle.embedding is None:
            return prompt
        return bundle

    @staticmethod
    def _dedupe_key(
        match: Dict[str, Any],
        fields: Sequence[str],
    ) -> Tuple[str, ...]:
        metadata = match.get("metadata", {}) or {}
        values: List[str] = []
        for field in fields:
            value = metadata.get(field)
            if value in (None, ""):
                continue
            values.append(str(value))
        if values:
            return tuple(values)

        source = metadata.get("source_path") or metadata.get("file_name") or "unknown"
        text = match.get("text", "")
        digest = hashlib.sha1(f"{source}:{text}".encode("utf-8")).hexdigest()
        return (source, digest)

    def _load_chunk_sidecar(self, metadata_path: str) -> List[Dict[str, Any]]:
        if not metadata_path:
            return []
        path = Path(metadata_path)
        if not path.exists():
            return []
        key = str(path.resolve())
        if key in self._chunk_sidecar_cache:
            return self._chunk_sidecar_cache[key]

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            chunks = payload.get("content", {}).get("chunks", []) or []
        except Exception as exc:  # pragma: no cover - best effort cache load
            logger.debug("Failed to load chunk metadata %s: %s", path, exc)
            chunks = []

        self._chunk_sidecar_cache[key] = chunks
        return chunks

    def _expand_neighbors(
        self,
        matches: Sequence[Dict[str, Any]],
        neighbor_span: int,
    ) -> List[Dict[str, Any]]:
        if neighbor_span <= 0:
            return []

        expansions: List[Dict[str, Any]] = []
        for match in matches:
            metadata = match.get("metadata", {}) or {}
            chunk_id = metadata.get("chunk_id")
            metadata_path = metadata.get("parsed_metadata_path")
            if not chunk_id or not metadata_path:
                continue
            chunk_list = self._load_chunk_sidecar(str(metadata_path))
            if not chunk_list:
                continue
            try:
                index = next(i for i, chunk in enumerate(chunk_list) if chunk.get("chunk_id") == chunk_id)
            except StopIteration:
                continue

            for offset in range(1, neighbor_span + 1):
                for neighbor_index in (index - offset, index + offset):
                    if neighbor_index < 0 or neighbor_index >= len(chunk_list):
                        continue
                    neighbor_chunk = chunk_list[neighbor_index]
                    text = neighbor_chunk.get("text", "")
                    if not text:
                        continue
                    neighbor_metadata = dict(neighbor_chunk.get("metadata") or {})
                    neighbor_metadata.setdefault("parsed_metadata_path", metadata_path)
                    neighbor_metadata.setdefault("source_path", metadata.get("source_path"))
                    neighbor_metadata.setdefault("file_name", metadata.get("file_name"))
                    neighbor_metadata["chunk_id"] = neighbor_chunk.get("chunk_id")
                    neighbor_metadata["chunk_index"] = neighbor_chunk.get("chunk_index")
                    neighbor_metadata["page_start"] = neighbor_chunk.get("page_start")
                    neighbor_metadata["page_end"] = neighbor_chunk.get("page_end")
                    neighbor_metadata["paragraph_start"] = neighbor_chunk.get("paragraph_start")
                    neighbor_metadata["paragraph_end"] = neighbor_chunk.get("paragraph_end")
                    expansions.append(
                        {
                            "text": text,
                            "metadata": neighbor_metadata,
                            "score": match.get("score", 0.0) + (offset * 0.001),
                        }
                    )

        return expansions

    def query(self, prompt: str, top_k: int = 3) -> List[Dict[str, Any]]:
        if not prompt.strip():
            return []

        index = self._index
        if index is None:
            return []

        retriever = index.as_retriever(similarity_top_k=top_k)
        retriever_input = self._retriever_input(prompt)
        results = retriever.retrieve(retriever_input)

        matches: List[Dict[str, Any]] = []
        for result in results:
            node = cast(Any, result.node)
            text = getattr(node, "text", "")
            metadata = dict(getattr(node, "metadata", {}) or {})
            matches.append(
                {
                    "text": text,
                    "metadata": metadata,
                    "score": result.score,
                }
            )
        return matches

    def multi_query(
        self,
        prompts: Sequence[str],
        top_k: int = 3,
        deduplicate: bool = True,
        neighbor_span: int = 0,
        dedupe_fields: Optional[Sequence[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Execute multiple related queries and optionally deduplicate results."""

        field_order = tuple(dedupe_fields) if dedupe_fields else ("source_path", "chunk_id")
        seen: Dict[Tuple[str, ...], Dict[str, Any]] = {}
        aggregated: List[Dict[str, Any]] = []
        seeds: List[Dict[str, Any]] = []

        def _record(match: Dict[str, Any]) -> None:
            if not deduplicate:
                aggregated.append(match)
                return

            key = self._dedupe_key(match, field_order)
            existing = seen.get(key)
            if existing:
                existing["score"] = min(existing.get("score", 0.0), match.get("score", 0.0))
                return
            seen[key] = match
            aggregated.append(match)

        for prompt in prompts:
            for match in self.query(prompt, top_k=top_k):
                seeds.append(match)
                _record(match)

        if neighbor_span > 0 and seeds:
            for neighbor in self._expand_neighbors(seeds, neighbor_span):
                _record(neighbor)

        aggregated.sort(key=lambda item: item.get("score", 0.0))
        logger.debug(
            "multi_query completed",
            extra={
                "prompt_count": len(prompts),
                "result_count": len(aggregated),
                "neighbor_span": neighbor_span,
            },
        )
        return aggregated

    @staticmethod
    def format_documents_for_llm(documents: Sequence[Dict[str, Any]]) -> str:
        """Render retrieved documents into an LLM-ready context block."""

        formatted_blocks: List[str] = []
        for idx, doc in enumerate(documents, start=1):
            text = doc.get("text", "").strip()
            if not text:
                continue
            metadata = doc.get("metadata", {})
            source = metadata.get("file_name") or metadata.get("source_path") or "unknown"
            header = f"[{idx}] {source}"
            chunk_index = metadata.get("chunk_index")
            if isinstance(chunk_index, int):
                header += f" [chunk {chunk_index + 1}]"
            page_start = metadata.get("page_start")
            page_end = metadata.get("page_end")
            page_single = metadata.get("page")
            if page_start is not None and page_end is not None:
                if page_start == page_end:
                    header += f" (page {page_start})"
                else:
                    header += f" (pages {page_start}-{page_end})"
            elif page_single is not None:
                header += f" (page {page_single})"
            formatted_blocks.append(f"{header}\n{text}")
        return "\n\n".join(formatted_blocks)

    @staticmethod
    def rerank_documents(
        documents: Sequence[Dict[str, Any]],
        max_per_source: int = 2,
    ) -> List[Dict[str, Any]]:
        if not documents:
            return []

        sorted_docs = sorted(documents, key=lambda item: item.get("score", 0.0))
        if max_per_source <= 0:
            return sorted_docs

        counts: Dict[str, int] = {}
        primary: List[Dict[str, Any]] = []
        overflow: List[Dict[str, Any]] = []

        for doc in sorted_docs:
            metadata = doc.get("metadata", {}) or {}
            source = metadata.get("source_path") or metadata.get("file_name") or "unknown"
            current = counts.get(source, 0)
            if current < max_per_source:
                primary.append(doc)
                counts[source] = current + 1
            else:
                overflow.append(doc)

        primary.extend(overflow)
        return primary

    @staticmethod
    def merge_adjacent_documents(documents: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not documents:
            return []

        merged: List[Dict[str, Any]] = []
        current: Optional[Dict[str, Any]] = None

        def _source(meta: Dict[str, Any]) -> str:
            return meta.get("source_path") or meta.get("file_name") or "unknown"

        for doc in documents:
            metadata = dict(doc.get("metadata", {}) or {})
            text = doc.get("text", "")
            if current is None:
                current = {"text": text, "metadata": metadata, "score": doc.get("score", 0.0)}
                continue

            current_meta = current.get("metadata", {}) or {}
            current_index = current_meta.get("chunk_index")
            chunk_index = metadata.get("chunk_index")
            same_source = _source(metadata) == _source(current_meta)
            contiguous = (
                same_source
                and isinstance(current_index, int)
                and isinstance(chunk_index, int)
                and chunk_index == current_index + 1
            )

            if same_source and contiguous:
                combined_text = f"{current['text'].rstrip()}\n\n{text.strip()}" if text else current["text"]
                current["text"] = combined_text
                current["score"] = min(current.get("score", 0.0), doc.get("score", 0.0))
                current_meta["chunk_index"] = chunk_index
                current_meta["chunk_id"] = metadata.get("chunk_id")
                current_meta["paragraph_end"] = metadata.get("paragraph_end")
                current_meta["page_end"] = metadata.get("page_end") or current_meta.get("page_end")
            else:
                merged.append(current)
                current = {"text": text, "metadata": metadata, "score": doc.get("score", 0.0)}

        if current is not None:
            merged.append(current)

        return merged

    @staticmethod
    def _clean_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
        clean = {}
        for key, value in metadata.items():
            if value is None:
                continue
            if isinstance(value, (str, int, float, bool)):
                clean[str(key)] = value
            else:
                clean[str(key)] = str(value)
        return clean

    @staticmethod
    def get_stats(storage_dir: Path, collection_name: str = "rag0") -> Dict[str, Any]:
        """Return basic statistics for an existing Chroma collection."""

        storage_path = Path(storage_dir)
        stats: Dict[str, Any] = {
            "storage_dir": str(storage_path),
            "collection_name": collection_name,
            "document_count": 0,
        }

        if not storage_path.exists():
            stats["storage_exists"] = False
            return stats

        stats["storage_exists"] = True

        chroma_client = chromadb.PersistentClient(path=str(storage_path))

        try:
            collection = chroma_client.get_collection(collection_name)
        except Exception:
            stats["collection_available"] = False
            stats["available_collections"] = [col.name for col in chroma_client.list_collections()]
            return stats

        stats["collection_available"] = True

        try:
            stats["document_count"] = collection.count()
        except Exception:
            stats["document_count_error"] = "count_failed"

        try:
            metadata = collection.metadata or {}
            if metadata:
                stats["collection_metadata"] = metadata
        except Exception:
            stats["metadata_error"] = "metadata_unavailable"

        return stats
