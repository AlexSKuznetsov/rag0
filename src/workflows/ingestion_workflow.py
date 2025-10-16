"""Temporal workflow that ingests a document and updates the index."""

from __future__ import annotations

from typing import Any, Dict, Optional

from temporalio import workflow

DETECT_DOCUMENT_ACTIVITY = "detect_document_type_activity"
PARSE_DOCUMENT_ACTIVITY = "parse_document_activity"
STORE_PARSED_ACTIVITY = "store_parsed_document_activity"
UPDATE_INDEX_ACTIVITY = "update_index_activity"


@workflow.defn
class IngestionWorkflow:
    """Workflow that ingests a document and updates the index."""

    @workflow.run
    async def run(self, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        payload = payload or {}
        source_path = payload.get("source_path")
        parsed_dir = payload.get("parsed_dir", "parsed")
        index_dir = payload.get("index_dir", "storage/index")
        chunk_size = int(payload.get("chunk_size", 700))
        chunk_overlap = int(payload.get("chunk_overlap", 150))
        chunk_merge_threshold = int(payload.get("chunk_merge_threshold", 60))

        if not source_path:
            raise ValueError("source_path must be provided in payload")

        detection_result = await workflow.execute_activity(
            DETECT_DOCUMENT_ACTIVITY,
            args=(source_path,),
            schedule_to_close_timeout=workflow.timedelta(minutes=2),
        )
        document_type = detection_result["document_type"]

        parsed_result = await workflow.execute_activity(
            PARSE_DOCUMENT_ACTIVITY,
            args=(source_path, document_type, chunk_size, chunk_overlap, chunk_merge_threshold),
            schedule_to_close_timeout=workflow.timedelta(minutes=5),
        )

        stored_result = await workflow.execute_activity(
            STORE_PARSED_ACTIVITY,
            args=(parsed_result["metadata"], parsed_result["content"], parsed_dir),
            schedule_to_close_timeout=workflow.timedelta(minutes=2),
        )

        await workflow.execute_activity(
            UPDATE_INDEX_ACTIVITY,
            args=(
                stored_result["parsed_path"],
                index_dir,
                stored_result.get("metadata_path"),
            ),
            schedule_to_close_timeout=workflow.timedelta(minutes=2),
        )

        return {
            "parsed_path": stored_result["parsed_path"],
            "document_type": document_type,
            "source_path": source_path,
        }


__all__ = ["IngestionWorkflow"]
