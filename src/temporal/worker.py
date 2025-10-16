"""Temporal worker entry point for the RAG0 workflows."""

from __future__ import annotations

import argparse
import asyncio

from temporalio.client import Client
from temporalio.worker import Worker

from ..activities import (
    answer_query_activity,
    detect_document_type_activity,
    index_stats_activity,
    ingest_command_activity,
    parse_cli_command_activity,
    parse_document_activity,
    question_command_activity,
    quit_command_activity,
    render_cli_menu_activity,
    stats_command_activity,
    store_parsed_document_activity,
    update_index_activity,
)
from ..workflows import (
    IngestionWorkflow,
    MainWorkflow,
    QuestionWorkflow,
)


async def _run_worker(address: str, namespace: str, task_queue: str) -> None:
    client = await Client.connect(address, namespace=namespace)
    worker = Worker(
        client,
        task_queue=task_queue,
        workflows=[IngestionWorkflow, QuestionWorkflow, MainWorkflow],
        activities=[
            render_cli_menu_activity,
            parse_cli_command_activity,
            ingest_command_activity,
            question_command_activity,
            stats_command_activity,
            quit_command_activity,
            detect_document_type_activity,
            parse_document_activity,
            store_parsed_document_activity,
            update_index_activity,
            answer_query_activity,
            index_stats_activity,
        ],
    )

    await worker.run()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Temporal worker for the RAG0 workflows",
    )
    parser.add_argument(
        "--address",
        default="127.0.0.1:7233",
        help="Temporal server address (host:port)",
    )
    parser.add_argument(
        "--namespace",
        default="default",
        help="Temporal namespace",
    )
    parser.add_argument(
        "--task-queue",
        default="rag0",
        help="Temporal task queue",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    asyncio.run(_run_worker(args.address, args.namespace, args.task_queue))


if __name__ == "__main__":  # pragma: no cover
    main()
