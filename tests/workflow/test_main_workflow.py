import asyncio

import pytest
from src.activities import (
    ingest_command_activity,
    parse_cli_command_activity,
    question_command_activity,
    quit_command_activity,
    render_cli_menu_activity,
    stats_command_activity,
)
from src.workflows import MainWorkflow, MainWorkflowConfig

temporalio_testing = pytest.importorskip("temporalio.testing")
temporalio_worker = pytest.importorskip("temporalio.worker")


@pytest.mark.asyncio
async def test_main_workflow_handles_quit() -> None:
    env = await temporalio_testing.WorkflowEnvironment.start_time_skipping()
    try:
        async with temporalio_worker.Worker(
            env.client,
            task_queue="test-main-workflow",
            workflows=[MainWorkflow],
            activities=[
                render_cli_menu_activity,
                parse_cli_command_activity,
                ingest_command_activity,
                question_command_activity,
                stats_command_activity,
                quit_command_activity,
            ],
        ):
            handle = await env.client.start_workflow(
                MainWorkflow.run,
                MainWorkflowConfig(),
                id="test-main-workflow-quit",
                task_queue="test-main-workflow",
            )
            await handle.signal(MainWorkflow.submit_input, "/quit")
            result = await asyncio.wait_for(handle.result(), timeout=5)
    finally:
        await env.shutdown()

    assert result["status"] == "quit"
    assert result["command"] == "quit"
