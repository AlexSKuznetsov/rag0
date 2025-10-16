# CLI ↔ Temporal Workflow Event Timeline

```mermaid
sequenceDiagram
    autonumber
    participant CLI as CLI (`src/app.py`)
    participant MainWF as MainWorkflow (`src/workflows/main_workflow.py`)
    participant RenderMenu as render_cli_menu_activity
    participant ParseCmd as parse_cli_command_activity

    CLI->>MainWF: Start workflow (`Client.start_workflow`)
    MainWF->>RenderMenu: execute_activity( render_cli_menu )
    RenderMenu-->>MainWF: {"prompt": MENU_TEXT}
    MainWF-->>CLI: query get_next_prompt → revision=1
    CLI->>CLI: _await_prompt() prints menu (flush=True)
    loop User command loop
        CLI->>MainWF: signal submit_input(raw_input)
        alt empty input
            MainWF->>RenderMenu: execute_activity( render_cli_menu )
            RenderMenu-->>MainWF: {"prompt": MENU_TEXT}
            MainWF-->>CLI: query get_next_prompt → revision++
            CLI->>CLI: _await_prompt() refreshes menu
        else non-empty input
            MainWF->>ParseCmd: execute_activity( parse_cli_command )
            ParseCmd-->>MainWF: {"command": …, "arguments": …}
            MainWF->>MainWF: dispatch command (activity or child workflow)
            MainWF-->>CLI: query get_last_result → revision++
            CLI->>CLI: _poll_for_result() prints response
        end
    end
    CLI->>MainWF: signal submit_input("/quit")
    MainWF-->>CLI: query get_last_result → status="quit"
    CLI-->>MainWF: await handle.result() (graceful shutdown)
```

## Notes from First-Run Investigation
- Initial menu text is emitted by `render_cli_menu_activity`; the CLI polls `MainWorkflow.get_next_prompt` via `_await_prompt`.
- Adding `flush=True` to the CLI print ensures the menu renders before the first user input even when stdout buffering is enabled.
- Workflow activity completion events confirm the menu was produced; the missing display was caused by buffered output on the CLI side.
