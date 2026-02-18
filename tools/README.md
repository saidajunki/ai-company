# ai-company: Shared Internal Tools

This directory is the **shared tool belt** for CEO/Sub-agents.

## Goals

- Avoid re-inventing shell commands (drift + cost).
- Provide repeatable, documented actions anyone can run.

## Structure

- Runner: `/opt/apps/ai-company/tools/ai`
- Executable tools: `/opt/apps/ai-company/tools/bin/<tool>`
- Tool docs: `/opt/apps/ai-company/tools/docs/<tool>.md`

## Conventions

- Tool name: `^[a-z][a-z0-9_-]{1,40}$`
- Tools must be **idempotent** when possible.
- Tools must include `--help`/`help` usage.
- If a tool is used by multiple agents, its doc must explain:
  - Purpose (what drift/cost it prevents)
  - Inputs/outputs
  - Examples
  - Source of Truth paths

## Create a new tool

```bash
/opt/apps/ai-company/tools/ai new <tool>
$EDITOR /opt/apps/ai-company/tools/bin/<tool>
$EDITOR /opt/apps/ai-company/tools/docs/<tool>.md
```

Then commit & push.

## When to create a tool vs a procedure SoT

- **Tool**: logic is reusable and benefits from being executable.
- **Procedure SoT**: a fixed runbook (multi-line commands) that should be re-postable verbatim.

