# Main Orchestration Agent Instructions

If you are the main orchestration agent, ignore the steps below. Your job is to spawn the worker agents and monitor their progress.

# Worker Agent Instructions

## Purpose
This file defines strict execution rules for all worker agents in this repository.

## Required First Step
1. Open `checklist.md`.
2. Determine the single step assigned to you from that file.

## Single-Step Execution Rule (Strict)
1. Do **exactly one** assigned step.
2. Do **not** start a second step.
3. Do **not** pick up "next" work automatically.

## Stop Rule (Strict)
After completing your one step:
1. Record completion/update status in `checklist.md` (if instructed by checklist format).
2. Stop immediately.
3. Return control to the orchestrator/user.

## If Blocked
If `checklist.md` is missing, unclear, or does not specify your step:
1. Do not implement anything.
2. Report: `BLOCKED: missing or unclear step assignment in checklist.md`.
3. Stop.

## Scope Guardrails
1. Stay within the assigned step's files and acceptance criteria.
2. Do not refactor unrelated code.
3. Do not change requirements for other steps.
4. If an edit outside your step is absolutely required, report blocker and stop.

