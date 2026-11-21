# Claude Code CLI (`claude`) Project Instructions — `applied-ai-foundry` (v4.0)

Always load and follow the workspace agent instructions, v4.0 curriculum, and daily progress tracker defined in:
- @AGENTS.md
- @LEARNING_PLAN.md
- @GROK_CAREER_AND_PROGRESS_SYNC.md

## Developer Environment & Execution Rules
1. **IDE & Terminal Stack**:
   - The developer works in **VS Code IDE** + **Antigravity Desktop App** (with the terminal pane open on the right side for project execution) + **Claude Code CLI (`claude`)**.
   - **Virtual Environment**: Always execute Python and pip commands explicitly via `.venv/bin/python` and `.venv/bin/pip` (never global `python3` or `pip`).
2. **Hardware & Memory Guardrail (Apple M4 24GB Unified Memory)**:
   - Before running heavy PyTorch `mps` (`src/week1_mps_starter.py`) or `mlx-lm` jobs, remind the user to unload LM Studio (`lms unload --all`) if `Qwen3-Coder-30B-A3B` (`Q3_K_L`, ~14.5GB) is loaded.
3. **Anti-Vibe-Coding Gate**:
   - **Never** auto-complete or overwrite the user's hand-written 25–30 line core weekly algorithm katas (`src/week1_bpe_kata.py`, SVD weight dissection, PyTorch forward hooks, MoE MSAN pruning, NPO loss).
   - Use Claude Code CLI for **scaffolding, telemetry harnesses, `FastMCP` client testing (`claude mcp add`), `.claude/settings.json` `PreToolUse` security hooks, headless `-p` eval triage, and Week 5–6 Docker/CLI/Cloud Run packaging**.
4. **Daily Google Drive Sync (`gdrive`)**:
   - Google Drive Folder (`AI-FDE-Sprint-2026`): `1_IHEWhtTKvCpsOoPqGL36_G_42Gc8qXO`
   - Active v4.0 `LEARNING_PLAN.md` ID: `1DTY2snGS1CdS-VAa23luE4wC5p4X7JwMGXMpahVESeQ`
   - Active v4.0 `GROK_CAREER_AND_PROGRESS_SYNC.md` ID: `1YwnHeVfQye214j5P_o3ID_NiGCQAJRLN1MTzKQhfy1g`
