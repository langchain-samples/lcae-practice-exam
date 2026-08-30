"""Upload the LCAE practice exam traces into your LangSmith workspace.

Usage:
    cp .env.example .env
    # paste your LANGSMITH_API_KEY into .env
    uv sync
    uv run python upload_traces.py

The script reads `traces.json`, creates runs (and any feedback) via the
LangSmith SDK, and prints the URL of the seeded project when done.
"""

from __future__ import annotations

import json
import os
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv
from langsmith import Client

LANGSMITH_API_URL = "https://api.smith.langchain.com"
TRACES_FILE = Path(__file__).resolve().parent / "traces.json"


def _dotted(start: datetime, run_id: uuid.UUID, parent: str | None = None) -> str:
    ts = start.strftime("%Y%m%dT%H%M%S%f") + "Z"
    leaf = f"{ts}{run_id}"
    return f"{parent}.{leaf}" if parent else leaf


def _llm_run_extra(model: str, provider: str, input_tokens: int, output_tokens: int, depth: int) -> dict:
    return {
        "metadata": {
            "ls_model_name": model,
            "ls_provider": provider,
            "ls_run_depth": depth,
            "usage_metadata": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            },
        }
    }


def _resolve_depth(run_defs: dict[str, dict], run_id: str) -> int:
    depth = 0
    current = run_defs[run_id].get("parent")
    while current is not None:
        depth += 1
        current = run_defs[current].get("parent")
    return depth


def upload_trace(client: Client, project: str, trace: dict) -> dict[str, uuid.UUID]:
    """Create every run in a trace and return the {logical_id: uuid} map."""
    runs_by_id: dict[str, dict] = {r["id"]: r for r in trace["runs"]}
    ids: dict[str, uuid.UUID] = {rid: uuid.uuid4() for rid in runs_by_id}
    trace_metadata = trace.get("metadata", {})

    trace_start = datetime.now(timezone.utc) + timedelta(minutes=trace["start_offset_minutes"])

    absolute: dict[str, tuple[datetime, datetime]] = {}
    for rid, run in runs_by_id.items():
        start = trace_start + timedelta(milliseconds=run["start_ms"])
        end = start + timedelta(milliseconds=run["duration_ms"])
        absolute[rid] = (start, end)

    def order_key(rid: str) -> tuple:
        return (_resolve_depth(runs_by_id, rid), absolute[rid][0])

    dotted: dict[str, str] = {}
    for rid in sorted(runs_by_id, key=order_key):
        run = runs_by_id[rid]
        parent = run.get("parent")
        parent_dotted = dotted[parent] if parent else None
        dotted[rid] = _dotted(absolute[rid][0], ids[rid], parent_dotted)

    root_id = next(rid for rid, run in runs_by_id.items() if run.get("parent") is None)

    for rid in sorted(runs_by_id, key=order_key):
        run = runs_by_id[rid]
        parent = run.get("parent")
        start, end = absolute[rid]

        extra: dict = {}
        if run["type"] == "llm":
            extra = _llm_run_extra(
                model=run["model"],
                provider=run["provider"],
                input_tokens=run["input_tokens"],
                output_tokens=run["output_tokens"],
                depth=_resolve_depth(runs_by_id, rid),
            )
        # Trace-level metadata (account, user_id, version) is propagated to
        # every run so LangSmith's sidebar aggregations (which group by run-
        # level metadata) can attribute token counts and cost to the account.
        extra.setdefault("metadata", {})
        extra["metadata"].update(trace_metadata)

        create_kwargs = {
            "id": ids[rid],
            "name": run["name"],
            "run_type": run["type"],
            "project_name": project,
            "trace_id": ids[root_id],
            "dotted_order": dotted[rid],
            "inputs": run.get("inputs", {}),
            "start_time": start,
            "end_time": end,
            "extra": extra,
        }
        if parent is not None:
            create_kwargs["parent_run_id"] = ids[parent]
        if run.get("error"):
            create_kwargs["error"] = run["error"]
        else:
            create_kwargs["outputs"] = run.get("outputs", {})

        client.create_run(**create_kwargs)

    return ids


def main() -> int:
    load_dotenv(Path(__file__).resolve().parent / ".env")
    api_key = os.environ.get("LANGSMITH_API_KEY")
    if not api_key:
        print(
            "ERROR: LANGSMITH_API_KEY is not set. Copy .env.example to .env and add your key.",
            file=sys.stderr,
        )
        return 1

    # The LangSmith SDK auto-captures any LANGSMITH_* env var into every trace's
    # metadata. Strip the ones that add noise without helping the practice-exam
    # questions (Cursor IDE integration, endpoint URLs) so the important keys
    # (version, user_id) are easy to find in the UI.
    for noisy_var in ("LANGSMITH_CURSOR_ENDPOINT", "LANGSMITH_CURSOR_PROJECT", "LANGSMITH_ENDPOINT"):
        os.environ.pop(noisy_var, None)

    payload = json.loads(TRACES_FILE.read_text())
    project = payload["project_name"]
    traces = payload["traces"]

    client = Client(api_key=api_key, api_url=LANGSMITH_API_URL)

    # Ensure the project exists up front so we can use its session_id when
    # attaching feedback (create_feedback now requires it).
    try:
        session_id = client.read_project(project_name=project).id
    except Exception:
        session_id = client.create_project(project_name=project).id

    print(f"Uploading {len(traces)} traces to project '{project}' ...")

    for i, trace in enumerate(traces, 1):
        id_map = upload_trace(client, project, trace)

        # Flush so any feedback attaches to a persisted run.
        client.flush()

        for fb in trace.get("feedback", []):
            client.create_feedback(
                run_id=id_map[fb["run_id"]],
                key=fb["key"],
                score=fb["score"],
                session_id=session_id,
            )

        if i % 5 == 0 or i == len(traces):
            print(f"  {i}/{len(traces)} uploaded")

    client.flush()

    ui_base = LANGSMITH_API_URL.replace("api.smith", "smith")
    print("Done.")
    print(f"  project: {project}")
    print(f"  URL:     {ui_base}/o/-/projects/p/{project}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
