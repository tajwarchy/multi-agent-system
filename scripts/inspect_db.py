"""
scripts/inspect_db.py
CLI tool to inspect the SQLite agent step log.

Usage:
    python scripts/inspect_db.py                      # list recent query IDs
    python scripts/inspect_db.py <query_id>           # show full trace for a run
"""

import sys
from rich.console import Console
from rich.table   import Table
from rich         import box

console = Console()

def list_runs() -> None:
    from src.logging.sqlite_logger import get_all_query_ids, get_trace
    ids = get_all_query_ids()
    if not ids:
        console.print("[yellow]No runs logged yet. Run a query first.[/yellow]")
        return

    tbl = Table(title="Recent query runs", box=box.SIMPLE_HEAVY)
    tbl.add_column("Query ID",   style="cyan",   width=38)
    tbl.add_column("Query",      style="white",  width=45)
    tbl.add_column("Route",      style="yellow", width=18)
    tbl.add_column("Steps",      width=6)
    tbl.add_column("Timestamp",  style="dim",    width=22)

    for qid in ids[:20]:
        trace = get_trace(qid)
        if not trace:
            continue
        tbl.add_row(
            qid,
            trace[0]["query"][:44],
            trace[0]["route"] or "—",
            str(len(trace)),
            trace[0]["timestamp"][:19],
        )
    console.print(tbl)
    console.print(f"\nRun: [bold]python scripts/inspect_db.py <query_id>[/bold] to see full trace.")


def show_trace(query_id: str) -> None:
    from src.logging.sqlite_logger import get_trace
    trace = get_trace(query_id)
    if not trace:
        console.print(f"[red]No trace found for query_id={query_id}[/red]")
        return

    console.print(f"\n[bold]Query:[/bold]    {trace[0]['query']}")
    console.print(f"[bold]Route:[/bold]    [yellow]{trace[0]['route']}[/yellow]")
    console.print(f"[bold]Query ID:[/bold] {query_id}\n")

    tbl = Table(title="Agent Trace", box=box.SIMPLE_HEAVY, show_lines=True)
    tbl.add_column("#",          style="dim",     width=3)
    tbl.add_column("Agent",      style="cyan",    width=14)
    tbl.add_column("Tool",       style="magenta", width=30)
    tbl.add_column("Latency",    style="yellow",  width=10)
    tbl.add_column("Success",    width=8)
    tbl.add_column("Input",      width=30)
    tbl.add_column("Output",     width=40)

    for i, step in enumerate(trace, 1):
        tbl.add_row(
            str(i),
            step["agent_name"],
            step["tool_called"] or "—",
            f"{step['latency_ms']:.0f} ms" if step["latency_ms"] else "—",
            "[green]✓[/green]" if step["success"] else "[red]✗[/red]",
            (step["input"]  or "")[:28],
            (step["output"] or "")[:38],
        )
    console.print(tbl)

    errors = trace[0].get("errors", [])
    if errors:
        console.print(f"\n[red]Errors:[/red] {errors}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        show_trace(sys.argv[1])
    else:
        list_runs()