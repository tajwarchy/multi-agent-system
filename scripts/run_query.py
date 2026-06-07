"""
scripts/run_query.py
CLI test harness for the multi-agent graph.
Runs a query end-to-end and pretty-prints the full agent trace.

Usage:
    python scripts/run_query.py "What is the square root of 144?"
    python scripts/run_query.py "Who invented the internet?"
    python scripts/run_query.py "What is 15% of 250, and what is compound interest?"
"""

import logging
import sys
import time

from rich.console import Console
from rich.panel   import Panel
from rich.table   import Table
from rich         import box

# Configure logging before importing graph modules
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)

from src.graph.graph import run_graph

console = Console()


def main() -> None:
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What is the square root of 256?"

    console.print(Panel(f"[bold cyan]Query:[/bold cyan] {query}", title="Multi-Agent System"))

    t0     = time.monotonic()
    result = run_graph(query)
    total  = (time.monotonic() - t0) * 1000

    # ── Final answer ──────────────────────────────────────────────────────────
    console.print(Panel(
        result.get("final_answer") or "[red]No answer generated[/red]",
        title="[bold green]Final Answer[/bold green]",
    ))

    # ── Routing info ──────────────────────────────────────────────────────────
    console.print(f"\n[bold]Route:[/bold]    [yellow]{result.get('route')}[/yellow]")
    console.print(f"[bold]Query ID:[/bold] {result.get('query_id')}  [dim](use this for /trace/{{query_id}})[/dim]")
    console.print(f"[bold]Total:[/bold]    {total:.0f} ms\n")

    # ── Agent trace table ─────────────────────────────────────────────────────
    trace = result.get("agent_trace", [])
    if trace:
        tbl = Table(title="Agent Trace", box=box.SIMPLE_HEAVY, show_lines=True)
        tbl.add_column("#",           style="dim",    width=3)
        tbl.add_column("Agent",       style="cyan",   width=14)
        tbl.add_column("Tool",        style="magenta",width=34)
        tbl.add_column("Latency",     style="yellow", width=10)
        tbl.add_column("Success",     width=8)
        tbl.add_column("Output (preview)", style="white", width=50)

        for i, step in enumerate(trace, 1):
            tbl.add_row(
                str(i),
                step.get("agent_name", ""),
                step.get("tool_called") or "—",
                f"{step.get('latency_ms', 0):.0f} ms",
                "[green]✓[/green]" if step.get("success") else "[red]✗[/red]",
                str(step.get("output", ""))[:100].replace("\n", " "),
            )
        console.print(tbl)

    # ── Errors ────────────────────────────────────────────────────────────────
    errors = result.get("errors", [])
    if errors:
        console.print(Panel(
            "\n".join(f"• {e}" for e in errors),
            title="[bold red]Errors / Warnings[/bold red]",
        ))
    else:
        console.print("[bold green]No errors.[/bold green]")


if __name__ == "__main__":
    main()