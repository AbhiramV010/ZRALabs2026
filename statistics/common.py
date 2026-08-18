"""Shared helpers for the statistics reports: data, escaping, tables."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

RESULTS = json.loads((HERE / "bench_results.json").read_text(encoding="utf-8"))

# ---------------------------------------------------------------- helpers

def esc(text):
    return (str(text).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def fmt(value, places=0):
    return f"{value:,.{places}f}"


def path_from(points):
    return " ".join(
        ("M" if index == 0 else "L") + f"{x:.2f} {y:.2f}"
        for index, (x, y) in enumerate(points)
    )


def table(headers, rows, aligns=None, caption=None):
    aligns = aligns or ["left"] + ["right"] * (len(headers) - 1)
    head = "".join(
        f'<th scope="col" class="a-{a}">{esc(h)}</th>'
        for h, a in zip(headers, aligns)
    )
    body = ""
    for row in rows:
        cells = "".join(
            (f'<th scope="row" class="a-{a}">{c}</th>' if index == 0
             else f'<td class="a-{a}">{c}</td>')
            for index, (c, a) in enumerate(zip(row, aligns))
        )
        body += f"<tr>{cells}</tr>"
    cap = f"<caption>{esc(caption)}</caption>" if caption else ""
    return (f'<div class="scroll"><table>{cap}<thead><tr>{head}</tr></thead>'
            f"<tbody>{body}</tbody></table></div>")


def datatable(summary, headers, rows, aligns=None):
    """A chart's table twin, folded away but always reachable."""
    return (f'<details class="twin"><summary>{esc(summary)}</summary>'
            f"{table(headers, rows, aligns)}</details>")
