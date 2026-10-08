"""Combines saved evalforge runs into one leaderboard (Markdown + HTML).

Accuracy suites (QA, MCQ) score 0-100% of items passed. Generation suites
(translation, summary) score mean sentence chrF, 0-100. Every number comes with a
95% bootstrap confidence interval over items, because with ~25 items a few
points of difference is usually noise.
"""

from __future__ import annotations

import html
import json
import random
from collections import Counter
from pathlib import Path

from evalforge.report import _CSS

from .normalize import foreign_script_letters
from .suites import SUITES

BOOTSTRAP_SAMPLES = 2000


def item_scores(run: dict) -> list[float]:
    """Per-item score on a 0-100 scale for one run."""
    metric = SUITES.get(run["suite"], ("", "accuracy"))[1]
    if metric == "chrF":
        return [100 * r["value"] for r in run["results"]]
    return [100.0 * r["passed"] for r in run["results"]]


def bootstrap_ci(values: list[float], samples: int = BOOTSTRAP_SAMPLES, seed: int = 0) -> tuple[float, float]:
    """95% percentile bootstrap interval for the mean of `values`."""
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(rng.choices(values, k=n)) / n for _ in range(samples))
    return means[int(0.025 * samples)], means[int(0.975 * samples) - 1]


def _script_rate(runs: list[dict]) -> float | None:
    """Share of Amharic-output items where the model actually answered in Fidel."""
    total = bad = 0
    for run in runs:
        if run["suite"] in ("en_am", "am_sum"):
            for r in run["results"]:
                total += 1
                bad += r["detail"].startswith("not in Amharic script") or bool(r.get("error"))
    return 100 * (total - bad) / total if total else None


def _drift_rate(runs: list[dict]) -> float | None:
    """Share of Amharic-output items whose answer slips into a third script
    (3+ letters that are neither Fidel nor Latin, e.g. Chinese or Cyrillic)."""
    total = drift = 0
    for run in runs:
        if run["suite"] in ("en_am", "am_sum"):
            for r in run["results"]:
                total += 1
                drift += foreign_script_letters(r["answer"]) >= 3
    return 100 * drift / total if total else None


def build(runs: list[dict]) -> list[dict]:
    """One row per model: score and CI per suite, macro average, script rate, MCQ letter picks."""
    by_model: dict[str, list[dict]] = {}
    for run in runs:
        by_model.setdefault(run["model"], []).append(run)

    rows = []
    for model, model_runs in by_model.items():
        cells = {}
        for run in model_runs:
            vals = item_scores(run)
            mean = sum(vals) / len(vals) if vals else 0.0
            cells[run["suite"]] = {"score": mean, "ci": bootstrap_ci(vals), "n": len(vals)}
        picks = Counter()
        for run in model_runs:
            if run["suite"] == "am_mcq":
                picks.update(r["detail"].removeprefix("picked ") for r in run["results"] if r["detail"].startswith("picked "))
        complete = all(s in cells for s in SUITES)
        rows.append({
            "model": model,
            "suites": cells,
            "average": sum(c["score"] for c in cells.values()) / len(cells) if complete else None,
            "script_rate": _script_rate(model_runs),
            "drift_rate": _drift_rate(model_runs),
            "mcq_picks": dict(sorted(picks.items())),
        })
    rows.sort(key=lambda r: -(r["average"] if r["average"] is not None else -1))
    return rows


def _fmt(cell: dict | None, metric: str) -> str:
    if not cell:
        return "–"
    unit = "%" if metric == "accuracy" else ""
    lo, hi = cell["ci"]
    return f"{cell['score']:.1f}{unit} <sub>[{lo:.0f}–{hi:.0f}]</sub>"


def markdown(rows: list[dict]) -> str:
    head = "| Model | Average | " + " | ".join(f"{lab} ({m})" for lab, m in SUITES.values()) + " | Answered in Fidel | Drifted to other scripts |"
    sep = "|" + "---|" * (len(SUITES) + 4)
    lines = [head, sep]
    for r in rows:
        avg = f"**{r['average']:.1f}**" if r["average"] is not None else "–"
        cells = [_fmt(r["suites"].get(s), m) for s, (_, m) in SUITES.items()]
        script = f"{r['script_rate']:.0f}%" if r["script_rate"] is not None else "–"
        drift = f"{r['drift_rate']:.0f}%" if r["drift_rate"] is not None else "–"
        lines.append(f"| {r['model']} | {avg} | " + " | ".join(cells) + f" | {script} | {drift} |")
    return "\n".join(lines)


def html_page(rows: list[dict], title: str = "Amharic LLM leaderboard") -> str:
    e = html.escape
    head = "".join(f"<th>{e(lab)}<br><small>{m}</small></th>" for lab, m in SUITES.values())
    body = ""
    for r in rows:
        avg = f"<b>{r['average']:.1f}</b>" if r["average"] is not None else "–"
        cells = "".join(f"<td>{_fmt(r['suites'].get(s), m)}</td>" for s, (_, m) in SUITES.items())
        script = f"{r['script_rate']:.0f}%" if r["script_rate"] is not None else "–"
        drift = f"{r['drift_rate']:.0f}%" if r["drift_rate"] is not None else "–"
        picks = ", ".join(f"{k}:{v}" for k, v in r["mcq_picks"].items()) or "–"
        body += f"<tr><td>{e(r['model'])}</td><td>{avg}</td>{cells}<td>{script}</td><td>{drift}</td><td>{e(picks)}</td></tr>"
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{e(title)}</title><style>{_CSS}</style></head><body><main>
<h1>{e(title)}</h1>
<p class="sub">QA and MCQ are % correct; translation and summary are mean chrF (0–100). Brackets are 95% bootstrap intervals.</p>
<div class="wrap"><table><tr><th>Model</th><th>Average</th>{head}<th>Answered in Fidel</th><th>Drifted to other scripts</th><th>MCQ letters picked</th></tr>{body}</table></div>
<p class="sub">Per-item answers are in details.html next to this file.</p>
</main></body></html>"""


def load_runs(paths) -> list[dict]:
    return [json.loads(Path(p).read_text(encoding="utf-8")) for p in paths]
