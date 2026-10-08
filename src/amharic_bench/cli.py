"""Command line: `amharic-bench run`, `amharic-bench leaderboard`, `amharic-bench list`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from evalforge.cli import _slug
from evalforge.models import get_model
from evalforge.report import html_report
from evalforge.runner import run_suite

from . import leaderboard
from .normalize import normalize
from .suites import SUITES, load


def _write_leaderboard(paths: list[Path], out_dir: Path) -> str:
    runs = leaderboard.load_runs(paths)
    rows = leaderboard.build(runs)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "leaderboard.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "leaderboard.html").write_text(leaderboard.html_page(rows), encoding="utf-8")
    (out_dir / "details.html").write_text(html_report(runs, "amharic-llm-bench · every answer"), encoding="utf-8")
    md = leaderboard.markdown(rows)
    (out_dir / "leaderboard.md").write_text(md + "\n", encoding="utf-8")
    return md


def cmd_run(args) -> int:
    names = args.suite or list(SUITES)
    out_dir = Path(args.out)
    saved = sorted(out_dir.glob("*__*.json"))  # keep earlier runs on the leaderboard
    for spec in args.model:
        model = get_model(spec)
        for name in names:
            suite = load(name)
            run = run_suite(suite, model, workers=args.workers, limit=args.limit,
                            cache_path=out_dir / ".cache.json")
            path = run.save(out_dir / f"{name}__{_slug(model.name)}.json")
            if path not in saved:
                saved.append(path)
            vals = leaderboard.item_scores(run.to_dict())
            print(f"{model.name:<28} {name:<8} {sum(vals) / len(vals):5.1f}  ({len(vals)} items, {run.summary()['errors']} errors)", flush=True)
    print("\n" + _write_leaderboard(saved, out_dir))
    print(f"\nLeaderboard: {out_dir / 'leaderboard.html'}")
    return 0


def cmd_leaderboard(args) -> int:
    print(_write_leaderboard([Path(p) for p in args.results], Path(args.out)))
    return 0


def cmd_list(args) -> int:
    for name, (label, metric) in SUITES.items():
        print(f"  {name:<8} {label:<8} {metric:<9} {len(load(name).items)} items")
    return 0


def cmd_normalize(args) -> int:
    print(normalize(args.text))
    return 0


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to a code page without Fidel
    p = argparse.ArgumentParser(prog="amharic-bench", description="Benchmark LLMs on Amharic tasks.")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run the suites against one or more models")
    r.add_argument("-m", "--model", action="append", required=True, help="provider:model, repeatable (e.g. ollama:qwen2.5:3b)")
    r.add_argument("-s", "--suite", action="append", choices=list(SUITES), help="only these suites (default: all)")
    r.add_argument("-o", "--out", default="results", help="output folder (default: results)")
    r.add_argument("-w", "--workers", type=int, default=2)
    r.add_argument("-n", "--limit", type=int, help="only the first N items of each suite")
    r.set_defaults(fn=cmd_run)

    lb = sub.add_parser("leaderboard", help="rebuild the leaderboard from saved result files")
    lb.add_argument("results", nargs="+")
    lb.add_argument("-o", "--out", default="results")
    lb.set_defaults(fn=cmd_leaderboard)

    sub.add_parser("list", help="show the suites").set_defaults(fn=cmd_list)

    nz = sub.add_parser("normalize", help="show the normalized form of some Amharic text")
    nz.add_argument("text")
    nz.set_defaults(fn=cmd_normalize)

    args = p.parse_args(argv)
    try:
        return args.fn(args)
    except (ValueError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
