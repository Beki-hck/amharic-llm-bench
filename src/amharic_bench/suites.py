"""The built-in Amharic suites and what each one measures."""

from __future__ import annotations

from importlib import resources

from evalforge.suite import Suite, load_suite

from . import scorers  # noqa: F401  make sure the scorers are registered before loading

# name -> (column label, what the score means)
SUITES: dict[str, tuple[str, str]] = {
    "am_qa": ("QA", "accuracy"),
    "am_mcq": ("MCQ", "accuracy"),
    "en_am": ("EN→AM", "chrF"),
    "am_en": ("AM→EN", "chrF"),
    "am_sum": ("Summary", "chrF"),
}


def suite_path(name: str):
    path = resources.files("amharic_bench") / "suites" / f"{name}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"No built-in suite {name!r}. Available: {', '.join(SUITES)}")
    return path


def load(name: str) -> Suite:
    with resources.as_file(suite_path(name)) as p:
        suite = load_suite(p)
    suite.name = name
    return suite
