"""Writes review.csv: every Amharic item in one sheet for a native speaker to check.

Open it in Excel or Google Sheets, put "ok" or a corrected version in the last
columns, and the fixes go back into scripts/build_suites.py.
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_suites as b  # noqa: E402

out = Path(sys.argv[1] if len(sys.argv) > 1 else "review.csv")
with out.open("w", newline="", encoding="utf-8-sig") as f:  # BOM so Excel shows Fidel correctly
    w = csv.writer(f)
    w.writerow(["id", "type", "amharic", "answer / english", "ok? (y/n)", "correction or note"])
    n = 0
    for p, (passage, qs) in enumerate(b.QA, 1):
        w.writerow([f"passage-{p}", "QA passage", passage, "", "", ""])
        for q, answers, _ in qs:
            n += 1
            w.writerow([f"qa-{n:02d}", "QA question", q, " | ".join(answers), "", ""])
    for i, (q, right, wrong, _) in enumerate(b.MCQ, 1):
        w.writerow([f"mcq-{i:02d}", "multiple choice", q, f"RIGHT: {right} | wrong: {' / '.join(wrong)}", "", ""])
    for i, (en, ams, _) in enumerate(b.PAIRS, 1):
        w.writerow([f"pair-{i:02d}", "translation", " | ".join(ams), en, "", ""])
    for i, (passage, ref, _) in enumerate(b.SUMMARIES, 1):
        w.writerow([f"sum-{i:02d}", "summary passage", passage, "", "", ""])
        w.writerow([f"sum-{i:02d}-ref", "reference summary", ref, "", "", ""])
print(f"wrote {out}")
