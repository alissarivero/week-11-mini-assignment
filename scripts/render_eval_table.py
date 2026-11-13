"""PNG of evaluation-suite pass/fail for the submission screenshots."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pipeline.config import REPORTS_DIR, ensure_dirs  # noqa: E402


def main() -> None:
    ensure_dirs()
    payload = json.loads((REPORTS_DIR / "eval_suite.json").read_text())
    fig, ax = plt.subplots(figsize=(10, 3.6 + 0.28 * len(payload)))
    ax.axis("off")
    rows = [
        [
            r["case_id"],
            "PASS" if r["passed"] else "FAIL",
            f"{r['score']:.3f}",
            str(r["threshold"]),
        ]
        for r in payload
    ]
    table = ax.table(
        cellText=rows,
        colLabels=["case", "result", "score", "threshold"],
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.4)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#1d3557")
            cell.set_text_props(color="white")
        elif col == 1:
            cell.set_facecolor("#d8f3dc" if rows[row - 1][1] == "PASS" else "#f8d7da")
    ax.set_title("Part 3 evaluation suite")
    fig.tight_layout()
    dest = REPORTS_DIR / "eval_suite.png"
    fig.savefig(dest, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(dest)


if __name__ == "__main__":
    main()
