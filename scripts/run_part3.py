"""Run the adaptive graph, evaluation suite, and baseline comparison."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adaptive.compare import compare  # noqa: E402
from adaptive.eval_suite import run_suite  # noqa: E402
from adaptive.graph import graph_mermaid, run_adaptive  # noqa: E402
from pipeline.config import REPORTS_DIR, ensure_dirs  # noqa: E402


def main() -> None:
    ensure_dirs()
    result = run_adaptive(target_accuracy=0.82, max_rounds=5)
    mermaid = graph_mermaid()
    (REPORTS_DIR / "adaptive_graph.mmd").write_text(mermaid)
    print("Adaptive accuracy:", result.get("metrics", {}).get("accuracy"))
    print(mermaid)

    suite = run_suite()
    payload = [
        {
            "case_id": r.case_id,
            "passed": r.passed,
            "score": r.score,
            "threshold": r.threshold,
            "expected": r.expected,
            "observed": r.observed,
            **r.details,
        }
        for r in suite
    ]
    (REPORTS_DIR / "eval_suite.json").write_text(json.dumps(payload, indent=2))
    passed = sum(r.passed for r in suite)
    print(f"Eval suite {passed}/{len(suite)} passed")
    for row in payload:
        flag = "PASS" if row["passed"] else "FAIL"
        print(f"  [{flag}] {row['case_id']}: {row['observed']}")

    cmp = compare()
    print(cmp.to_string(index=False))


if __name__ == "__main__":
    main()
