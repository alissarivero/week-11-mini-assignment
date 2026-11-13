from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adaptive.eval_suite import all_cases, run_suite  # noqa: E402
from pipeline.ingest import ingest_bikes, ingest_weather  # noqa: E402
from pipeline.load import load_merged  # noqa: E402
from pipeline.transform import clean_bikes, clean_weather, merge_datasets  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def prepared_db():
    bike = ingest_bikes()
    weather = ingest_weather()
    merged = merge_datasets(clean_bikes(bike), clean_weather(weather))
    load_merged(merged)


def test_every_case_declares_input_expected_scoring_and_threshold():
    for case in all_cases():
        assert case.input
        assert case.expected
        assert case.scoring
        assert case.threshold is not None


def test_suite_meets_pass_rate():
    results = run_suite()
    assert results, "suite produced no cases"
    pass_rate = sum(r.passed for r in results) / len(results)
    assert pass_rate >= 0.85, [r for r in results if not r.passed]
