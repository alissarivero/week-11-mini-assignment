# Week 11 — Airflow pipeline + agent-based ML

End-to-end homework: a **scheduled Airflow DAG** that joins two related datasets into PostgreSQL, a **modified in-class agent loop** that trains until a metric target is hit, and a **bonus LangGraph** workflow that routes, retries, and is scored by an evaluation suite.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/generate_data.py
python scripts/run_part1.py      # ETL + baseline model (SQLite if Postgres is down)
python scripts/run_part2.py      # agent training loop + live graph PNG
python scripts/run_part3.py      # adaptive graph, eval suite, baseline comparison
PYTHONPATH=src pytest tests -q
```

### Containerized stack (Airflow UI + Postgres + dashboard)

```bash
docker compose up --build -d
# Airflow:   http://localhost:8080   (admin / admin)
# Dashboard: http://localhost:8501
docker compose exec airflow airflow dags trigger bike_demand_etl
```

Dev Container: open this folder in Cursor/VS Code and **Reopen in Container**. The compose file and `.devcontainer/devcontainer.json` forward ports `8080`, `8501`, and `5432`.

---

## Part 1 — Airflow data pipeline

### Data sources

Two related tables covering 2023–2024 (generated, but structured like a real bike-share + weather join):

| Table | File | Grain | Fields |
| --- | --- | --- | --- |
| Station rentals | `data/raw/bike_rentals.csv` | station × day | `ride_date`, `station_id`, `station_name`, `rentals`, `casual_rides`, `member_rides` |
| City weather | `data/raw/weather.csv` | day | `date`, `temp_c`, `humidity`, `wind_kph`, `precip_mm`, `weather_code` |

Demand is constructed so weather and weekend effects are recoverable: warmer/drier days and weekends produce more trips. A few dirty values (negative rentals, humidity `140`, missing temperature, messy station names) are injected so the clean steps have real work to do.

### Transformations

- **Bikes:** parse dates, normalize station ids/names, drop negatives and duplicates.
- **Weather:** clip humidity, interpolate missing temperature, fill precipitation, lowercase weather codes.
- **Merge:** inner join on calendar date, add `is_weekend`, `month`, `day_of_week`.
- **Load:** replace table `bike_weather_daily` in PostgreSQL (`analytics` database). Local scripts fall back to SQLite at `data/analytics.db`.
- **Analyze:** time series + scatter + station bar charts; logistic regression classifying **high vs low demand** (rentals above the median).

### Pipeline structure

DAG `bike_demand_etl` (`dags/bike_demand_pipeline.py`):

```
ingest.ingest_bikes ──► clean_bikes.strip_and_validate ──┐
                                                         ├─► merge ─► load ─► analyze
ingest.ingest_weather ─► clean_weather.interpolate_and_clip ┘
```

Independent ingest/clean branches run **in parallel**. Related steps sit in **TaskGroups** (`ingest`, `clean_bikes`, `clean_weather`). Downstream merge/load/analyze stay sequential because each needs the previous artifact.

### Scheduling and parallelism

- `schedule="@daily"`, `catchup=False`, `start_date=2026-01-01`.
- `LocalExecutor` so the two ingest tasks (and the two clean TaskGroups) actually overlap on one worker machine.
- One retry with a two-minute delay on task failure.

### Baseline analysis results

After `python scripts/run_part1.py` the hold-out **logistic regression accuracy and F1** are written to `reports/baseline_metrics.json`. Charts land in `reports/daily_rentals.png`, `reports/rentals_vs_temp.png`, `reports/rentals_by_station.png`, and `reports/baseline_confusion_matrix.png`. Temperature, precipitation, and station identity are the strongest drivers of high-demand days.

---

## Part 2 — Agent-based ML workflow (for-loop template)

The in-class pattern was a linear **Loader → Train → Evaluate** pass. This repo keeps the coordinator as an explicit `for` loop (`src/agents/workflow.py`) but **adds/modifies agents**:

| Agent | Change vs in-class |
| --- | --- |
| DataLoader | Reads the **cleaned Postgres/SQLite table from Part 1**, not a raw CSV |
| FeatureEngineer | **New** — staged feature sets (temp only → weather/calendar → station + weather code) |
| Trainer | Fits logreg, then a random forest if the tuner asks |
| Evaluator | Reports **accuracy and F1** on a stratified hold-out split |
| Tuner | **New** — if accuracy `< target`, expand features or switch models |
| Reporter | **New** — writes `reports/agent_loop_metrics.json` |

The live graph is `apps/streamlit_app.py` (tab **Part 2**). Nodes light up as events fire; a Tuner→FeatureEngineer back-edge shows the loop. A static PNG is also exported to `reports/agent_graph.png`.

```bash
streamlit run apps/streamlit_app.py
python scripts/run_part2.py
```

Typical result: round 1 (humidity only) is barely better than chance; the tuner then adds wind, then temperature/rain/weekend, then the full station-aware set until accuracy crosses **0.88**. Exact numbers are in `reports/agent_loop_metrics.json` and `reports/agent_loop_accuracy.png`.

---

## Part 3 — Adaptive LangGraph + evaluation (bonus)

`src/adaptive/graph.py` compiles a **state graph** with distinct agents (`loader`, `quality`, `features`, `trainer`, `evaluator`, `router`, `retry`, `reporter`).

Adaptation at runtime:

- **Quality gate** drops residual nulls before training.
- **Router** reads intermediate accuracy and either **tunes** (more features / different model), **retries** a failed node (up to 3), or **finishes**.
- Failed evaluator/trainer steps **re-enter** after a repair instead of aborting the DAG.

### Evaluation suite

Every case in `src/adaptive/eval_suite.py` declares **input**, **expected behavior**, **scoring**, and a **pass/fail threshold**:

| ID | What it measures |
| --- | --- |
| T1_happy_path | Final accuracy ≥ 0.80 |
| T2_feature_expansion | Router-driven gain vs round 1 ≥ 0.03 |
| T3_retry_failed_step | Retry node re-enters a failed step |
| T4_coordination_order | Loader, evaluator, and reporter all fire |
| T5_intermediate_quality | Weak first model still ≥ 0.52 |
| T6_simple_agent_baseline | In-class single-pass agent still scores |
| T7_fixed_airflow_baseline | Fixed DAG analyze task ≥ 0.80 |

```bash
python scripts/run_part3.py
PYTHONPATH=src pytest tests -q
```

### Comparison to the two class baselines

| System | Shape | Adaptation | Typical accuracy |
| --- | --- | --- | --- |
| **Fixed Airflow DAG** | Predefined ingest→clean→merge→load→one logreg | None; failed tasks retry the same operator | Strong (full feature set once) |
| **Simple in-class agents** | Loader → Train → Evaluate, one loop, `humidity` only | None | Weaker; no feature expansion |
| **Adaptive LangGraph** | Graph with conditional edges | Selects next agent, retries, stops on target | Matches or beats the DAG with extra rounds only when needed |

The fixed DAG is the most **observable and deterministic** — every run has the same graph in the Airflow UI, which is why it is the right tool for daily ETL. The simple agent loop is easy to demo but **cannot react** to a bad first model. The LangGraph workflow is better when the **number of training rounds is unknown**, at the cost of a more dynamic (harder to audit) execution path. Charts: `reports/baseline_comparison.png`.

---

## Repository map

```
dags/bike_demand_pipeline.py   Airflow DAG (Part 1)
src/pipeline/                  ingest, clean, merge, load, analyze
src/agents/                    in-class loop + live graph helpers (Part 2)
src/adaptive/                  LangGraph, eval suite, baseline comparison (Part 3)
apps/streamlit_app.py          live agent graph + eval dashboard
tests/test_eval_suite.py
Dockerfile / Dockerfile.dashboard / docker-compose.yml
.devcontainer/devcontainer.json
reports/                       metrics, charts, screenshots
```

## What to submit

Upload this folder as a git repo (or a zip) with the code, Docker files, README, `data/raw/`, and `reports/screenshots/`. Do not upload `.venv/`.

Captured evidence (already in `reports/screenshots/`):

| File | What it shows |
| --- | --- |
| `01_airflow_successful_dag_run.png` | Two green DAG runs, all tasks success |
| `01_airflow_grid_and_graph.png` | Grid + Graph in the Airflow UI |
| `02_airflow_dag_graph.png` | DAG Graph tab (parallel ingest/clean → merge → load → analyze) |
| `03_agent_graph_live.png` | Live Streamlit agent graph |
| `04_model_evaluation_live.png` | Accuracy **0.881**, F1 **0.881**, 4 rounds, target met |
| `04b_baseline_confusion_matrix.png` | Part 1 logistic regression |
| `05_eval_suite.png` | Part 3 tests (7/7 passed) |
| `06_baseline_comparison.png` | Adaptive graph vs Airflow DAG vs simple in-class agents |

Airflow login: `http://localhost:8080` with **admin / admin**. Dashboard: `http://localhost:8501`.
