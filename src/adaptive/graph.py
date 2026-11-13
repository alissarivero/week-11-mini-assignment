"""Adaptive multi-agent graph: route, retry failed steps, and stop on quality."""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from langgraph.graph import END, StateGraph

from agents.ml_agents import evaluate_model, load_data, select_features, train_model, tune


class AdaptiveState(TypedDict, total=False):
    uri: str | None
    target_accuracy: float
    max_rounds: int
    round: int
    stage: int
    model_name: str
    retries: int
    last_error: str
    failed_node: str
    columns: list[str]
    metrics: dict[str, Any]
    history: list[dict[str, Any]]
    route: str
    n_rows: int
    quality_issues: list[str]


def _ensure(state: AdaptiveState) -> AdaptiveState:
    state.setdefault("round", 0)
    state.setdefault("stage", 0)
    state.setdefault("model_name", "logreg")
    state.setdefault("retries", 0)
    state.setdefault("history", [])
    state.setdefault("target_accuracy", 0.88)
    state.setdefault("max_rounds", 5)
    state.setdefault("quality_issues", [])
    return state


def loader_node(state: AdaptiveState) -> AdaptiveState:
    state = _ensure(state)
    try:
        df = load_data(state.get("uri"))
        issues = []
        if df["rentals"].isna().any():
            issues.append("null_rentals")
        if len(df) < 50:
            issues.append("too_few_rows")
        state["n_rows"] = int(len(df))
        state["quality_issues"] = issues
        state["last_error"] = ""
        state["failed_node"] = ""
        # stash frame on a module-level cache keyed by this run
        _CACHE["df"] = df
    except Exception as exc:  # noqa: BLE001
        state["last_error"] = str(exc)
        state["failed_node"] = "loader"
        state["retries"] = int(state.get("retries", 0)) + 1
    return state


def quality_node(state: AdaptiveState) -> AdaptiveState:
    if state.get("quality_issues"):
        df = _CACHE.get("df")
        if df is not None:
            df = df.dropna(subset=["rentals", "temp_c"])
            _CACHE["df"] = df
            state["n_rows"] = int(len(df))
            state["quality_issues"] = []
    return state


def features_node(state: AdaptiveState) -> AdaptiveState:
    state = _ensure(state)
    try:
        feat = select_features(_CACHE["df"], state["stage"])
        state["columns"] = feat["columns"]
        state["last_error"] = ""
        state["failed_node"] = ""
    except Exception as exc:  # noqa: BLE001
        state["last_error"] = str(exc)
        state["failed_node"] = "features"
        state["retries"] = int(state.get("retries", 0)) + 1
    return state


def trainer_node(state: AdaptiveState) -> AdaptiveState:
    state = _ensure(state)
    try:
        bundle = train_model(_CACHE["df"], state["columns"], state["model_name"])
        _CACHE["bundle"] = bundle
        state["last_error"] = ""
        state["failed_node"] = ""
    except Exception as exc:  # noqa: BLE001
        state["last_error"] = str(exc)
        state["failed_node"] = "trainer"
        state["retries"] = int(state.get("retries", 0)) + 1
    return state


def evaluator_node(state: AdaptiveState) -> AdaptiveState:
    state = _ensure(state)
    try:
        metrics = evaluate_model(_CACHE["bundle"])
        state["metrics"] = metrics
        state["round"] = int(state.get("round", 0)) + 1
        state["history"] = list(state.get("history", [])) + [
            {
                "round": state["round"],
                "stage": state["stage"],
                "model_name": state["model_name"],
                "accuracy": metrics["accuracy"],
                "f1": metrics["f1"],
            }
        ]
        state["last_error"] = ""
        state["failed_node"] = ""
    except Exception as exc:  # noqa: BLE001
        state["last_error"] = str(exc)
        state["failed_node"] = "evaluator"
        state["retries"] = int(state.get("retries", 0)) + 1
    return state


def router_node(state: AdaptiveState) -> AdaptiveState:
    """Choose retry, tune, or finish from intermediate results."""
    state = _ensure(state)
    if state.get("last_error"):
        state["route"] = "retry" if state["retries"] <= 3 else "fail"
        return state
    metrics = state.get("metrics") or {}
    acc = float(metrics.get("accuracy", 0))
    if acc >= state["target_accuracy"] or state["round"] >= state["max_rounds"]:
        state["route"] = "finish"
        return state
    decision = tune(state["stage"], metrics, state["target_accuracy"])
    state["stage"] = decision["stage"]
    state["model_name"] = decision["model_name"]
    state["route"] = "tune"
    return state


def retry_node(state: AdaptiveState) -> AdaptiveState:
    """Re-enter the failed node after a cheap repair."""
    failed = state.get("failed_node")
    if failed == "loader":
        return loader_node(state)
    if failed == "features":
        state["stage"] = 0
        return features_node(state)
    if failed == "trainer":
        state["model_name"] = "logreg"
        return trainer_node(state)
    if "bundle" not in _CACHE:
        trainer_node(state)
    return evaluator_node(state)


def reporter_node(state: AdaptiveState) -> AdaptiveState:
    metrics = state.get("metrics") or {}
    state["route"] = "done"
    state["met_target"] = bool(metrics.get("accuracy", 0) >= state.get("target_accuracy", 0.82))
    return state


def _after_router(state: AdaptiveState) -> Literal["retry", "features", "reporter", "fail"]:
    route = state.get("route")
    if route == "retry":
        return "retry"
    if route == "tune":
        return "features"
    if route == "fail":
        return "fail"
    return "reporter"


_CACHE: dict[str, Any] = {}


def build_adaptive_graph():
    workflow = StateGraph(AdaptiveState)
    workflow.add_node("loader", loader_node)
    workflow.add_node("quality", quality_node)
    workflow.add_node("features", features_node)
    workflow.add_node("trainer", trainer_node)
    workflow.add_node("evaluator", evaluator_node)
    workflow.add_node("router", router_node)
    workflow.add_node("retry", retry_node)
    workflow.add_node("reporter", reporter_node)

    workflow.set_entry_point("loader")
    workflow.add_edge("loader", "quality")
    workflow.add_edge("quality", "features")
    workflow.add_edge("features", "trainer")
    workflow.add_edge("trainer", "evaluator")
    workflow.add_edge("evaluator", "router")
    workflow.add_conditional_edges(
        "router",
        _after_router,
        {
            "retry": "retry",
            "features": "features",
            "reporter": "reporter",
            "fail": "reporter",
        },
    )
    workflow.add_edge("retry", "router")
    workflow.add_edge("reporter", END)
    return workflow.compile()


def run_adaptive(uri: str | None = None, **kwargs: Any) -> AdaptiveState:
    graph = build_adaptive_graph()
    state: AdaptiveState = {"uri": uri, **kwargs}
    return graph.invoke(state)


def graph_mermaid() -> str:
    return build_adaptive_graph().get_graph().draw_mermaid()
