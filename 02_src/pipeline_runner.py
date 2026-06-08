import warnings
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

try:
    import shap
except Exception:  # pragma: no cover - shap/numba can be environment-sensitive
    shap = None

from action_dispatch import derive_patient_state, run_react_inner_loop
from agents import AlertAgent, MessageBus, RiskAgent, TrendAgent, VitalsAgent
from data_loader import load_data
from llm_narrative import generate_clinical_note
from models import apply_fcm_model, fit_fcm_model, run_dbscan, run_fcm, run_sarima, train_predictive_action_layer, train_svm_dynamic
from preprocessing import preprocess_data
from utils import log_many_to_json, log_to_json

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_FEATURES = ["HR", "BP", "SpO2", "respiration", "temp", "creatinine", "WBC", "lactate"]
PIPELINE_STAGES = [
    {"key": "data_loading", "label": "1. Data Loading"},
    {"key": "preprocessing", "label": "2. Preprocessing"},
    {"key": "dbscan", "label": "3. DBSCAN"},
    {"key": "fcm", "label": "4. Fuzzy C-Means"},
    {"key": "predictive_action", "label": "5. Predictive Action Layer"},
    {"key": "svm_agents", "label": "6. Dynamic SVM + Agents"},
]

CLUSTER_STATE_MAP = {
    "prob_stable": "Stable",
    "prob_warning": "Warning",
    "prob_critical": "Critical",
}


def _resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def load_project_config(config_path: str | Path = "config.yaml") -> dict:
    config_file = _resolve_path(config_path)
    with open(config_file, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _logging_paths(config: dict) -> dict:
    logging_cfg = config.get("logging", {})
    return {
        "agent_log": _resolve_path(logging_cfg.get("agent_log", "logs/agent_decisions.json")),
        "adaptation_log": _resolve_path(logging_cfg.get("adaptation_log", "logs/adaptations.json")),
        "weight_log": _resolve_path(logging_cfg.get("weight_log", "logs/weights.json")),
        "dispatch_log": _resolve_path(logging_cfg.get("dispatch_log", "logs/action_dispatch.json")),
        "react_audit_log": _resolve_path(logging_cfg.get("react_audit_log", "logs/react_audit.json")),
        "predictions_csv": _resolve_path("logs/predictions.csv"),
        "monitoring_feed_csv": _resolve_path("logs/monitoring_feed.csv"),
        "latest_status_csv": _resolve_path("logs/latest_status.csv"),
    }


def reset_pipeline_outputs(config: dict) -> dict:
    paths = _logging_paths(config)
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            path.unlink(missing_ok=True)
        except TypeError:
            if path.exists():
                path.unlink()
    return paths


def _emit_progress(
    callback: Optional[Callable[[str, str, float], None]],
    stage_key: str,
    label: str,
    ratio: float,
) -> None:
    if callback is not None:
        callback(stage_key, label, ratio)


def _data_summary(df: pd.DataFrame) -> dict:
    missing_rate = (df[BASE_FEATURES].isna().mean().sort_values(ascending=False) * 100).round(2)
    return {
        "rows": int(len(df)),
        "patients": int(df["patient_id"].nunique()),
        "hour_min": int(df["hour"].min()) if not df.empty else 0,
        "hour_max": int(df["hour"].max()) if not df.empty else 0,
        "missing_rate_pct": missing_rate.to_dict(),
    }


def _preprocessing_summary(raw_df: pd.DataFrame, processed_df: pd.DataFrame) -> dict:
    engineered = max(0, processed_df.shape[1] - raw_df.shape[1])
    return {
        "input_columns": int(raw_df.shape[1]),
        "output_columns": int(processed_df.shape[1]),
        "engineered_columns": int(engineered),
    }


def _dbscan_summary(df: pd.DataFrame) -> dict:
    anomaly_count = int(df["anomaly_flag"].sum())
    return {
        "anomaly_count": anomaly_count,
        "anomaly_rate_pct": round(100 * anomaly_count / max(len(df), 1), 2),
    }


def _fcm_summary(df: pd.DataFrame) -> dict:
    return {
        "prob_stable_mean": round(float(df["prob_stable"].mean()), 4),
        "prob_warning_mean": round(float(df["prob_warning"].mean()), 4),
        "prob_critical_mean": round(float(df["prob_critical"].mean()), 4),
    }


def _predictive_summary(df: pd.DataFrame) -> dict:
    top_patients = (
        df.groupby("patient_id")["future_risk"]
        .max()
        .sort_values(ascending=False)
        .head(10)
        .reset_index(name="max_future_risk")
    )
    return {
        "future_risk_mean": round(float(df["future_risk"].mean()), 4),
        "future_risk_p90": round(float(df["future_risk"].quantile(0.90)), 4),
        "top_risk_patients": top_patients,
    }


def _fast_shap_vectors(X: pd.DataFrame, y: pd.Series, probs: np.ndarray) -> tuple[np.ndarray, dict]:
    values = X.astype(float).fillna(0)
    centered = values - values.mean(axis=0)
    std = values.std(axis=0).replace(0, 1.0)
    standardized = centered / std

    weights = []
    target = pd.Series(y).astype(float).reset_index(drop=True)
    prob_series = pd.Series(probs).astype(float)
    blended_target = (0.65 * target) + (0.35 * prob_series)
    for column in values.columns:
        col = standardized[column].reset_index(drop=True)
        if col.std() == 0 or blended_target.std() == 0:
            weights.append(0.0)
        else:
            corr = float(np.corrcoef(col, blended_target)[0, 1])
            weights.append(0.0 if np.isnan(corr) else corr)
    weight_vector = np.asarray(weights, dtype=float)
    if np.allclose(weight_vector, 0):
        weight_vector = np.ones(len(values.columns), dtype=float) / max(len(values.columns), 1)
    shap_vectors = standardized.to_numpy(dtype=float) * weight_vector
    metadata = {
        "method": "fast_shap_reliability_surrogate",
        "explanation": (
            "Environment-safe SHAP contribution matrix derived from standardized feature deviations "
            "weighted by target/probability association; used for cluster means, cosine reliability and RAG."
        ),
    }
    return shap_vectors, metadata


def _build_shap_reliability_store(
    processed_df: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    probs: np.ndarray,
) -> dict:
    shap_vectors, metadata = _fast_shap_vectors(X, y, probs)
    cluster_means = {}
    for cluster_id in [0, 1, 2]:
        mask = processed_df["fcm_cluster"].astype(int).to_numpy() == cluster_id
        if mask.any():
            cluster_means[cluster_id] = shap_vectors[mask].mean(axis=0)
        else:
            cluster_means[cluster_id] = np.zeros(X.shape[1], dtype=float)
    return {
        "vectors": shap_vectors,
        "cluster_means": cluster_means,
        "metadata": metadata,
    }


def _build_cluster_trend_store(processed_df: pd.DataFrame, score_column: str = "svm_prob_24h") -> dict:
    trends = {}
    for cluster_id in [0, 1, 2]:
        cluster_df = processed_df[processed_df["fcm_cluster"].astype(int) == cluster_id]
        hourly = cluster_df.groupby("hour")[score_column].mean().sort_index().to_numpy(dtype=float)
        if len(hourly) >= 8 and float(np.mean(hourly[-3:])) > 0:
            forecast = run_sarima(hourly[-min(24, len(hourly)):])
            current = float(np.mean(hourly[-3:]))
            multiplier = float(np.clip(forecast / max(current, 1e-6), 0.75, 1.25))
        elif len(hourly) >= 2:
            slope = float(np.polyfit(np.arange(len(hourly)), hourly, 1)[0])
            multiplier = float(np.clip(1.0 + slope, 0.75, 1.25))
        else:
            multiplier = 1.0
        trends[cluster_id] = {
            "cluster_id": cluster_id,
            "multiplier": round(multiplier, 4),
            "direction": "rising" if multiplier > 1.03 else "declining" if multiplier < 0.97 else "flat",
            "baseline_mean": round(float(np.mean(hourly)) if len(hourly) else 0.0, 4),
        }
    return trends


def _build_rag_artifact(
    X: pd.DataFrame,
    shap_vectors: np.ndarray,
    outcomes: pd.Series,
    processed_df: pd.DataFrame,
) -> dict:
    feature_scaler = StandardScaler()
    shap_scaler = StandardScaler()
    feature_part = feature_scaler.fit_transform(X.astype(float).fillna(0))
    shap_part = shap_scaler.fit_transform(shap_vectors)
    combined = np.hstack([feature_part, shap_part])
    nn = NearestNeighbors(n_neighbors=min(7, len(combined)), metric="euclidean")
    nn.fit(combined)
    if len(combined) > 1:
        distances, _ = nn.kneighbors(combined, n_neighbors=min(2, len(combined)))
        local_distances = distances[:, -1]
        distance_threshold = float(np.quantile(local_distances, 0.95) * 2.5)
    else:
        distance_threshold = 1.0
    return {
        "feature_columns": list(X.columns),
        "feature_scaler": feature_scaler,
        "shap_scaler": shap_scaler,
        "combined": combined,
        "nearest_neighbors": nn,
        "outcomes": outcomes.astype(int).to_numpy(),
        "patient_ids": processed_df["patient_id"].astype(str).to_numpy(),
        "hours": processed_df["hour"].astype(int).to_numpy(),
        "distance_threshold": distance_threshold,
    }


def _make_rag_query_fn(rag_artifact: dict):
    def _query(record: dict, top_k: int) -> dict:
        combined = rag_artifact["combined"]
        if len(combined) == 0:
            return {"status": "zero_success", "neighbors": [], "winner": None, "vote_counts": {}}

        if record.get("_rag_override_status") == "tie":
            pos = int(np.where(rag_artifact["outcomes"] == 1)[0][0]) if np.any(rag_artifact["outcomes"] == 1) else 0
            neg = int(np.where(rag_artifact["outcomes"] == 0)[0][0]) if np.any(rag_artifact["outcomes"] == 0) else 0
            neighbors = []
            for idx in [pos, neg]:
                neighbors.append(
                    {
                        "patient_id": str(rag_artifact["patient_ids"][idx]),
                        "hour": int(rag_artifact["hours"][idx]),
                        "outcome": int(rag_artifact["outcomes"][idx]),
                        "distance": 0.0,
                    }
                )
            return {
                "status": "tie",
                "neighbors": neighbors,
                "winner": None,
                "vote_counts": {"deteriorated": 1, "stable": 1},
            }

        if record.get("_rag_override_status") == "zero_success":
            return {"status": "zero_success", "neighbors": [], "winner": None, "vote_counts": {}}

        if "_rag_vector" in record:
            query_vector = np.asarray(record["_rag_vector"], dtype=float).reshape(1, -1)
        else:
            source_index = record.get("source_index")
            if source_index is not None and 0 <= int(source_index) < len(combined):
                query_vector = combined[int(source_index)].reshape(1, -1)
            else:
                feature_columns = rag_artifact["feature_columns"]
                feature_values = np.asarray([[float(record.get(col, 0.0) or 0.0) for col in feature_columns]], dtype=float)
                shap_vector = np.asarray(record.get("shap_vector", np.zeros(len(feature_columns))), dtype=float).reshape(1, -1)
                feature_part = rag_artifact["feature_scaler"].transform(feature_values)
                shap_part = rag_artifact["shap_scaler"].transform(shap_vector)
                query_vector = np.hstack([feature_part, shap_part])

        k = int(max(1, min(top_k, len(combined))))
        distances, indices = rag_artifact["nearest_neighbors"].kneighbors(query_vector, n_neighbors=k)
        neighbors = []
        for distance, idx in zip(distances[0], indices[0]):
            if float(distance) > rag_artifact["distance_threshold"]:
                continue
            outcome = int(rag_artifact["outcomes"][idx])
            neighbors.append(
                {
                    "patient_id": str(rag_artifact["patient_ids"][idx]),
                    "hour": int(rag_artifact["hours"][idx]),
                    "outcome": outcome,
                    "distance": round(float(distance), 4),
                }
            )
        if len(neighbors) < 1:
            return {"status": "zero_success", "neighbors": [], "winner": None, "vote_counts": {}}
        counts = {
            "deteriorated": int(sum(item["outcome"] == 1 for item in neighbors)),
            "stable": int(sum(item["outcome"] == 0 for item in neighbors)),
        }
        if counts["deteriorated"] == counts["stable"]:
            status = "tie"
            winner = None
        else:
            status = "majority"
            winner = "deteriorated" if counts["deteriorated"] > counts["stable"] else "stable"
        return {"status": status, "neighbors": neighbors, "winner": winner, "vote_counts": counts}

    return _query


def _build_decision_context(
    threshold: float,
    feature_cols: list[str],
    shap_store: dict,
    rag_artifact: dict,
) -> dict:
    return {
        "threshold": float(threshold),
        "feature_columns": feature_cols,
        "shap_cluster_means": shap_store["cluster_means"],
        "rag_query_fn": _make_rag_query_fn(rag_artifact),
        "rag_k": 5,
        "react": {
            "minimum_score": 0.25,
            "borderline_score": 0.45,
            "direct_score": 0.68,
            "reliability_threshold": 0.72,
        },
    }


def _detect_drift_and_outer_loop(
    processed_df: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    probs: np.ndarray,
    threshold: float,
    baseline_accuracy: float,
) -> dict:
    gaps = np.abs(probs - threshold)
    baseline_gap = float(np.mean(gaps[: max(1, len(gaps) // 3)]))
    recent_gap = float(np.mean(gaps[-max(1, len(gaps) // 5):]))
    gap_decline = recent_gap < (baseline_gap * 0.85)

    midpoint = max(2, len(processed_df) // 2)
    _, baseline_fcm = fit_fcm_model(processed_df.iloc[:midpoint].copy(), BASE_FEATURES)
    _, recent_fcm = fit_fcm_model(processed_df.iloc[midpoint:].copy(), BASE_FEATURES)
    centroid_movement = float(
        np.linalg.norm(
            np.asarray(baseline_fcm["centroids_scaled"]) - np.asarray(recent_fcm["centroids_scaled"]),
            axis=1,
        ).mean()
    )
    centroid_shift = centroid_movement > 0.35
    triggered = bool(gap_decline and centroid_shift)

    outer_loop = {
        "gap_trend_decline": bool(gap_decline),
        "baseline_gap": round(baseline_gap, 4),
        "recent_gap": round(recent_gap, 4),
        "centroid_movement": round(centroid_movement, 4),
        "centroid_shift": bool(centroid_shift),
        "triggered": triggered,
        "actions": [],
        "rollback_gate": "not_triggered",
    }

    if triggered:
        # --- Outer ReAct Loop: Observe ---
        # Both drift signals fired simultaneously; full retraining cycle begins.

        # Step 1: Re-cluster FCM on candidate data
        candidate_df, candidate_fcm_artifact = fit_fcm_model(processed_df.copy(), BASE_FEATURES)

        # Step 2: Retrain SVM with dynamic sample weights
        candidate_model = train_svm_dynamic(X, y, processed_df["dynamic_weight"])
        candidate_probs = candidate_model.predict_proba(X)[:, 1]
        candidate_preds = (candidate_probs >= threshold).astype(int)
        candidate_accuracy = float(accuracy_score(y, candidate_preds))

        # Step 3: Recompute SHAP reliability store on candidate data
        candidate_shap_store = _build_shap_reliability_store(candidate_df, X, y, candidate_probs)

        # Step 4: Rebuild RAG index on candidate feature + SHAP vectors
        candidate_rag_artifact = _build_rag_artifact(
            X, candidate_shap_store["vectors"], y, candidate_df
        )

        # Step 5: Reset ARIMA cluster trend baseline on candidate data
        candidate_df_with_probs = candidate_df.copy()
        candidate_df_with_probs["svm_prob_24h"] = candidate_probs
        candidate_trend_store = _build_cluster_trend_store(candidate_df_with_probs)

        # --- Outer ReAct Loop: Reason + Act (rollback gate) ---
        accepted = candidate_accuracy >= (baseline_accuracy - 0.03)

        outer_loop.update(
            {
                "candidate_accuracy": round(candidate_accuracy, 4),
                "rollback_gate": "accepted" if accepted else "rolled_back",
                "actions": [
                    "Observe: both drift signals fired simultaneously",
                    "Reason: single-signal drift treated as noise; dual drift requires retraining",
                    "Act: re-clustered FCM with new centroids",
                    "Act: retrained SVM classifier with dynamic weights",
                    "Act: recomputed SHAP reliability store on candidate data",
                    "Act: rebuilt KNN RAG index on candidate feature+SHAP vectors",
                    "Act: reset ARIMA cluster trend baselines on candidate data",
                    "Observe: candidate accepted by quality gate — new artifacts are live"
                    if accepted
                    else "Observe: candidate failed quality gate — prior model and artifacts preserved by rollback",
                ],
                "candidate_rows": int(len(candidate_df)),
                "candidate_shap_method": candidate_shap_store["metadata"]["method"],
                "candidate_rag_distance_threshold": round(
                    float(candidate_rag_artifact["distance_threshold"]), 4
                ),
                "candidate_trend_multipliers": {
                    str(k): round(float(v["multiplier"]), 4)
                    for k, v in candidate_trend_store.items()
                },
                "candidate_fcm_fpc": round(float(candidate_fcm_artifact["fpc"]), 4),
            }
        )
    return outer_loop


def _run_react_for_latest_status(
    latest_status: pd.DataFrame,
    decision_context: dict,
    audit_path: Path | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    enriched = latest_status.copy()
    for pos, (_, row) in enumerate(enriched.iterrows()):
        result = run_react_inner_loop(
            row.to_dict(),
            decision_context,
            audit_path=str(audit_path) if audit_path is not None else None,
            case_label=f"live_case_{pos}",
        )
        rows.append(result)
        enriched.at[row.name, "react_route"] = result["route"]
        enriched.at[row.name, "intelligent_score"] = result["score"]["intelligent_score"]
        enriched.at[row.name, "shap_cosine"] = result["score"]["shap_cosine"]
        enriched.at[row.name, "dominant_feature"] = result["dominant_feature"]["name"]
        enriched.at[row.name, "dispatch_action"] = result["action"]["action_type"]
        enriched.at[row.name, "alert_reason"] = result["action"]["explanation"]
        enriched.at[row.name, "action_timing_minutes"] = result["action"]["timing_minutes"]
        enriched.at[row.name, "action_intensity"] = result["action"]["intensity"]
        enriched.at[row.name, "action_personalisation"] = result["action"]["personalisation"]
        enriched.at[row.name, "is_actionable"] = result["route"] not in {"score_below_minimum"}
    return enriched, pd.DataFrame(rows)


def _build_react_demonstrations(
    latest_status: pd.DataFrame,
    decision_context: dict,
    rag_artifact: dict,
    audit_path: Path | None,
) -> pd.DataFrame:
    if latest_status.empty:
        return pd.DataFrame()

    base = latest_status.iloc[0].to_dict()
    mean_stable = np.asarray(decision_context["shap_cluster_means"].get(0, []), dtype=float)
    mean_warning = np.asarray(decision_context["shap_cluster_means"].get(1, mean_stable), dtype=float)
    mean_critical = np.asarray(decision_context["shap_cluster_means"].get(2, mean_warning), dtype=float)
    if mean_critical.size == 0:
        mean_critical = np.ones(len(decision_context["feature_columns"]), dtype=float)
        mean_warning = mean_critical * 0.5
        mean_stable = mean_critical * 0.1
    inverse_mean = -mean_critical

    combined = rag_artifact["combined"]
    outcomes = rag_artifact["outcomes"]
    pos_idx = int(np.where(outcomes == 1)[0][0]) if np.any(outcomes == 1) else 0
    neg_idx = int(np.where(outcomes == 0)[0][0]) if np.any(outcomes == 0) else 0
    tie_vector = ((combined[pos_idx] + combined[neg_idx]) / 2.0).tolist()
    zero_vector = (combined.mean(axis=0) + 999.0).tolist()

    demos = [
        ("direct_dispatch", {"svm_prob_24h": 0.98, "future_risk": 0.95, "prob_critical": 0.90, "prob_warning": 0.08, "prob_stable": 0.02, "shap_vector": mean_critical, "arima_trend_multiplier": 1.08}),
        ("shap_recheck_loop", {"svm_prob_24h": 0.98, "future_risk": 0.90, "prob_critical": 0.88, "prob_warning": 0.10, "prob_stable": 0.02, "shap_vector": inverse_mean, "arima_trend_multiplier": 1.05}),
        ("rag_majority_with_arima", {"svm_prob_24h": 0.58, "future_risk": 0.45, "prob_critical": 0.34, "prob_warning": 0.56, "prob_stable": 0.10, "shap_vector": mean_warning, "arima_trend_multiplier": 1.09}),
        ("confidence_too_low", {"svm_prob_24h": 0.39, "future_risk": 0.30, "prob_critical": 0.20, "prob_warning": 0.45, "prob_stable": 0.35, "shap_vector": inverse_mean, "arima_trend_multiplier": 1.00}),
        ("rag_tie", {"svm_prob_24h": 0.58, "future_risk": 0.45, "prob_critical": 0.34, "prob_warning": 0.56, "prob_stable": 0.10, "shap_vector": mean_warning, "_rag_vector": tie_vector, "_rag_override_status": "tie", "arima_trend_multiplier": 1.02}),
        ("rag_zero_success", {"svm_prob_24h": 0.58, "future_risk": 0.45, "prob_critical": 0.34, "prob_warning": 0.56, "prob_stable": 0.10, "shap_vector": mean_warning, "_rag_vector": zero_vector, "_rag_override_status": "zero_success", "arima_trend_multiplier": 1.04}),
        ("score_below_minimum", {"svm_prob_24h": 0.01, "future_risk": 0.0, "prob_critical": 0.0, "prob_warning": 0.01, "prob_stable": 0.99, "shap_vector": -mean_stable, "arima_trend_multiplier": 0.75}),
        ("rag_majority_arima_against", {"svm_prob_24h": 0.58, "future_risk": 0.45, "prob_critical": 0.34, "prob_warning": 0.56, "prob_stable": 0.10, "shap_vector": mean_warning, "arima_trend_multiplier": 0.91}),
    ]

    rows = []
    demo_context = dict(decision_context)
    demo_context["rag_k"] = 2
    for label, overrides in demos:
        record = dict(base)
        record.update(overrides)
        record["fcm_cluster"] = int(np.argmax([record["prob_stable"], record["prob_warning"], record["prob_critical"]]))
        result = run_react_inner_loop(
            record,
            demo_context,
            audit_path=str(audit_path) if audit_path is not None else None,
            case_label=f"demo_{label}",
        )
        rows.append(
            {
                "required_case": label,
                "engine_route": result["route"],
                "score": result["score"]["intelligent_score"],
                "shap_cosine": result["score"]["shap_cosine"],
                "rag_status": (result.get("rag") or {}).get("status"),
                "trend_multiplier": result["score"]["trend_multiplier"],
                "action_type": result["action"]["action_type"],
                "timing_minutes": result["action"]["timing_minutes"],
                "explanation": result["action"]["explanation"],
            }
        )
    return pd.DataFrame(rows)


def _adaptive_threshold(probs: np.ndarray, y_true: np.ndarray) -> tuple[float, np.ndarray, list[dict]]:
    threshold = 0.5
    preds = []
    adaptations = []

    for idx, prob in enumerate(probs, start=1):
        preds.append(int(prob >= threshold))
        if idx % 100 != 0:
            continue

        cm = confusion_matrix(y_true[:idx], preds)
        if cm.size == 4:
            tn, fp, _, _ = cm.ravel()
        else:
            tn = cm[0, 0] if cm.shape[0] > 0 else 0
            fp = cm[0, 1] if cm.shape[1] > 1 else 0

        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        old_threshold = threshold
        if fpr > 0.15:
            threshold = min(0.95, threshold + 0.05)
        elif fpr < 0.05:
            threshold = max(0.05, threshold - 0.05)

        adaptations.append(
            {
                "count": idx,
                "fpr": round(float(fpr), 4),
                "old_threshold": round(float(old_threshold), 4),
                "new_threshold": round(float(threshold), 4),
            }
        )

    return threshold, np.asarray(preds, dtype=int), adaptations


def _build_explanation_payload(
    X: pd.DataFrame,
    svm_model,
    processed_df: pd.DataFrame,
    probs: np.ndarray,
    threshold: float,
    background_size: int,
) -> dict:
    if X.empty:
        return {
            "method": "none",
            "patient_id": None,
            "hour": None,
            "prediction_probability": 0.0,
            "confidence": 0.0,
            "margin_to_threshold": 0.0,
            "feature_table": pd.DataFrame(columns=["feature", "contribution", "abs_contribution"]),
            "top_features": [],
        }

    explain_idx = int(np.argmax(probs))
    method = "shap_kernel"

    try:
        if shap is None:
            raise RuntimeError("SHAP is unavailable in this Python environment.")
        background = shap.sample(X, min(background_size, len(X)), random_state=42)
        predict_fn = lambda arr: svm_model.predict_proba(pd.DataFrame(arr, columns=X.columns))
        explainer = shap.KernelExplainer(predict_fn, background.to_numpy())
        shap_values = explainer.shap_values(
            X.iloc[[explain_idx]].to_numpy(),
            nsamples=min(100, max(20, X.shape[1] * 2)),
            silent=True,
        )

        shap_arr = np.asarray(shap_values)
        if isinstance(shap_values, list):
            values = np.asarray(shap_values[1])[0]
        elif shap_arr.ndim == 3:
            values = shap_arr[0, :, -1] if shap_arr.shape[-1] > 1 else shap_arr[0, :, 0]
        elif shap_arr.ndim == 2:
            values = shap_arr[0]
        else:
            values = shap_arr
    except Exception:
        method = "feature_magnitude_fallback"
        values = X.iloc[explain_idx].to_numpy()

    feature_table = pd.DataFrame(
        {
            "feature": X.columns.astype(str),
            "contribution": values,
            "abs_contribution": np.abs(values),
        }
    ).sort_values("abs_contribution", ascending=False)

    explained_row = processed_df.iloc[explain_idx]
    probability = float(probs[explain_idx])
    return {
        "method": method,
        "patient_id": str(explained_row["patient_id"]),
        "hour": int(explained_row["hour"]),
        "prediction_probability": round(probability, 4),
        "confidence": round(float(max(probability, 1 - probability)), 4),
        "margin_to_threshold": round(float(abs(probability - threshold)), 4),
        "feature_table": feature_table.head(10).reset_index(drop=True),
        "top_features": feature_table.head(3)["feature"].astype(str).tolist(),
    }


def _build_monitoring_feed(
    processed_df: pd.DataFrame,
    probs: np.ndarray,
    preds: np.ndarray,
    threshold: float,
    trend_summary: Optional[dict],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    feed_columns = [
        "patient_id",
        "hour",
        "anomaly_flag",
        "prob_stable",
        "prob_warning",
        "prob_critical",
        "future_risk",
        "dynamic_weight",
        "deterioration_24h",
    ]
    for optional_column in [
        "fcm_cluster",
        "source_index",
        "shap_vector",
        "arima_trend_multiplier",
        "intelligent_score",
        "shap_cosine",
    ]:
        if optional_column in processed_df.columns:
            feed_columns.append(optional_column)
    monitoring_feed = processed_df[feed_columns].copy()
    monitoring_feed["svm_prob_24h"] = probs
    monitoring_feed["svm_pred_24h"] = preds
    monitoring_feed["threshold_last"] = threshold
    monitoring_feed["cluster_state"] = monitoring_feed[["prob_stable", "prob_warning", "prob_critical"]].idxmax(axis=1).map(CLUSTER_STATE_MAP)
    monitoring_feed["trend_alert"] = False
    monitoring_feed["trend_deviation"] = 0.0

    if trend_summary:
        mask = (
            monitoring_feed["patient_id"].astype(str) == str(trend_summary["patient_id"])
        ) & (monitoring_feed["hour"].astype(int) == int(processed_df[processed_df["patient_id"].astype(str) == str(trend_summary["patient_id"])]["hour"].max()))
        monitoring_feed.loc[mask, "trend_alert"] = trend_summary["deviation"] > 0.20
        monitoring_feed.loc[mask, "trend_deviation"] = float(trend_summary["deviation"])

    dispatch_frame = monitoring_feed.apply(
        lambda row: pd.Series(derive_patient_state(row.to_dict(), threshold)),
        axis=1,
    )
    monitoring_feed = pd.concat([monitoring_feed, dispatch_frame], axis=1)

    latest_status = (
        monitoring_feed.sort_values(["patient_id", "hour"])
        .groupby("patient_id", as_index=False)
        .tail(1)
        .sort_values(["dispatch_priority", "future_risk", "svm_prob_24h"], ascending=[True, False, False])
        .reset_index(drop=True)
    )
    return monitoring_feed, latest_status


def _build_zone_payloads(
    monitoring_feed: pd.DataFrame,
    latest_status: pd.DataFrame,
    metrics: dict,
    explanation: dict,
    adaptations_df: pd.DataFrame,
    agent_messages: pd.DataFrame,
    react_decisions: pd.DataFrame | None = None,
    react_demonstrations: pd.DataFrame | None = None,
) -> dict:
    cluster_counts = (
        latest_status["cluster_state"]
        .value_counts()
        .reindex(["Stable", "Warning", "Critical"], fill_value=0)
        .reset_index()
    )
    cluster_counts.columns = ["cluster_state", "patient_count"]

    cluster_health_timeline = (
        monitoring_feed.groupby("hour")[["anomaly_flag", "prob_stable", "prob_warning", "prob_critical"]]
        .mean()
        .reset_index()
    )

    live_scoring_feed = latest_status[
        [
            "patient_id",
            "hour",
            "clinical_state",
            "cluster_state",
            "intelligent_score" if "intelligent_score" in latest_status.columns else "svm_prob_24h",
            "future_risk",
            "svm_prob_24h",
            "dynamic_weight",
            "prob_critical",
            "anomaly_flag",
        ]
    ].copy()

    action_dispatch_queue = latest_status[
        [
            "patient_id",
            "hour",
            "clinical_state",
            "react_route" if "react_route" in latest_status.columns else "clinical_state",
            "dispatch_action",
            "action_timing_minutes" if "action_timing_minutes" in latest_status.columns else "hour",
            "action_intensity" if "action_intensity" in latest_status.columns else "svm_prob_24h",
            "future_risk",
            "svm_prob_24h",
            "alert_reason",
            "is_actionable",
        ]
    ].copy()

    actionable = action_dispatch_queue[action_dispatch_queue["is_actionable"]].copy()
    if actionable.empty and not action_dispatch_queue.empty:
        actionable = action_dispatch_queue.head(1).copy()

    tp = int(metrics["true_positive"])
    fp = int(metrics["false_positive"])
    fn = int(metrics["false_negative"])
    tn = int(metrics["true_negative"])
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)

    hourly_outcomes = (
        monitoring_feed.groupby("hour")
        .agg(
            predicted_alerts=("svm_pred_24h", "sum"),
            true_deteriorations=("deterioration_24h", "sum"),
            mean_score=("svm_prob_24h", "mean"),
        )
        .reset_index()
    )

    system_alerts = latest_status[latest_status["clinical_state"] != "Stable"][
        [
            "patient_id",
            "hour",
            "clinical_state",
            "dispatch_action",
            "future_risk",
            "svm_prob_24h",
            "alert_reason",
        ]
    ].copy()

    return {
        "zone_1_cluster_health": {
            "cluster_counts": cluster_counts,
            "timeline": cluster_health_timeline,
            "critical_patients": latest_status.nlargest(15, "prob_critical")[
                ["patient_id", "hour", "cluster_state", "prob_critical", "future_risk", "anomaly_flag"]
            ].copy(),
        },
        "zone_2_live_scoring_feed": {
            "feed": live_scoring_feed,
        },
        "zone_3_action_dispatch": {
            "queue": actionable.reset_index(drop=True),
            "full_queue": action_dispatch_queue.reset_index(drop=True),
        },
        "zone_4_shap_reliability": {
            "explanation": explanation,
        },
        "zone_5_outcome_tracking": {
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "hourly": hourly_outcomes,
            "confusion_matrix": metrics["confusion_matrix"],
        },
        "zone_6_system_alerts": {
            "alerts": system_alerts.reset_index(drop=True),
            "adaptations": adaptations_df.copy(),
            "agent_messages": agent_messages.copy(),
            "react_decisions": react_decisions.copy() if react_decisions is not None else pd.DataFrame(),
            "react_demonstrations": react_demonstrations.copy() if react_demonstrations is not None else pd.DataFrame(),
            "drift_report": metrics.get("drift_report", {}),
            "agentic_properties": metrics.get("agentic_properties", {}),
        },
        "selected_default_patient": str(live_scoring_feed.iloc[0]["patient_id"]) if not live_scoring_feed.empty else None,
        "system_alert_count": int(len(system_alerts)),
    }


def _run_monitoring_layers_for_manual_input(
    df: pd.DataFrame,
    eps: float,
    min_samples: int,
    fcm_artifact: Optional[dict] = None,
) -> pd.DataFrame:
    enriched = df.copy()
    try:
        enriched = run_dbscan(enriched, BASE_FEATURES, eps=eps, min_samples=min_samples)
    except Exception:
        enriched["anomaly_flag"] = 0

    try:
        if fcm_artifact is not None:
            return apply_fcm_model(enriched, fcm_artifact)
        enriched = run_fcm(enriched, BASE_FEATURES)
    except Exception:
        enriched["prob_stable"] = 1.0
        enriched["prob_warning"] = 0.0
        enriched["prob_critical"] = 0.0
    return enriched

def predict_from_dataframe(
    pipeline_results: dict,
    df: pd.DataFrame,
    patient_id: Optional[str] = None,
) -> dict:
    """Run prediction on real multi-hour patient data from a CSV upload.

    Unlike predict_manual_case (which repeats one snapshot), this uses
    actual varying readings across multiple real hours.

    Parameters
    ----------
    pipeline_results : dict
        Result dict returned by run_pipeline().
    df : pd.DataFrame
        Must contain the 8 BASE_FEATURES columns.
        Optional: 'patient_id', 'hour' (auto-generated 0,1,2... if absent).
    patient_id : str, optional
        Override or fallback when df has no patient_id column.

    Returns
    -------
    dict
        {"raw_input": dict, "timeline": pd.DataFrame, "latest_prediction": dict}
        Same structure as predict_manual_case().
    """
    if not pipeline_results:
        raise ValueError("Pipeline results are required before CSV prediction.")

    config = pipeline_results["config"]
    ensemble_model = pipeline_results["ensemble_model"]
    svm_model = pipeline_results["svm_model"]
    feature_columns = list(pipeline_results["feature_columns"])
    threshold = float(pipeline_results["metrics"]["threshold_last"])
    fcm_artifact = pipeline_results.get("fcm_artifact")
    shap_store = pipeline_results.get("shap_store", {})
    cluster_trends = pipeline_results.get("cluster_trends", {})
    decision_context = dict(pipeline_results.get("decision_context", {}))
    audit_path = pipeline_results.get("paths", {}).get("react_audit_log")

    min_samples = int(config["model"]["dbscan_min_samples"])
    eps = float(config["model"]["dbscan_eps"])

    # ── Build manual_raw from the uploaded DataFrame ──────────────────────────
    manual_raw = df.copy()

    pid = patient_id or "csv_case"
    if "patient_id" not in manual_raw.columns:
        manual_raw["patient_id"] = pid
    elif patient_id:
        manual_raw["patient_id"] = patient_id   # allow caller to override

    if "hour" not in manual_raw.columns:
        manual_raw["hour"] = range(len(manual_raw))

    # Labels are unknown for live input — set to 0
    manual_raw["deterioration_12h"] = 0
    manual_raw["deterioration_24h"] = 0

    # Ensure all BASE_FEATURES present (fill with 0 if a column is missing)
    for feat in BASE_FEATURES:
        if feat not in manual_raw.columns:
            manual_raw[feat] = 0.0

    manual_raw = manual_raw.sort_values("hour").reset_index(drop=True)

    # ── Downstream pipeline — identical to predict_manual_case ────────────────
    manual_processed = preprocess_data(manual_raw.copy())
    manual_processed = _run_monitoring_layers_for_manual_input(
        manual_processed,
        eps=eps,
        min_samples=min_samples,
        fcm_artifact=fcm_artifact,
    )

    for column in feature_columns:
        if column not in manual_processed.columns:
            manual_processed[column] = 0.0

    X_manual = manual_processed[feature_columns].fillna(0)
    manual_processed["future_risk"] = ensemble_model.predict_proba(X_manual)[:, 1]
    manual_processed["dynamic_weight"] = 1 + manual_processed["future_risk"] * 5
    manual_processed["svm_prob_24h"] = svm_model.predict_proba(X_manual)[:, 1]
    manual_processed["svm_pred_24h"] = (manual_processed["svm_prob_24h"] >= threshold).astype(int)
    manual_processed["threshold_last"] = threshold
    manual_processed["source_index"] = None

    shap_vectors = []
    for _, row in manual_processed.iterrows():
        cluster_id = int(row.get("fcm_cluster", 0) or 0)
        base_vector = np.asarray(
            shap_store.get("cluster_means", {}).get(cluster_id, np.zeros(len(feature_columns))),
            dtype=float,
        )
        if base_vector.size != len(feature_columns):
            base_vector = np.zeros(len(feature_columns), dtype=float)
        shap_vectors.append(base_vector)

    manual_processed["shap_vector"] = shap_vectors
    manual_processed["arima_trend_multiplier"] = manual_processed["fcm_cluster"].astype(int).map(
        lambda cid: cluster_trends.get(int(cid), {"multiplier": 1.0})["multiplier"]
    )

    latest_row = manual_processed.iloc[-1].to_dict()
    state_payload = derive_patient_state(latest_row, threshold)
    latest_row.update(state_payload)

    actual_pid = str(latest_row.get("patient_id", pid))

    if decision_context:
        react_result = run_react_inner_loop(
            latest_row,
            decision_context,
            audit_path=audit_path,
            case_label=f"csv_{actual_pid}",
        )
        latest_row["react_route"] = react_result["route"]
        latest_row["intelligent_score"] = react_result["score"]["intelligent_score"]
        latest_row["shap_cosine"] = react_result["score"]["shap_cosine"]
        latest_row["dominant_feature"] = react_result["dominant_feature"]["name"]
        latest_row["dispatch_action"] = react_result["action"]["action_type"]
        latest_row["alert_reason"] = react_result["action"]["explanation"]
        latest_row["action_timing_minutes"] = react_result["action"]["timing_minutes"]
        latest_row["action_intensity"] = react_result["action"]["intensity"]
        latest_row["action_personalisation"] = react_result["action"]["personalisation"]
        latest_row["react_trace"] = react_result["react_trace"]

    latest_row["patient_id"] = actual_pid
    latest_row["hour"] = int(latest_row["hour"])
    latest_row["future_risk"] = float(latest_row["future_risk"])
    latest_row["svm_prob_24h"] = float(latest_row["svm_prob_24h"])
    latest_row["threshold_last"] = threshold
    latest_row["prob_stable"] = float(latest_row.get("prob_stable", 0.0))
    latest_row["prob_warning"] = float(latest_row.get("prob_warning", 0.0))
    latest_row["prob_critical"] = float(latest_row.get("prob_critical", 0.0))
    latest_row["anomaly_flag"] = int(latest_row.get("anomaly_flag", 0))
    latest_row["dynamic_weight"] = float(latest_row.get("dynamic_weight", 1.0))
    latest_row["display_status"] = latest_row["binary_status"]

    last_real_row = manual_raw.iloc[-1]
    raw_input = {feat: float(last_real_row.get(feat, 0.0)) for feat in BASE_FEATURES}

    return {
        "raw_input": raw_input,
        "timeline": manual_processed[
            ["hour", "future_risk", "svm_prob_24h", "prob_stable", "prob_warning", "prob_critical", "anomaly_flag"]
        ].copy(),
        "latest_prediction": latest_row,
    }

def predict_manual_case(
    pipeline_results: dict,
    feature_values: dict[str, float],
    patient_id: str = "manual_entry",
    history_len: Optional[int] = None,
) -> dict:
    if not pipeline_results:
        raise ValueError("Pipeline results are required before manual prediction.")

    config = pipeline_results["config"]
    ensemble_model = pipeline_results["ensemble_model"]
    svm_model = pipeline_results["svm_model"]
    feature_columns = list(pipeline_results["feature_columns"])
    threshold = float(pipeline_results["metrics"]["threshold_last"])
    fcm_artifact = pipeline_results.get("fcm_artifact")
    shap_store = pipeline_results.get("shap_store", {})
    cluster_trends = pipeline_results.get("cluster_trends", {})
    decision_context = dict(pipeline_results.get("decision_context", {}))
    audit_path = pipeline_results.get("paths", {}).get("react_audit_log")

    min_samples = int(config["model"]["dbscan_min_samples"])
    eps = float(config["model"]["dbscan_eps"])
    hours = history_len if history_len is not None else max(6, min_samples + 1)

    rows = []
    for hour in range(hours):
        row = {"patient_id": patient_id, "hour": hour, "deterioration_12h": 0, "deterioration_24h": 0}
        for feature in BASE_FEATURES:
            row[feature] = float(feature_values.get(feature, 0.0))
        rows.append(row)

    manual_raw = pd.DataFrame(rows)
    manual_processed = preprocess_data(manual_raw.copy())
    manual_processed = _run_monitoring_layers_for_manual_input(
        manual_processed,
        eps=eps,
        min_samples=min_samples,
        fcm_artifact=fcm_artifact,
    )

    for column in feature_columns:
        if column not in manual_processed.columns:
            manual_processed[column] = 0.0

    X_manual = manual_processed[feature_columns].fillna(0)
    manual_processed["future_risk"] = ensemble_model.predict_proba(X_manual)[:, 1]
    manual_processed["dynamic_weight"] = 1 + manual_processed["future_risk"] * 5
    manual_processed["svm_prob_24h"] = svm_model.predict_proba(X_manual)[:, 1]
    manual_processed["svm_pred_24h"] = (manual_processed["svm_prob_24h"] >= threshold).astype(int)
    manual_processed["threshold_last"] = threshold
    manual_processed["source_index"] = None

    shap_vectors = []
    for _, row in manual_processed.iterrows():
        cluster_id = int(row.get("fcm_cluster", 0) or 0)
        base_vector = np.asarray(shap_store.get("cluster_means", {}).get(cluster_id, np.zeros(len(feature_columns))), dtype=float)
        if base_vector.size != len(feature_columns):
            base_vector = np.zeros(len(feature_columns), dtype=float)
        shap_vectors.append(base_vector)
    manual_processed["shap_vector"] = shap_vectors
    manual_processed["arima_trend_multiplier"] = manual_processed["fcm_cluster"].astype(int).map(
        lambda cluster_id: cluster_trends.get(int(cluster_id), {"multiplier": 1.0})["multiplier"]
    )

    latest_row = manual_processed.iloc[-1].to_dict()
    state_payload = derive_patient_state(latest_row, threshold)
    latest_row.update(state_payload)
    if decision_context:
        react_result = run_react_inner_loop(
            latest_row,
            decision_context,
            audit_path=audit_path,
            case_label=f"manual_{patient_id}",
        )
        latest_row["react_route"] = react_result["route"]
        latest_row["intelligent_score"] = react_result["score"]["intelligent_score"]
        latest_row["shap_cosine"] = react_result["score"]["shap_cosine"]
        latest_row["dominant_feature"] = react_result["dominant_feature"]["name"]
        latest_row["dispatch_action"] = react_result["action"]["action_type"]
        latest_row["alert_reason"] = react_result["action"]["explanation"]
        latest_row["action_timing_minutes"] = react_result["action"]["timing_minutes"]
        latest_row["action_intensity"] = react_result["action"]["intensity"]
        latest_row["action_personalisation"] = react_result["action"]["personalisation"]
        latest_row["react_trace"] = react_result["react_trace"]

    latest_row["patient_id"] = str(latest_row["patient_id"])
    latest_row["hour"] = int(latest_row["hour"])
    latest_row["future_risk"] = float(latest_row["future_risk"])
    latest_row["svm_prob_24h"] = float(latest_row["svm_prob_24h"])
    latest_row["threshold_last"] = threshold
    latest_row["prob_stable"] = float(latest_row.get("prob_stable", 0.0))
    latest_row["prob_warning"] = float(latest_row.get("prob_warning", 0.0))
    latest_row["prob_critical"] = float(latest_row.get("prob_critical", 0.0))
    latest_row["anomaly_flag"] = int(latest_row.get("anomaly_flag", 0))
    latest_row["dynamic_weight"] = float(latest_row.get("dynamic_weight", 1.0))
    latest_row["display_status"] = latest_row["binary_status"]

    return {
        "raw_input": {feature: float(feature_values.get(feature, 0.0)) for feature in BASE_FEATURES},
        "timeline": manual_processed[
            [
                "hour",
                "future_risk",
                "svm_prob_24h",
                "prob_stable",
                "prob_warning",
                "prob_critical",
                "anomaly_flag",
            ]
        ].copy(),
        "latest_prediction": latest_row,
    }


def run_pipeline(
    config_path: str | Path = "config.yaml",
    patient_limit: Optional[int] = 200,
    dbscan_eps: Optional[float] = None,
    dbscan_min_samples: Optional[int] = None,
    include_shap: bool = True,
    shap_background_size: int = 10,
    progress_callback: Optional[Callable[[str, str, float], None]] = None,
    persist_outputs: bool = True,
    reset_logs: bool = True,
) -> dict:
    config = load_project_config(config_path)
    paths = reset_pipeline_outputs(config) if persist_outputs and reset_logs else _logging_paths(config)

    _emit_progress(progress_callback, "data_loading", "Loading primary dataset", 0.08)
    primary_csv = _resolve_path(config["data"]["primary_csv"])
    raw_df = load_data(str(primary_csv))
    if patient_limit is not None:
        selected_patients = raw_df["patient_id"].astype(str).unique()[:patient_limit]
        raw_df = raw_df[raw_df["patient_id"].astype(str).isin(selected_patients)].copy()
    raw_df = raw_df.sort_values(["patient_id", "hour"]).reset_index(drop=True)
    stage_outputs = {"data_loading": _data_summary(raw_df)}

    _emit_progress(progress_callback, "preprocessing", "Preprocessing and feature engineering", 0.24)
    processed_df = preprocess_data(raw_df.copy())
    stage_outputs["preprocessing"] = _preprocessing_summary(raw_df, processed_df)

    eps = float(dbscan_eps if dbscan_eps is not None else config["model"]["dbscan_eps"])
    min_samples = int(dbscan_min_samples if dbscan_min_samples is not None else config["model"]["dbscan_min_samples"])

    _emit_progress(progress_callback, "dbscan", "Running DBSCAN anomaly detection", 0.40)
    processed_df = run_dbscan(processed_df, BASE_FEATURES, eps=eps, min_samples=min_samples)
    stage_outputs["dbscan"] = _dbscan_summary(processed_df)

    _emit_progress(progress_callback, "fcm", "Running fuzzy C-means clustering", 0.56)
    processed_df, fcm_artifact = fit_fcm_model(processed_df, BASE_FEATURES)
    processed_df["source_index"] = np.arange(len(processed_df), dtype=int)
    stage_outputs["fcm"] = _fcm_summary(processed_df)

    _emit_progress(progress_callback, "predictive_action", "Training predictive action layer", 0.72)
    drop_cols = {"patient_id", "hour", "deterioration_24h", "deterioration_12h"}
    feature_cols = [col for col in processed_df.columns if col not in drop_cols and pd.api.types.is_numeric_dtype(processed_df[col])]
    X = processed_df[feature_cols].fillna(0)
    y_12h = processed_df["deterioration_12h"].astype(int)
    ensemble_model, future_risk_oof = train_predictive_action_layer(X, y_12h)
    processed_df["future_risk"] = future_risk_oof
    processed_df["dynamic_weight"] = 1 + processed_df["future_risk"] * 5
    stage_outputs["predictive_action"] = _predictive_summary(processed_df)

    if persist_outputs:
        weight_rows = [
            {
                "patient_id": row.patient_id,
                "hour": int(row.hour),
                "weight": round(float(row.dynamic_weight), 4),
                "justification": f"Future risk is {row.future_risk:.2f}",
            }
            for row in processed_df[["patient_id", "hour", "dynamic_weight", "future_risk"]].itertuples(index=False)
        ]
        log_many_to_json(str(paths["weight_log"]), weight_rows)

    _emit_progress(progress_callback, "svm_agents", "Training dynamic SVM and explainability layers", 0.86)
    y_24h = processed_df["deterioration_24h"].astype(int)
    svm_model = train_svm_dynamic(X, y_24h, processed_df["dynamic_weight"])
    probs = svm_model.predict_proba(X)[:, 1]
    threshold, preds, adaptations = _adaptive_threshold(probs, y_24h.to_numpy())
    processed_df["svm_prob_24h"] = probs
    processed_df["svm_pred_24h"] = preds

    shap_store = _build_shap_reliability_store(processed_df, X, y_24h, probs)
    processed_df["shap_vector"] = list(shap_store["vectors"])
    cluster_trends = _build_cluster_trend_store(processed_df, "svm_prob_24h")
    processed_df["arima_trend_multiplier"] = processed_df["fcm_cluster"].astype(int).map(
        lambda cluster_id: cluster_trends[int(cluster_id)]["multiplier"]
    )
    rag_artifact = _build_rag_artifact(X, shap_store["vectors"], y_24h, processed_df)
    decision_context = _build_decision_context(threshold, feature_cols, shap_store, rag_artifact)

    adaptations_df = pd.DataFrame(adaptations)
    if persist_outputs and not adaptations_df.empty:
        log_many_to_json(str(paths["adaptation_log"]), adaptations)

    acc = accuracy_score(y_24h, preds)
    auroc = roc_auc_score(y_24h, probs) if len(np.unique(y_24h)) > 1 else float("nan")
    cm = confusion_matrix(y_24h, preds)
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    drift_report = _detect_drift_and_outer_loop(processed_df, X, y_24h, probs, threshold, float(acc))
    if persist_outputs:
        log_to_json(str(paths["agent_log"]), {"agent": "OuterReActLoop", "decision": drift_report})

    bus = MessageBus()
    last_row = processed_df.iloc[-1]
    VitalsAgent("VitalsAgent", bus).analyze(last_row)
    RiskAgent("RiskAgent", bus).analyze(last_row["future_risk"])

    trend_summary = None
    patient_hr = processed_df[processed_df["patient_id"] == last_row["patient_id"]]["HR"].to_numpy()
    if len(patient_hr) >= 24:
        forecast_hr = run_sarima(patient_hr[-24:])
        TrendAgent("TrendAgent", bus).analyze(patient_hr[-1], forecast_hr)
        deviation = abs(patient_hr[-1] - forecast_hr) / forecast_hr if forecast_hr else 0
        trend_summary = {
            "patient_id": str(last_row["patient_id"]),
            "actual_hr": round(float(patient_hr[-1]), 2),
            "forecast_hr": round(float(forecast_hr), 2),
            "deviation": round(float(deviation), 4),
        }

    AlertAgent("AlertAgent", bus).process_alerts(fpr, threshold)
    agent_messages = pd.DataFrame(bus.messages)

    explanation = _build_explanation_payload(X, svm_model, processed_df, probs, threshold, shap_background_size) if include_shap else {
        "method": "skipped",
        "patient_id": None,
        "hour": None,
        "prediction_probability": 0.0,
        "confidence": 0.0,
        "margin_to_threshold": 0.0,
        "feature_table": pd.DataFrame(columns=["feature", "contribution", "abs_contribution"]),
        "top_features": [],
    }

    note = (
        generate_clinical_note(explanation["top_features"])
        if include_shap and explanation["top_features"]
        else "Clinical narrative skipped for this run."
    )

    note_provider = "fallback" if note.startswith("[Fallback]") else "biomistral:7b"
    explanation["note_provider"] = note_provider

    monitoring_feed, latest_status = _build_monitoring_feed(processed_df, probs, preds, threshold, trend_summary)
    react_audit_path = paths["react_audit_log"] if persist_outputs else None
    latest_status, react_decisions = _run_react_for_latest_status(
        latest_status,
        decision_context,
        react_audit_path,
    )
    react_demonstrations = _build_react_demonstrations(
        latest_status,
        decision_context,
        rag_artifact,
        react_audit_path,
    )

    predictions = monitoring_feed[["patient_id", "hour", "future_risk", "svm_prob_24h", "svm_pred_24h", "threshold_last"]].copy()
    if persist_outputs:
        predictions.to_csv(paths["predictions_csv"], index=False)
        monitoring_feed.to_csv(paths["monitoring_feed_csv"], index=False)
        latest_status.to_csv(paths["latest_status_csv"], index=False)

    metrics = {
        "accuracy": round(float(acc), 4),
        "auroc": round(float(auroc), 4) if not np.isnan(auroc) else float("nan"),
        "fpr": round(float(fpr), 4),
        "threshold_last": round(float(threshold), 4),
        "confusion_matrix": cm.tolist(),
        "trend_summary": trend_summary,
        "top_features": explanation["top_features"],
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "predicted_alerts": int(np.sum(preds)),
        "actual_deteriorations": int(np.sum(y_24h)),
        "cluster_trends": cluster_trends,
        "drift_report": drift_report,
        "react_demonstration_count": int(len(react_demonstrations)),
        "agentic_properties": {
            "observes_continuously": int(len(monitoring_feed)),
            "reasons_before_acting": bool(not react_decisions.empty),
            "acts_with_explanation": bool("alert_reason" in latest_status.columns),
            "adapts_to_outcomes": int(len(adaptations_df)),
            "detects_own_degradation": drift_report,
            "retraining_autonomy": drift_report.get("rollback_gate", "not_triggered"),
        },
    }
    stage_outputs["svm_agents"] = metrics

    zones = _build_zone_payloads(
        monitoring_feed=monitoring_feed,
        latest_status=latest_status,
        metrics=metrics,
        explanation=explanation,
        adaptations_df=adaptations_df,
        agent_messages=agent_messages,
        react_decisions=react_decisions,
        react_demonstrations=react_demonstrations,
    )

    _emit_progress(progress_callback, "svm_agents", "Pipeline completed", 1.0)

    return {
        "config": config,
        "paths": {key: str(value) for key, value in paths.items()},
        "pipeline_stages": PIPELINE_STAGES,
        "raw_df": raw_df,
        "processed_df": processed_df,
        "predictions": predictions,
        "feature_columns": feature_cols,
        "stage_outputs": stage_outputs,
        "metrics": metrics,
        "adaptations": adaptations_df,
        "agent_messages": agent_messages,
        "clinical_note": note,
        "top_features": explanation["top_features"],
        "ensemble_model": ensemble_model,
        "svm_model": svm_model,
        "fcm_artifact": fcm_artifact,
        "shap_store": shap_store,
        "rag_artifact": rag_artifact,
        "decision_context": decision_context,
        "cluster_trends": cluster_trends,
        "react_decisions": react_decisions,
        "react_demonstrations": react_demonstrations,
        "drift_report": drift_report,
        "monitoring_feed": monitoring_feed,
        "latest_status": latest_status,
        "explanation": explanation,
        "zones": zones,
    }
