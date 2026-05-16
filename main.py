import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "02_src"))

from pipeline_runner import BASE_FEATURES, predict_manual_case, run_pipeline
from utils import read_json_log

try:
    from fastapi import FastAPI
    from pydantic import BaseModel, Field
except Exception:  # pragma: no cover - CLI can run without API extras
    FastAPI = None
    BaseModel = object
    Field = None


PIPELINE_CACHE: Optional[dict] = None


def _json_safe(value):
    if isinstance(value, pd.DataFrame):
        return [_json_safe(row) for row in value.to_dict(orient="records")]
    if isinstance(value, pd.Series):
        return _json_safe(value.to_dict())
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        safe = {}
        for key, item in value.items():
            if callable(item) or key in {"nearest_neighbors", "feature_scaler", "shap_scaler"}:
                continue
            safe[str(key)] = _json_safe(item)
        return safe
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _get_or_train(patient_limit: int = 80) -> dict:
    global PIPELINE_CACHE
    if PIPELINE_CACHE is None:
        PIPELINE_CACHE = run_pipeline(patient_limit=patient_limit, include_shap=False, reset_logs=False)
    return PIPELINE_CACHE


if FastAPI is not None:
    app = FastAPI(title="Chronic Patient Monitoring SRA API")

    class TrainingRequest(BaseModel):
        patient_limit: int = 80
        include_shap: bool = False
        reset_logs: bool = False

    class LiveCaseRequest(BaseModel):
        patient_id: str = "api_case"
        history_len: int = 6
        HR: float = 82.0
        BP: float = 118.0
        SpO2: float = 97.0
        respiration: float = 18.0
        temp: float = 37.1
        creatinine: float = 1.0
        WBC: float = 7.0
        lactate: float = 1.2

    @app.post("/api/training-cycle")
    def api_training_cycle(request: TrainingRequest):
        global PIPELINE_CACHE
        PIPELINE_CACHE = run_pipeline(
            patient_limit=request.patient_limit,
            include_shap=request.include_shap,
            reset_logs=request.reset_logs,
        )
        return _json_safe(
            {
                "loop_layer": "offline_training_cycle",
                "implemented_steps": [
                    "FCM clustering with saved centroids",
                    "SVM training",
                    "SHAP cluster means",
                    "KNN RAG index build",
                    "ARIMA cluster trend baseline",
                    "ReAct route demonstrations",
                ],
                "metrics": PIPELINE_CACHE["metrics"],
                "react_demonstrations": PIPELINE_CACHE["react_demonstrations"],
                "artifacts_ready": {
                    "fcm_centroids": True,
                    "svm_model": True,
                    "shap_cluster_means": True,
                    "rag_index": True,
                    "arima_baselines": True,
                },
            }
        )

    @app.post("/api/live-cycle")
    def api_live_cycle(request: LiveCaseRequest):
        results = _get_or_train()
        features = {feature: float(getattr(request, feature)) for feature in BASE_FEATURES}
        prediction = predict_manual_case(
            results,
            features,
            patient_id=request.patient_id,
            history_len=request.history_len,
        )
        return _json_safe(
            {
                "loop_layer": "live_prediction_cycle",
                "centroid_policy": "saved FCM centroids reused; FCM is not rerun for this arrival",
                "inner_react_loop": prediction["latest_prediction"].get("react_trace", []),
                "latest_prediction": prediction["latest_prediction"],
            }
        )

    @app.get("/api/inner-react-loop")
    def api_inner_react_loop():
        results = _get_or_train()
        return _json_safe(
            {
                "loop_layer": "inner_react_loop",
                "latest_live_decisions": results["react_decisions"],
                "seven_route_demonstrations": results["react_demonstrations"],
                "audit_tail": read_json_log(results["paths"]["react_audit_log"], tail=12),
            }
        )

    @app.get("/api/rag-layer")
    def api_rag_layer():
        results = _get_or_train()
        demonstrations = results["react_demonstrations"]
        return _json_safe(
            {
                "loop_layer": "rag_layer",
                "explicit_rag": "KNN over combined feature + SHAP vectors, queried for borderline scores",
                "implicit_rag": "Cosine similarity to cluster mean SHAP vector",
                "rag_related_routes": demonstrations[
                    demonstrations["engine_route"].astype(str).str.contains("rag", case=False, na=False)
                ],
                "cluster_trends": results["cluster_trends"],
            }
        )

    @app.get("/api/drift-outer-loop")
    def api_drift_outer_loop():
        results = _get_or_train()
        return _json_safe(
            {
                "loop_layer": "outer_react_drift_loop",
                "policy": "gap trend decline and centroid movement must both fire before retraining",
                "drift_report": results["drift_report"],
            }
        )

    @app.get("/api/dashboard-zones")
    def api_dashboard_zones():
        results = _get_or_train()
        zones = results["zones"]
        return _json_safe(
            {
                "loop_layer": "dashboard_monitoring",
                "zone_1_cluster_health": zones["zone_1_cluster_health"],
                "zone_2_live_scoring_feed": zones["zone_2_live_scoring_feed"],
                "zone_3_action_dispatch": zones["zone_3_action_dispatch"],
                "zone_4_shap_reliability": zones["zone_4_shap_reliability"],
                "zone_5_outcome_tracking": zones["zone_5_outcome_tracking"],
                "zone_6_system_alerts": zones["zone_6_system_alerts"],
            }
        )
else:
    app = None


def _parse_args():
    parser = argparse.ArgumentParser(description="Run the chronic patient monitoring pipeline.")
    parser.add_argument("--patient-limit", type=int, default=200, help="Number of patients to process.")
    parser.add_argument("--dbscan-eps", type=float, default=None, help="Override DBSCAN eps.")
    parser.add_argument("--dbscan-min-samples", type=int, default=None, help="Override DBSCAN min samples.")
    parser.add_argument("--skip-shap", action="store_true", help="Skip SHAP and the clinical note stage.")
    parser.add_argument("--shap-background-size", type=int, default=10, help="SHAP background sample size.")
    parser.add_argument("--keep-logs", action="store_true", help="Append to existing logs instead of clearing them first.")
    return parser.parse_args()


def main():
    args = _parse_args()
    started_at = time.time()

    print("Starting pipeline. Full runs can take a few minutes on 200 patients.")
    results = run_pipeline(
        patient_limit=args.patient_limit,
        dbscan_eps=args.dbscan_eps,
        dbscan_min_samples=args.dbscan_min_samples,
        include_shap=not args.skip_shap,
        shap_background_size=args.shap_background_size,
        reset_logs=not args.keep_logs,
    )

    metrics = results["metrics"]
    stage_outputs = results["stage_outputs"]
    zones = results["zones"]
    elapsed = time.time() - started_at

    print("\n--- Pipeline Summary ---")
    print(f"Rows processed: {stage_outputs['data_loading']['rows']}")
    print(f"Patients processed: {stage_outputs['data_loading']['patients']}")
    print(f"Engineered columns: {stage_outputs['preprocessing']['engineered_columns']}")
    print(f"DBSCAN anomaly count: {stage_outputs['dbscan']['anomaly_count']}")
    print(f"Average critical probability: {stage_outputs['fcm']['prob_critical_mean']:.4f}")

    print("\n--- Model Performance ---")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"AUROC: {metrics['auroc']:.4f}")
    print(f"FPR: {metrics['fpr']:.4f}")
    print(f"Threshold: {metrics['threshold_last']:.4f}")
    print(f"Predicted alerts: {metrics['predicted_alerts']}")
    print(f"Actual deteriorations: {metrics['actual_deteriorations']}")
    print(f"Confusion Matrix:\n{pd.DataFrame(metrics['confusion_matrix'])}")

    dispatch_queue = zones["zone_3_action_dispatch"]["queue"]
    if not dispatch_queue.empty:
        top_dispatch = dispatch_queue.iloc[0]
        print("\n--- Top Action Dispatch ---")
        print(f"Patient: {top_dispatch['patient_id']}")
        print(f"State: {top_dispatch['clinical_state']}")
        print(f"Action: {top_dispatch['dispatch_action']}")
        print(f"Reason: {top_dispatch['alert_reason']}")

    if metrics["top_features"]:
        print("\n--- Explainability ---")
        print(f"Top features: {', '.join(metrics['top_features'])}")
        print(f"Clinical Note:\n{results['clinical_note']}")

    print(f"\nCompleted in {elapsed:.1f} seconds.")
    print("Artifacts written to logs/, including predictions, monitoring feed, and latest status.")


if __name__ == "__main__":
    main()
