import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "02_src"))

from action_dispatch import (
    SMTP_SETTINGS_PATH,
    build_email_payload,
    build_smtp_settings_from_state,
    get_provider_preset,
    load_smtp_settings,
    log_dispatch_result,
    save_smtp_settings,
    send_email_notification,
    send_test_email,
)
from pipeline_runner import BASE_FEATURES, load_project_config, predict_manual_case, predict_from_dataframe, run_pipeline
from utils import read_json_log

st.set_page_config(page_title="Chronic Patient Monitoring SRA", layout="wide")

EMAIL_UI_DEFAULTS = {
    "recipient_email": "",
    "smtp_host": "smtp.gmail.com",
    "smtp_port": 587,
    "smtp_security": "STARTTLS",
    "smtp_username": "sender@gmail.com",
    "smtp_password": "app-password",
    "sender_email": "sender@gmail.com",
    "email_provider": "Gmail",
    "gemini_api_key": "",
}

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }
    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(circle at top left, rgba(255, 231, 210, 0.82), transparent 35%),
            radial-gradient(circle at top right, rgba(205, 239, 228, 0.88), transparent 30%),
            linear-gradient(180deg, #fffaf4 0%, #f3f8f4 52%, #eef5fb 100%);
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #dcefe9 0%, #c8e2da 100%);
    }
    [data-testid="stSidebar"] * {
        color: #10282d;
    }
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div {
        color: #10282d !important;
    }
    [data-testid="stSidebar"] [data-baseweb="input"] > div,
    [data-testid="stSidebar"] [data-baseweb="select"] > div,
    [data-testid="stSidebar"] textarea {
        background: #fffdf9 !important;
        border: 1px solid #b5cbc4 !important;
        border-radius: 12px !important;
    }
    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] textarea,
    [data-testid="stSidebar"] [data-baseweb="select"] input,
    [data-testid="stSidebar"] [data-baseweb="select"] span {
        color: #111111 !important;
        -webkit-text-fill-color: #111111 !important;
    }
    [data-testid="stSidebar"] input::placeholder,
    [data-testid="stSidebar"] textarea::placeholder {
        color: #5f6768 !important;
        opacity: 1 !important;
    }
    div[data-testid="metric-container"] {
        border: 1px solid #d0e3dd;
        background: linear-gradient(180deg, #ffffff 0%, #f3f8f7 100%);
        border-radius: 14px;
        padding: 0.8rem 1rem;
        box-shadow: 0 10px 24px rgba(24, 58, 67, 0.06);
    }
    div[data-testid="stVerticalBlock"] div[data-testid="stButton"] > button {
        background: linear-gradient(135deg, #d96c3d 0%, #ba4731 100%);
        color: #ffffff;
        border: none;
        border-radius: 12px;
        font-weight: 600;
    }
    div[data-testid="stVerticalBlock"] div[data-testid="stButton"] > button:hover {
        background: linear-gradient(135deg, #c95f34 0%, #a53d29 100%);
        color: #ffffff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _resolve_config_path(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (root / value).resolve()


def _load_log_frame(path: Path, tail: int = 200) -> pd.DataFrame:
    return pd.DataFrame(read_json_log(str(path), tail=tail))


def _status_palette(status: str) -> tuple[str, str, str]:
    if status == "Dangerous Situation":
        return "#8a1c1c", "#ffe1e1", "#f8a3a3"
    if status == "Warning":
        return "#7a4a05", "#fff1d9", "#f5c46a"
    return "#155b43", "#e2f7ee", "#84d5af"


def _render_status_card(title: str, status: str, message: str) -> None:
    text_color, bg_color, border_color = _status_palette(status)
    st.markdown(
        f"""
        <div style="
            background:{bg_color};
            border:1px solid {border_color};
            border-left:8px solid {text_color};
            border-radius:16px;
            padding:1rem 1.1rem;
            margin:0.3rem 0 0.8rem 0;
        ">
            <div style="font-size:0.88rem; color:{text_color}; font-weight:700; text-transform:uppercase;">{title}</div>
            <div style="font-size:1.45rem; color:{text_color}; font-weight:800; margin-top:0.2rem;">{status}</div>
            <div style="font-size:0.98rem; color:{text_color}; margin-top:0.45rem;">{message}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _send_dispatch_email(
    row: dict,
    dispatch_log_path: Path,
    recipient_email: str | None = None,
    source: str = "dispatch_zone",
) -> tuple[bool, str]:
    email_payload = build_email_payload(row)
    target_email = recipient_email or st.session_state.get("recipient_email", "")
    try:
        send_email_notification(
            smtp_host=st.session_state.get("smtp_host", ""),
            smtp_port=int(st.session_state.get("smtp_port", 587)),
            username=st.session_state.get("smtp_username", ""),
            password=st.session_state.get("smtp_password", ""),
            sender_email=st.session_state.get("sender_email", ""),
            recipient_email=target_email,
            subject=email_payload["subject"],
            body=email_payload["body"],
            security_mode=st.session_state.get("smtp_security", "STARTTLS"),
        )
        log_dispatch_result(
            str(dispatch_log_path),
            {
                "patient_id": row["patient_id"],
                "hour": int(row["hour"]),
                "clinical_state": row["clinical_state"],
                "binary_status": row.get("binary_status", row["clinical_state"]),
                "dispatch_action": row["dispatch_action"],
                "recipient_email": target_email,
                "status": "sent",
                "source": source,
                "subject": email_payload["subject"],
                "gemini_used": email_payload.get("gemini_used", ""),
                "gemini_error": email_payload.get("gemini_error", ""),
            },
        )
        return True, f"Email sent for patient {row['patient_id']} with status {row.get('binary_status', row['clinical_state'])}."
    except Exception as exc:
        log_dispatch_result(
            str(dispatch_log_path),
            {
                "patient_id": row["patient_id"],
                "hour": int(row["hour"]),
                "clinical_state": row["clinical_state"],
                "binary_status": row.get("binary_status", row["clinical_state"]),
                "dispatch_action": row["dispatch_action"],
                "recipient_email": target_email,
                "status": "failed",
                "source": source,
                "subject": email_payload["subject"],
                "gemini_used": email_payload.get("gemini_used", ""),
                "gemini_error": email_payload.get("gemini_error", ""),
                "error": str(exc),
            },
        )
        return False, str(exc)


def _pick_selected_patient(latest_status: pd.DataFrame, default_patient: str | None) -> str | None:
    if latest_status.empty:
        return None
    choices = latest_status["patient_id"].astype(str).tolist()
    if "selected_patient" not in st.session_state or st.session_state["selected_patient"] not in choices:
        st.session_state["selected_patient"] = default_patient or choices[0]
    index = choices.index(st.session_state["selected_patient"])
    return st.sidebar.selectbox("Patient focus (from test csv)", choices, index=index, key="selected_patient")


st.title("Chronic Patient Monitoring SRA Dashboard")
st.caption("Six live monitoring zones with action dispatch and email notification support.")

config = load_project_config()
dispatch_log_path = _resolve_config_path(PROJECT_ROOT, config["logging"]["dispatch_log"])
agent_log_path = _resolve_config_path(PROJECT_ROOT, config["logging"]["agent_log"])
adaptation_log_path = _resolve_config_path(PROJECT_ROOT, config["logging"]["adaptation_log"])
weight_log_path = _resolve_config_path(PROJECT_ROOT, config["logging"]["weight_log"])

if "pipeline_results" not in st.session_state:
    st.session_state["pipeline_results"] = None
if "dispatch_feedback" not in st.session_state:
    st.session_state["dispatch_feedback"] = None
if "manual_prediction" not in st.session_state:
    st.session_state["manual_prediction"] = None
loaded_email_defaults = load_smtp_settings(EMAIL_UI_DEFAULTS)
for key, value in loaded_email_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value
if st.session_state.get("gemini_api_key"):
    os.environ["GEMINI_API_KEY"] = st.session_state["gemini_api_key"].strip()

with st.sidebar:
    st.header("Run Controls")
    patient_limit = st.slider("Number of patients", min_value=20, max_value=200, value=80, step=20)
    dbscan_eps = st.slider("DBSCAN eps", min_value=0.1, max_value=2.0, value=float(config["model"]["dbscan_eps"]), step=0.1)
    dbscan_min_samples = st.slider("DBSCAN min samples", min_value=2, max_value=20, value=int(config["model"]["dbscan_min_samples"]), step=1)
    include_shap = st.checkbox("Run SHAP + clinical note", value=True)
    shap_background_size = st.slider("SHAP background size", min_value=5, max_value=20, value=10, step=5)
    reset_logs = st.checkbox("Clear previous logs before run", value=True)
    auto_send_email = st.checkbox("Auto-send email after run", value=False)
    run_pipeline_button = st.button(
    "Run pipeline",
    type="primary",
    use_container_width=True
    )
    st.divider()
    st.header("Email Action")
    provider_options = ["Gmail", "Outlook", "Yahoo", "Custom"]
    current_provider = st.session_state.get("email_provider", "Gmail")
    if current_provider not in provider_options:
        current_provider = "Custom"
    selected_provider = st.selectbox(
        "Email provider",
        provider_options,
        index=provider_options.index(current_provider),
        key="email_provider",
    )
    st.text_input("Recipient email", key="recipient_email")
    st.text_input("SMTP host", key="smtp_host")
    st.number_input("SMTP port", min_value=1, max_value=65535, step=1, key="smtp_port")
    st.selectbox(
        "SMTP security",
        ["STARTTLS", "SSL", "None"],
        index=["STARTTLS", "SSL", "None"].index(st.session_state["smtp_security"]),
        key="smtp_security",
    )
    st.text_input("SMTP username", key="smtp_username")
    st.text_input("SMTP password", type="password", key="smtp_password")
    st.text_input("Sender email", key="sender_email")
    st.text_input("Gemini API key", type="password", key="gemini_api_key")
    settings_button_cols = st.columns(2)
    apply_provider_button = settings_button_cols[0].button(
    "Apply provider",
    use_container_width=True
)   
    save_settings_button = settings_button_cols[1].button("Save email settings", width="stretch")
    test_email_button = st.button("Send test email", width="stretch")
    test_gemini_button = st.button("Test Gemini", width="stretch")
    st.caption(f"Saved SMTP file: `{SMTP_SETTINGS_PATH.name}`")
    st.caption("Use an app password for Gmail or Outlook if normal login is blocked.")
    st.caption("Gemini key is used only to write the email body; SMTP still sends the email.")

if apply_provider_button:
    preset = get_provider_preset(selected_provider)
    st.session_state["smtp_host"] = preset["smtp_host"] or st.session_state.get("smtp_host", "")
    st.session_state["smtp_port"] = preset["smtp_port"]
    st.session_state["smtp_security"] = preset["smtp_security"]
    st.session_state["email_provider"] = selected_provider
    st.rerun()

if save_settings_button:
    try:
        saved_path = save_smtp_settings(build_smtp_settings_from_state(st.session_state))
        st.session_state["dispatch_feedback"] = {
            "success": True,
            "message": f"SMTP settings saved locally to {saved_path}.",
        }
    except Exception as exc:
        st.session_state["dispatch_feedback"] = {"success": False, "message": str(exc)}
    st.rerun()

if test_email_button:
    try:
        recipient = send_test_email(build_smtp_settings_from_state(st.session_state))
        log_dispatch_result(
            str(dispatch_log_path),
            {
                "patient_id": "smtp_test",
                "hour": 0,
                "clinical_state": "SMTP Test",
                "binary_status": "SMTP Test",
                "dispatch_action": "Send test email",
                "recipient_email": recipient,
                "status": "sent",
                "source": "smtp_test",
                "subject": "[Patient Monitor] SMTP test email",
            },
        )
        st.session_state["dispatch_feedback"] = {
            "success": True,
            "message": f"Test email sent successfully to {recipient}.",
        }
    except Exception as exc:
        log_dispatch_result(
            str(dispatch_log_path),
            {
                "patient_id": "smtp_test",
                "hour": 0,
                "clinical_state": "SMTP Test",
                "binary_status": "SMTP Test",
                "dispatch_action": "Send test email",
                "recipient_email": st.session_state.get("recipient_email", ""),
                "status": "failed",
                "source": "smtp_test",
                "subject": "[Patient Monitor] SMTP test email",
                "error": str(exc),
            },
        )
        st.session_state["dispatch_feedback"] = {"success": False, "message": str(exc)}
    st.rerun()

if test_gemini_button:
    sample_payload = build_email_payload(
        {
            "patient_id": "gemini_test",
            "hour": 0,
            "clinical_state": "Critical",
            "binary_status": "Dangerous Situation",
            "dispatch_action": "Escalate now",
            "future_risk": 0.82,
            "svm_prob_24h": 0.77,
            "alert_reason": "Gemini connectivity test",
            "top_features": ["heart_rate", "spo2"],
            "dominant_feature": {"name": "spo2", "value": 0.45},
        }
    )
    if sample_payload.get("gemini_used") == "True":
        st.session_state["dispatch_feedback"] = {
            "success": True,
            "message": "Gemini test succeeded. The email body is being generated by Gemini.",
        }
    else:
        st.session_state["dispatch_feedback"] = {
            "success": False,
            "message": f"Gemini test failed: {sample_payload.get('gemini_error', 'unknown error')}",
        }
    st.rerun()

if run_pipeline_button:
    progress_bar = st.sidebar.progress(0)
    status_box = st.sidebar.empty()

    def _progress_callback(stage_key: str, label: str, ratio: float) -> None:
        del stage_key
        progress_bar.progress(int(ratio * 100))
        status_box.caption(label)

    try:
        with st.spinner("Running the monitoring pipeline..."):
            st.session_state["pipeline_results"] = run_pipeline(
                patient_limit=patient_limit,
                dbscan_eps=dbscan_eps,
                dbscan_min_samples=dbscan_min_samples,
                include_shap=include_shap,
                shap_background_size=shap_background_size,
                reset_logs=reset_logs,
                progress_callback=_progress_callback,
            )
        progress_bar.progress(100)
        status_box.success("Pipeline completed successfully.")

        results = st.session_state["pipeline_results"]
        dispatch_queue = results["zones"]["zone_3_action_dispatch"]["queue"]
        if auto_send_email and not dispatch_queue.empty and st.session_state.get("recipient_email"):
            success, feedback = _send_dispatch_email(dispatch_queue.iloc[0].to_dict(), dispatch_log_path)
            st.session_state["dispatch_feedback"] = {"success": success, "message": feedback}
        elif auto_send_email and not st.session_state.get("recipient_email"):
            st.session_state["dispatch_feedback"] = {
                "success": False,
                "message": "Auto-send is enabled, but no recipient email is configured.",
            }
    except Exception as exc:
        st.session_state["pipeline_results"] = None
        status_box.error("Pipeline failed.")
        st.exception(exc)

results = st.session_state["pipeline_results"]
if results is None:
    st.info("Run the pipeline from the sidebar to populate all six monitoring zones with live system data.")
    preview_cols = st.columns(3)
    with preview_cols[0]:
        st.subheader("Recent Agent Decisions")
        st.dataframe(_load_log_frame(agent_log_path, tail=25), width="stretch")
    with preview_cols[1]:
        st.subheader("Recent Adaptations")
        st.dataframe(_load_log_frame(adaptation_log_path, tail=25), width="stretch")
    with preview_cols[2]:
        st.subheader("Recent Dispatch History")
        st.dataframe(_load_log_frame(dispatch_log_path, tail=25), width="stretch")
    st.stop()

zones = results["zones"]
metrics = results["metrics"]
monitoring_feed = results["monitoring_feed"].copy()
latest_status = results["latest_status"].copy()
monitoring_feed["patient_id"] = monitoring_feed["patient_id"].astype(str)
latest_status["patient_id"] = latest_status["patient_id"].astype(str)

selected_patient = _pick_selected_patient(latest_status, zones["selected_default_patient"])
selected_status = latest_status[latest_status["patient_id"] == selected_patient].iloc[0] if selected_patient else None
selected_history = (
    monitoring_feed[monitoring_feed["patient_id"] == selected_patient].sort_values("hour") if selected_patient else pd.DataFrame()
)

top_metrics = st.columns(6)
top_metrics[0].metric("Patients", int(results["stage_outputs"]["data_loading"]["patients"]))
top_metrics[1].metric("Rows", int(results["stage_outputs"]["data_loading"]["rows"]))
top_metrics[2].metric("System Alerts", int(zones["system_alert_count"]))
top_metrics[3].metric("Predicted Alerts", int(metrics["predicted_alerts"]))
top_metrics[4].metric("Accuracy", f"{metrics['accuracy']:.3f}")
top_metrics[5].metric("AUROC", f"{metrics['auroc']:.3f}")

if st.session_state["dispatch_feedback"] is not None:
    feedback = st.session_state["dispatch_feedback"]
    if feedback["success"]:
        st.success(feedback["message"])
    else:
        st.warning(feedback["message"])
# ── CSV Patient Check ─────────────────────────────────────────────────────────
SAMPLE_CSV = (
    "patient_id,hour,HR,BP,SpO2,respiration,temp,creatinine,WBC,lactate\n"
    "patient_001,0,82.0,118.0,97.0,18.0,37.1,1.0,7.0,1.2\n"
    "patient_001,1,85.0,122.0,96.5,19.0,37.2,1.1,7.2,1.3\n"
    "patient_001,2,88.0,126.0,95.8,21.0,37.4,1.2,7.8,1.5\n"
    "patient_001,3,91.0,130.0,95.0,22.0,37.6,1.4,8.1,1.7\n"
    "patient_001,4,94.0,134.0,94.2,24.0,37.8,1.6,8.5,1.9\n"
    "patient_001,5,97.0,138.0,93.5,26.0,38.0,1.8,9.0,2.1\n"
)

with st.container(border=True):
    st.subheader("Patient CSV Check")
    st.caption(
        "Upload a CSV with real hourly readings. Required columns: "
        + ", ".join(BASE_FEATURES)
        + ". Optional: patient_id, hour."
    )

    st.download_button(
        "⬇ Download sample CSV template",
        data=SAMPLE_CSV,
        file_name="patient_template.csv",
        mime="text/csv",
    )

    uploaded_file = st.file_uploader(
        "Upload patient history CSV",
        type=["csv"],
        help="Columns: " + ", ".join(BASE_FEATURES) + " + optional patient_id, hour",
    )

    csv_df = None
    if uploaded_file is not None:
        try:
            csv_df = pd.read_csv(uploaded_file)
        except Exception as exc:
            st.error(f"Could not read CSV: {exc}")

        if csv_df is not None:
            missing_cols = [f for f in BASE_FEATURES if f not in csv_df.columns]
            if missing_cols:
                st.error(f"Missing required columns: {', '.join(missing_cols)}")
                csv_df = None

        if csv_df is not None:
            st.success(f"Loaded {len(csv_df)} rows × {len(csv_df.columns)} columns.")
            st.dataframe(csv_df.head(8), use_container_width=True)

            csv_action_cols = st.columns([2, 1, 1])

            if "patient_id" in csv_df.columns:
                patients_in_file = csv_df["patient_id"].astype(str).unique().tolist()
                if len(patients_in_file) > 1:
                    selected_csv_patient = csv_action_cols[0].selectbox(
                        "Patient to predict", patients_in_file, key="csv_patient_select"
                    )
                    csv_df_filtered = csv_df[
                        csv_df["patient_id"].astype(str) == selected_csv_patient
                    ].copy()
                else:
                    selected_csv_patient = patients_in_file[0]
                    csv_df_filtered = csv_df.copy()
                    csv_action_cols[0].text_input(
                        "Patient ID", value=selected_csv_patient, disabled=True
                    )
            else:
                selected_csv_patient = csv_action_cols[0].text_input(
                    "Case label", value="csv_case", key="csv_case_label"
                )
                csv_df_filtered = csv_df.copy()

            auto_email_csv = csv_action_cols[1].checkbox(
                "Auto-email if dangerous", value=True, key="csv_auto_email"
            )
            predict_csv_button = csv_action_cols[2].button(
                "Predict from CSV", type="primary", use_container_width=True
            )

            if predict_csv_button:
                try:
                    csv_prediction = predict_from_dataframe(
                        results,
                        csv_df_filtered,
                        patient_id=str(selected_csv_patient),
                    )
                    st.session_state["manual_prediction"] = csv_prediction
                    latest_csv = csv_prediction["latest_prediction"]
                    recipient = st.session_state.get("recipient_email", "").strip()

                    if auto_email_csv and latest_csv["display_status"] == "Dangerous Situation":
                        if recipient:
                            email_payload = build_email_payload(latest_csv)
                            success, feedback = _send_dispatch_email(
                                latest_csv,
                                dispatch_log_path,
                                recipient_email=recipient,
                                source="csv_patient_check_auto",
                            )
                            st.session_state["dispatch_feedback"] = {
                                "success": success, "message": feedback
                            }
                        else:
                            st.session_state["dispatch_feedback"] = {
                                "success": False,
                                "message": "Dangerous Situation detected, but no recipient email is configured.",
                            }
                    elif auto_email_csv:
                        st.session_state["dispatch_feedback"] = {
                            "success": True,
                            "message": "Result is Normal — no automatic email sent.",
                        }
                except Exception as exc:
                    st.session_state["manual_prediction"] = None
                    st.session_state["dispatch_feedback"] = {
                        "success": False, "message": str(exc)
                    }
                    st.exception(exc)

    # Results display
    manual_prediction = st.session_state.get("manual_prediction")
    if manual_prediction is not None:
        latest_manual = manual_prediction["latest_prediction"]
        _render_status_card(
            "Predicted Status",
            latest_manual["display_status"],
            f"Clinical state: {latest_manual['clinical_state']} | Action: {latest_manual['dispatch_action']}",
        )
        quick_cols = st.columns(4)
        quick_cols[0].metric("Future Risk", f"{latest_manual['future_risk']:.3f}")
        quick_cols[1].metric("24h Score", f"{latest_manual['svm_prob_24h']:.3f}")
        quick_cols[2].metric("Critical Membership", f"{latest_manual['prob_critical']:.3f}")
        quick_cols[3].metric("Anomaly Flag", int(latest_manual["anomaly_flag"]))

        manual_view_cols = st.columns([1.1, 1.3])
        with manual_view_cols[0]:
            st.write("Reason")
            st.info(latest_manual["alert_reason"])
            preview = build_email_payload(latest_manual)
            if preview.get("gemini_used") == "True":
                st.success("Gemini generated this email body.")
            else:
                st.warning(f"Using fallback template. Gemini reason: {preview.get('gemini_error', 'unknown')}")
            st.write("Email preview")
            st.code(preview["subject"] + "\n\n" + preview["body"], language="text")
        with manual_view_cols[1]:
            st.write("Prediction timeline across uploaded hours")
            st.line_chart(
                manual_prediction["timeline"].set_index("hour")[
                    ["future_risk", "svm_prob_24h", "prob_critical"]
                ]
            )

# if manual_submit:
#     try:
#         manual_prediction = predict_manual_case(
#             results,
#             manual_feature_values,
#             patient_id=manual_patient_id.strip() or "manual_case",
#             history_len=int(manual_history_len),
#         )

#         st.session_state["manual_prediction"] = manual_prediction
#         latest_manual = manual_prediction["latest_prediction"]

#         recipient = st.session_state.get("recipient_email", "").strip()

#         if auto_email_dangerous and latest_manual["display_status"] == "Dangerous Situation":
#             if recipient:
#                 success, feedback = _send_dispatch_email(
#                     latest_manual,
#                     dispatch_log_path,
#                     recipient_email=recipient,
#                     source="manual_patient_check_auto_dangerous",
#                 )
#                 st.session_state["dispatch_feedback"] = {
#                     "success": success,
#                     "message": feedback,
#                 }
#             else:
#                 st.session_state["dispatch_feedback"] = {
#                     "success": False,
#                     "message": "Dangerous Situation detected, but no email address was entered.",
#                 }

#         elif auto_email_dangerous and latest_manual["display_status"] != "Dangerous Situation":
#             st.session_state["dispatch_feedback"] = {
#                 "success": True,
#                 "message": "Result is Normal, so no automatic email was sent.",
#             }

#     except Exception as exc:
#         st.session_state["manual_prediction"] = None
#         st.session_state["dispatch_feedback"] = {
#             "success": False,
#             "message": str(exc),
#         }
#     manual_prediction = st.session_state.get("manual_prediction")
#     if manual_prediction is not None:
#         latest_manual = manual_prediction["latest_prediction"]
#         _render_status_card(
#             "Predicted Status",
#             latest_manual["display_status"],
#             f"Clinical state: {latest_manual['clinical_state']} | Action: {latest_manual['dispatch_action']}",
#         )
#         quick_cols = st.columns(4)
#         quick_cols[0].metric("Future Risk", f"{latest_manual['future_risk']:.3f}")
#         quick_cols[1].metric("24h Score", f"{latest_manual['svm_prob_24h']:.3f}")
#         quick_cols[2].metric("Critical Membership", f"{latest_manual['prob_critical']:.3f}")
#         quick_cols[3].metric("Anomaly Flag", int(latest_manual["anomaly_flag"]))

#         manual_view_cols = st.columns([1.1, 1.3])
#         with manual_view_cols[0]:
#             st.write("Reason")
#             st.info(latest_manual["alert_reason"])
#             if latest_manual["display_status"] == "Dangerous Situation":
#                 st.caption("This result is configured to auto-send an email as soon as you submit the form.")
#             else:
#                 st.caption("This result is Normal, so auto-email is skipped unless you choose to extend that behavior later.")
#             st.write("preview")
#             preview = build_email_payload(latest_manual)
#             st.code(preview["subject"] + "\n\n" + preview["body"], language="text")
#             if preview.get("gemini_used") == "True":
#                 st.success("Gemini generated this email body.")
#             else:
#                 st.warning(f"Using fallback email body. Gemini reason: {preview.get('gemini_error', 'unknown')}")
#         with manual_view_cols[1]:
#             st.write("Manual prediction timeline")
#             st.line_chart(
#                 manual_prediction["timeline"].set_index("hour")[["future_risk", "svm_prob_24h", "prob_critical"]]
#             )

row1 = st.columns(3)
row2 = st.columns(3)

with row1[0]:
    with st.container(border=True):
        st.subheader("Zone 1 Cluster Health")
        zone1 = zones["zone_1_cluster_health"]
        z1_cols = st.columns(3)
        z1_cols[0].metric("Anomaly Count", int(results["stage_outputs"]["dbscan"]["anomaly_count"]))
        z1_cols[1].metric("Critical Mean", f"{results['stage_outputs']['fcm']['prob_critical_mean']:.3f}")
        z1_cols[2].metric("Critical Patients", int((latest_status["cluster_state"] == "Critical").sum()))
        st.bar_chart(zone1["cluster_counts"].set_index("cluster_state"))
        st.line_chart(zone1["timeline"].set_index("hour")[["prob_stable", "prob_warning", "prob_critical"]])
        st.dataframe(zone1["critical_patients"].head(10), width="stretch")

with row1[1]:
    with st.container(border=True):
        st.subheader("Zone 2 Live Scoring Feed")
        if selected_status is not None:
            z2_cols = st.columns(3)
            z2_cols[0].metric("Patient State", selected_status["clinical_state"])
            z2_cols[1].metric("Intelligent Score", f"{float(selected_status.get('intelligent_score', selected_status['svm_prob_24h'])):.3f}")
            z2_cols[2].metric("24h Score", f"{float(selected_status['svm_prob_24h']):.3f}")
        if not selected_history.empty:
            st.line_chart(selected_history.set_index("hour")[["future_risk", "svm_prob_24h"]])
        st.dataframe(zones["zone_2_live_scoring_feed"]["feed"].head(20), width="stretch")

with row1[2]:
    with st.container(border=True):
        st.subheader("Zone 3 Action Dispatch")
        zone3 = zones["zone_3_action_dispatch"]
        if selected_status is not None:
            _render_status_card(
                "Dispatch State",
                selected_status["binary_status"],
                f"Clinical state: {selected_status['clinical_state']} | Recommended action: {selected_status['dispatch_action']}",
            )
            st.write(f"Reason: {selected_status['alert_reason']}")
            if "react_route" in selected_status:
                st.caption(
                    f"ReAct route: {selected_status['react_route']} | "
                    f"timing: {float(selected_status.get('action_timing_minutes', 0)):.0f} min | "
                    f"intensity: {float(selected_status.get('action_intensity', 0)):.3f}"
                )
                st.info(str(selected_status.get("action_personalisation", "")))
            email_preview = build_email_payload(selected_status.to_dict())
            st.code(email_preview["subject"] + "\n\n" + email_preview["body"], language="text")
            if email_preview.get("gemini_used") == "True":
                st.success("Gemini generated this email body.")
            else:
                st.warning(f"Using fallback email body. Gemini reason: {email_preview.get('gemini_error', 'unknown')}")
            if st.button("Send status email now", width="stretch"):
                success, feedback = _send_dispatch_email(selected_status.to_dict(), dispatch_log_path, source="zone_3_dispatch")
                st.session_state["dispatch_feedback"] = {"success": success, "message": feedback}
                st.rerun()
        else:
            st.info("No patient is currently selected.")
        st.write("Action queue")
        st.dataframe(zone3["full_queue"].head(12), width="stretch")
        st.write("Dispatch history")
        st.dataframe(_load_log_frame(dispatch_log_path, tail=12), width="stretch")

with row2[0]:
    with st.container(border=True):
        st.subheader("Zone 4 SHAP Reliability")
        explanation = zones["zone_4_shap_reliability"]["explanation"]
        z4_cols = st.columns(3)
        z4_cols[0].metric("Method", explanation["method"])
        z4_cols[1].metric("Confidence", f"{float(explanation['confidence']):.3f}")
        z4_cols[2].metric("Margin to Threshold", f"{float(explanation['margin_to_threshold']):.3f}")
        if selected_status is not None:
            z4_live = st.columns(3)
            z4_live[0].metric("SHAP Cosine", f"{float(selected_status.get('shap_cosine', 0)):.3f}")
            z4_live[1].metric("Dominant Feature", str(selected_status.get("dominant_feature", "N/A")))
            z4_live[2].metric("Implicit RAG", "Cluster SHAP mean")
        if explanation["patient_id"] is not None:
            st.write(f"Explained patient: {explanation['patient_id']} at hour {explanation['hour']}")
            st.write(f"Clinical note provider: {explanation['note_provider']}")
        st.dataframe(explanation["feature_table"], width="stretch")
        st.info(results["clinical_note"])

with row2[1]:
    with st.container(border=True):
        st.subheader("Zone 5 Outcome Tracking")
        zone5 = zones["zone_5_outcome_tracking"]
        z5_cols = st.columns(4)
        z5_cols[0].metric("Precision", f"{zone5['precision']:.3f}")
        z5_cols[1].metric("Recall", f"{zone5['recall']:.3f}")
        z5_cols[2].metric("FPR", f"{metrics['fpr']:.3f}")
        z5_cols[3].metric("Threshold", f"{metrics['threshold_last']:.3f}")
        st.line_chart(zone5["hourly"].set_index("hour")[["predicted_alerts", "true_deteriorations", "mean_score"]])
        st.write("Confusion matrix")
        st.dataframe(pd.DataFrame(zone5["confusion_matrix"]), width="stretch")

with row2[2]:
    with st.container(border=True):
        st.subheader("Zone 6 System Alerts")
        zone6 = zones["zone_6_system_alerts"]
        z6_cols = st.columns(3)
        z6_cols[0].metric("Alerted Patients", int(len(zone6["alerts"])))
        z6_cols[1].metric("Adaptations", int(len(zone6["adaptations"])))
        z6_cols[2].metric("Agent Messages", int(len(zone6["agent_messages"])))
        drift_report = zone6.get("drift_report", {})
        if drift_report:
            st.caption(
                f"Drift gate: gap decline={drift_report.get('gap_trend_decline')} | "
                f"centroid shift={drift_report.get('centroid_shift')} | "
                f"outer loop triggered={drift_report.get('triggered')} | "
                f"rollback={drift_report.get('rollback_gate')}"
            )
        agentic_properties = zone6.get("agentic_properties", {})
        if agentic_properties:
            st.write("Agentic property evidence")
            st.json(agentic_properties)
        st.write("Current alert queue")
        st.dataframe(zone6["alerts"].head(12), width="stretch")
        if not zone6.get("react_demonstrations", pd.DataFrame()).empty:
            st.write("Seven ReAct routing demonstrations")
            st.dataframe(zone6["react_demonstrations"], width="stretch")
        if not zone6["adaptations"].empty:
            st.write("Threshold adaptation history")
            st.line_chart(zone6["adaptations"].set_index("count")[["old_threshold", "new_threshold", "fpr"]])
        st.write("Recent agent messages")
        st.dataframe(zone6["agent_messages"].tail(12), width="stretch")

footer_cols = st.columns(3)
with footer_cols[0]:
    st.write("Weight log")
    st.dataframe(_load_log_frame(weight_log_path, tail=10), width="stretch")
with footer_cols[1]:
    st.write("Agent log")
    st.dataframe(_load_log_frame(agent_log_path, tail=10), width="stretch")
with footer_cols[2]:
    predictions_path = Path(results["paths"]["predictions_csv"])
    if predictions_path.exists():
        st.download_button(
            "Download predictions CSV",
            data=predictions_path.read_bytes(),
            file_name="predictions.csv",
            mime="text/csv",
            width="stretch",
        )
