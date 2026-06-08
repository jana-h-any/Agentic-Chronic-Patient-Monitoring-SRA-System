# 🏥 Agentic Chronic Patient Monitoring — SRA System

> **Autonomous AI system for early detection of ICU patient deterioration using Agentic AI, ReAct reasoning, RAG memory, and self-retraining.**

---

# Overview

The **Agentic Chronic Patient Monitoring SRA System** continuously monitors ICU patients and autonomously performs:

* **Observation**
* **Reasoning**
* **Action Selection**
* **Explanation Generation**
* **Drift Detection**
* **Self-Retraining**

without requiring human intervention.

The system combines:

* Multi-Agent Architecture
* ReAct Decision Loops
* Explicit + Implicit RAG
* Dynamic SVM
* Fuzzy C-Means Clustering
* SHAP Explainability
* ARIMA Trend Modeling
* Gemini LLM Integration

---

# Clinical Signals

The system monitors eight physiological variables:

| Signal      | Description                |
| ----------- | -------------------------- |
| HR          | Heart Rate                 |
| BP          | Blood Pressure             |
| SpO₂        | Oxygen Saturation          |
| Respiration | Respiratory Rate           |
| Temp        | Temperature                |
| Creatinine  | Kidney Function Indicator  |
| WBC         | White Blood Cell Count     |
| Lactate     | Tissue Perfusion Indicator |

---

# System Architecture

---

## Offline Training Cycle *(Executed Once)*

### 1. DBSCAN

Performs anomaly detection in feature space.

### 2. Fuzzy C-Means (FCM)

Clusters patients into:

* Stable
* Warning
* Critical

Cluster centroids are frozen after training.

### 3. Predictive Action Layer

Ensemble model forecasting **12-hour deterioration risk**:

```python
future_risk
```

### 4. Dynamic SVM

Predicts **24-hour deterioration probability** using adaptive sample weights derived from `future_risk`.

### 5. SHAP Analysis

Cluster-average SHAP vectors become the system's **implicit memory**.

### 6. KNN RAG Index

Provides explicit retrieval using:

* Patient features
* SHAP vectors

### 7. ARIMA Baselines

Creates cluster-specific trend models.

---

# Live Prediction Cycle

Whenever a new patient arrives:

### Step 1

Apply saved FCM centroids to obtain fuzzy memberships.

### Step 2

Generate:

* `future_risk`
* `svm_prob_24h`

### Step 3

Compute the **Intelligent Score**

The score combines five independent signals:

| Signal          | Description                        |
| --------------- | ---------------------------------- |
| **S**           | Dynamic severity                   |
| **Gap(SVM)**    | Confidence distance from threshold |
| **μ**           | Fuzzy cluster membership           |
| **ARIMA**       | Trend multiplier                   |
| **SHAP Cosine** | Reliability indicator              |

### Dynamic Severity

```text
S = 0.72 × cluster_base + 0.28 × future_risk
```

---

### Step 4 — ReAct Inner Loop

Observe → Reason → Act

Possible routing branches:

* score_below_minimum
* direct_dispatch
* shap_recheck_dispatch
* shap_recheck_human_review
* rag_majority_arima_full_trust
* rag_majority_arima_dampened_review
* rag_tie
* rag_zero_success

---

### Step 5 — Gemini LLM

Generates:

* Clinical notes
* Professional alert emails

using SHAP-derived context.

---

# Drift Detection & Outer Loop

Retraining occurs only when **two conditions fire simultaneously**.

---

## 1. Confidence Degradation

Classifier confidence drops below:

```text
85% of baseline
```

---

## 2. Centroid Movement

FCM centroid displacement exceeds:

```text
0.35
```

---

## Retraining Pipeline

```text
Re-cluster
      ↓
Retrain models
      ↓
Recompute SHAP
      ↓
Rebuild RAG
      ↓
Reset ARIMA
      ↓
Rollback Gate
```

Rollback condition:

```text
Accuracy ≥ Baseline − 3%
```

---

# Intelligent Score

Five-signal composite score providing robustness against single-point failures.

| Component   | Purpose                    |
| ----------- | -------------------------- |
| S           | Dynamic severity           |
| Gap(SVM)    | Classifier confidence      |
| μ           | Fuzzy certainty            |
| ARIMA       | Temporal behavior          |
| SHAP Cosine | Explainability reliability |

---

# Multi-Agent System

Four specialized agents communicate through a MessageBus.

```text
VitalsAgent
      ↓
TrendAgent
      ↓
RiskAgent
      ↓
AlertAgent
```

### VitalsAgent

Monitors physiological measurements.

### TrendAgent

Tracks temporal behavior and ARIMA trends.

### RiskAgent

Computes deterioration risk.

### AlertAgent

Dispatches explanations and notifications.

---

# Project Structure

```text
├── main.py
├── config.yaml
├── 02_src
│   ├── pipeline_runner.py
│   ├── action_dispatch.py
│   ├── agents.py
│   ├── llm_narrative.py
│   ├── models.py
│   ├── preprocessing.py
│   └── utils.py
├── 03_dashboard
│   └── streamlit_app.py
└── data
```

---

# Installation

```bash
git clone https://github.com/jana-h-any/Agentic-Chronic-Patient-Monitoring-SRA-System

cd Agentic-Chronic-Patient-Monitoring-SRA-System

pip install -r requirements.txt

export GEMINI_API_KEY="your_key_from_aistudio.google.com"
```

---

# Running

## Full Pipeline

```bash
python main.py --patient-limit 200
```

## Streamlit Dashboard

```bash
streamlit run 03_dashboard/streamlit_app.py
```

## FastAPI Server

```bash
uvicorn main:app --reload --port 8000
```

Swagger:

```text
http://localhost:8000/docs
```

---

# API Endpoints

| Endpoint                    | Description            |
| --------------------------- | ---------------------- |
| POST `/api/training-cycle`  | Offline training cycle |
| POST `/api/live-cycle`      | Live prediction        |
| GET `/api/inner-react-loop` | ReAct decisions        |
| GET `/api/rag-layer`        | RAG status             |
| GET `/api/drift-outer-loop` | Drift reports          |
| GET `/api/dashboard-zones`  | Dashboard data         |

---

# Dashboard

### Zone 1 — Cluster Health

* Patient distribution
* Membership timeline

### Zone 2 — Live Scoring Feed

* Intelligent score
* Future risk
* SVM probability

### Zone 3 — Action Dispatch

* ReAct route
* Action timing
* Email status

### Zone 4 — SHAP Reliability

* Dominant feature
* Cosine similarity
* Gemini note

### Zone 5 — Outcome Tracking

* Precision
* Recall
* FPR
* Confusion matrix

### Zone 6 — System Alerts

* Drift signals
* Retraining events
* Agent communication

---

# Novel Contributions

| # | Contribution                                       |
| - | -------------------------------------------------- |
| 1 | Predictive Action Layer for adaptive weighting     |
| 2 | Dynamic severity computation with audit logs       |
| 3 | Multi-Agent architecture with MessageBus           |
| 4 | Pre-scoring triage using 12h risk forecasting      |
| 5 | Gemini-generated SHAP-informed clinical narratives |

---

# Six Agentic Properties

### 1. Continuous Observation

Agents monitor patients continuously.

### 2. Reasoning Before Acting

ReAct determines the appropriate branch.

### 3. Explainable Actions

Every decision includes a justification.

### 4. Adaptation

Thresholds and weights evolve according to outcomes.

### 5. Degradation Detection

Population and model drift are continuously monitored.

### 6. Autonomous Retraining

The complete outer loop executes automatically with rollback protection.

---

# Technologies

* Python
* FastAPI
* Streamlit
* Scikit-Learn
* Fuzzy C-Means
* DBSCAN
* SHAP
* ARIMA
* KNN
* Gemini API

---

# Chronic Patient Monitoring — Agentic SRA System

### Observe → Reason → Act → Adapt → Retrain
