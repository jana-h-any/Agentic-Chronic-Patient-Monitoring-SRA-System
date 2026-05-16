import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
import skfuzzy as fuzz
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from statsmodels.tsa.statespace.sarimax import SARIMAX

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover - environment-dependent fallback
    XGBClassifier = None

def run_dbscan(df, features, eps=0.5, min_samples=5):
    scaler = StandardScaler()
    X = scaler.fit_transform(df[features].astype(float))
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    flags = dbscan.fit_predict(X)
    df['anomaly_flag'] = np.where(flags == -1, 1, 0)
    return df

def _fcm_severity_order(centroids_scaled, features):
    weights = {
        "HR": 1.0,
        "BP": -1.0,
        "SpO2": -1.2,
        "respiration": 0.8,
        "temp": 0.5,
        "creatinine": 0.7,
        "WBC": 0.5,
        "lactate": 1.2,
    }
    weight_vector = np.asarray([weights.get(feature, 0.0) for feature in features], dtype=float)
    severity = np.asarray(centroids_scaled).dot(weight_vector)
    return np.argsort(severity)


def fit_fcm_model(df, features):
    scaler = StandardScaler()
    X = scaler.fit_transform(df[features].astype(float))
    data = X.T
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        data, c=3, m=2, error=0.005, maxiter=1000, init=None
    )
    order = _fcm_severity_order(cntr, features)
    ordered_u = u[order]
    ordered_centroids = cntr[order]
    out = df.copy()
    out['prob_stable'] = ordered_u[0]
    out['prob_warning'] = ordered_u[1]
    out['prob_critical'] = ordered_u[2]
    out['fcm_cluster'] = np.argmax(ordered_u, axis=0)
    artifact = {
        "features": list(features),
        "mean": scaler.mean_.astype(float),
        "scale": scaler.scale_.astype(float),
        "centroids_scaled": ordered_centroids.astype(float),
        "fpc": float(fpc),
        "labels": ["Stable", "Warning", "Critical"],
    }
    return out, artifact


def apply_fcm_model(df, fcm_artifact):
    features = list(fcm_artifact["features"])
    mean = np.asarray(fcm_artifact["mean"], dtype=float)
    scale = np.asarray(fcm_artifact["scale"], dtype=float)
    scale = np.where(scale == 0, 1.0, scale)
    centroids = np.asarray(fcm_artifact["centroids_scaled"], dtype=float)
    X = (df[features].astype(float).to_numpy() - mean) / scale
    distances = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
    distances = np.maximum(distances, 1e-9)
    inverse = 1.0 / distances
    memberships = inverse / inverse.sum(axis=1, keepdims=True)
    out = df.copy()
    out['prob_stable'] = memberships[:, 0]
    out['prob_warning'] = memberships[:, 1]
    out['prob_critical'] = memberships[:, 2]
    out['fcm_cluster'] = np.argmax(memberships, axis=1)
    return out


def run_fcm(df, features):
    clustered, _ = fit_fcm_model(df, features)
    return clustered

def train_predictive_action_layer(X, y, n_splits=3):
    if XGBClassifier is not None:
        boost = XGBClassifier(
            eval_metric='logloss',
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
        )
    else:
        # Keep the ensemble runnable when xgboost is unavailable in the local environment.
        boost = GradientBoostingClassifier(
            n_estimators=250,
            learning_rate=0.05,
            max_depth=3,
            random_state=42,
        )
    rf = RandomForestClassifier(
        n_estimators=400,
        random_state=42,
        n_jobs=-1,
    )
    ensemble = VotingClassifier(estimators=[('boost', boost), ('rf', rf)], voting='soft', n_jobs=-1)

    tscv = TimeSeriesSplit(n_splits=n_splits)
    oof = np.full(len(y), np.nan, dtype=float)
    for train_idx, test_idx in tscv.split(X):
        m = clone(ensemble)
        m.fit(X.iloc[train_idx], y.iloc[train_idx])
        oof[test_idx] = m.predict_proba(X.iloc[test_idx])[:, 1]
    ensemble.fit(X, y)
    if np.isnan(oof).any():
        oof[np.isnan(oof)] = ensemble.predict_proba(X.iloc[np.isnan(oof)])[:, 1]
    return ensemble, oof

def train_svm_dynamic(X, y, sample_weights):
    param_grid = {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']}
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel='rbf', probability=True)),
    ])
    grid = GridSearchCV(
        pipe,
        param_grid={"svm__C": param_grid["C"], "svm__gamma": param_grid["gamma"]},
        cv=3,
        scoring="roc_auc",
        n_jobs=-1,
    )
    grid.fit(X, y, svm__sample_weight=sample_weights)
    return grid.best_estimator_

def run_sarima(history_hr):
    try:
        model = SARIMAX(history_hr, order=(2,1,1), seasonal_order=(1,1,1,24))
        fitted = model.fit(disp=False)
        forecast = fitted.forecast(steps=1)[0]
        return forecast
    except Exception:
        return np.mean(history_hr) # Fallback if SARIMA fails to converge
