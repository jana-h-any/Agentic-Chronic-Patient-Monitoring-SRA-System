import argparse
import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd


def load_data(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Primary dataset CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    return _ensure_targets(df)


def build_real_eicu_csv(
    eicu_path: str,
    output_csv: str,
    max_hours: int = 72,
    patient_limit: Optional[int] = None,
) -> pd.DataFrame:
    df = _load_from_eicu(eicu_path=eicu_path, max_hours=max_hours, patient_limit=patient_limit)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    return df


def _read_csv_any(path: Path, usecols=None, chunksize: Optional[int] = None):
    gz_path = path if path.exists() else path.with_suffix(path.suffix + ".gz")
    if not gz_path.exists():
        raise FileNotFoundError(str(gz_path))
    return pd.read_csv(gz_path, usecols=usecols, chunksize=chunksize, low_memory=False)


def _build_hourly_grid(patient_df: pd.DataFrame, max_hours: int) -> pd.DataFrame:
    rows = []
    for _, row in patient_df.iterrows():
        max_stay_hours = int(max(0, pd.to_numeric(row["unitdischargeoffset"], errors="coerce") or 0) // 60)
        upper = min(max_hours, max_stay_hours)
        rows.extend([(str(row["patientunitstayid"]), hour) for hour in range(upper + 1)])
    return pd.DataFrame(rows, columns=["patient_id", "hour"])


def _normalize_hour(offset_series: pd.Series) -> pd.Series:
    return (pd.to_numeric(offset_series, errors="coerce") // 60).astype("Int64")


def _aggregate_vital_periodic(path: Path, stay_ids: set[str], max_hours: int) -> pd.DataFrame:
    cols = ["patientunitstayid", "observationoffset", "temperature", "sao2", "heartrate", "respiration"]
    df = _read_csv_any(path / "vitalPeriodic.csv", usecols=cols)
    if hasattr(df, "__iter__") and not isinstance(df, pd.DataFrame):
        df = pd.concat(list(df), ignore_index=True)

    df["patient_id"] = df["patientunitstayid"].astype(str)
    df = df[df["patient_id"].isin(stay_ids)].copy()
    df["hour"] = _normalize_hour(df["observationoffset"])
    df = df.dropna(subset=["hour"])
    df["hour"] = df["hour"].astype(int)
    df = df[(df["hour"] >= 0) & (df["hour"] <= max_hours)]

    rename_map = {
        "heartrate": "HR",
        "sao2": "SpO2",
        "respiration": "respiration",
        "temperature": "temp",
    }
    for src in rename_map:
        df[src] = pd.to_numeric(df[src], errors="coerce")

    agg = df.groupby(["patient_id", "hour"])[list(rename_map.keys())].mean().reset_index()
    return agg.rename(columns=rename_map)


def _aggregate_vital_aperiodic(path: Path, stay_ids: set[str], max_hours: int) -> pd.DataFrame:
    cols = ["patientunitstayid", "observationoffset", "noninvasivesystolic"]
    df = _read_csv_any(path / "vitalAperiodic.csv", usecols=cols)
    if hasattr(df, "__iter__") and not isinstance(df, pd.DataFrame):
        df = pd.concat(list(df), ignore_index=True)

    df["patient_id"] = df["patientunitstayid"].astype(str)
    df = df[df["patient_id"].isin(stay_ids)].copy()
    df["hour"] = _normalize_hour(df["observationoffset"])
    df = df.dropna(subset=["hour"])
    df["hour"] = df["hour"].astype(int)
    df = df[(df["hour"] >= 0) & (df["hour"] <= max_hours)]
    df["noninvasivesystolic"] = pd.to_numeric(df["noninvasivesystolic"], errors="coerce")

    agg = df.groupby(["patient_id", "hour"])["noninvasivesystolic"].mean().reset_index()
    return agg.rename(columns={"noninvasivesystolic": "BP"})


def _match_lab_name(name: str) -> Optional[str]:
    text = str(name).lower()
    if "creatinine" in text:
        return "creatinine"
    if "wbc" in text or "white blood cell" in text:
        return "WBC"
    if "lactate" in text:
        return "lactate"
    return None


def _aggregate_labs(path: Path, stay_ids: set[str], max_hours: int) -> pd.DataFrame:
    cols = ["patientunitstayid", "labresultoffset", "labname", "labresult"]
    df = _read_csv_any(path / "lab.csv", usecols=cols)
    if hasattr(df, "__iter__") and not isinstance(df, pd.DataFrame):
        df = pd.concat(list(df), ignore_index=True)

    df["patient_id"] = df["patientunitstayid"].astype(str)
    df = df[df["patient_id"].isin(stay_ids)].copy()
    df["hour"] = _normalize_hour(df["labresultoffset"])
    df = df.dropna(subset=["hour"])
    df["hour"] = df["hour"].astype(int)
    df = df[(df["hour"] >= 0) & (df["hour"] <= max_hours)]
    df["labresult"] = pd.to_numeric(df["labresult"], errors="coerce")
    df["var"] = df["labname"].map(_match_lab_name)
    df = df.dropna(subset=["labresult", "var"])

    agg = df.groupby(["patient_id", "hour", "var"])["labresult"].mean().reset_index()
    if agg.empty:
        return pd.DataFrame(columns=["patient_id", "hour", "creatinine", "WBC", "lactate"])
    wide = agg.pivot_table(index=["patient_id", "hour"], columns="var", values="labresult", aggfunc="mean").reset_index()
    return wide


def _ensure_targets(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = ["patient_id", "hour", "HR", "BP", "SpO2", "respiration", "temp", "creatinine", "WBC", "lactate"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    df = df.copy()
    for c in ["HR", "BP", "SpO2", "respiration", "temp", "creatinine", "WBC", "lactate"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["hour"] = pd.to_numeric(df["hour"], errors="coerce").fillna(0).astype(int)
    df["patient_id"] = df["patient_id"].astype(str)

    if "deterioration_12h" in df.columns and "deterioration_24h" in df.columns:
        return df

    event = (
        (df["HR"] > 120)
        | (df["BP"] < 90)
        | (df["SpO2"] < 90)
        | (df["respiration"] > 30)
        | (df["temp"] > 39)
        | (df["lactate"] > 2.5)
        | (df["WBC"] > 12)
        | (df["creatinine"] > 2.0)
    ).fillna(False).astype(int)
    df["_event"] = event

    def future_max(g: pd.Series, horizon: int) -> pd.Series:
        return g.shift(-1)[::-1].rolling(horizon, min_periods=1).max()[::-1].fillna(0).astype(int)

    df["deterioration_12h"] = df.groupby("patient_id")["_event"].transform(lambda s: future_max(s, 12))
    df["deterioration_24h"] = df.groupby("patient_id")["_event"].transform(lambda s: future_max(s, 24))
    return df.drop(columns=["_event"])


def _load_from_eicu(eicu_path: str, max_hours: int = 72, patient_limit: Optional[int] = None) -> pd.DataFrame:
    root = Path(eicu_path)
    patient = _read_csv_any(root / "patient.csv", usecols=["patientunitstayid", "unitdischargeoffset"])
    if hasattr(patient, "__iter__") and not isinstance(patient, pd.DataFrame):
        patient = pd.concat(list(patient), ignore_index=True)

    patient["patientunitstayid"] = patient["patientunitstayid"].astype(str)
    patient = patient.dropna(subset=["patientunitstayid"]).copy()
    if patient_limit is not None:
        patient = patient.drop_duplicates(subset=["patientunitstayid"]).head(patient_limit).copy()
    else:
        patient = patient.drop_duplicates(subset=["patientunitstayid"]).copy()

    stay_ids = set(patient["patientunitstayid"].tolist())
    grid = _build_hourly_grid(patient, max_hours=max_hours)

    periodic = _aggregate_vital_periodic(root, stay_ids, max_hours=max_hours)
    aperiodic = _aggregate_vital_aperiodic(root, stay_ids, max_hours=max_hours)
    labs = _aggregate_labs(root, stay_ids, max_hours=max_hours)

    df = grid.merge(periodic, on=["patient_id", "hour"], how="left")
    df = df.merge(aperiodic, on=["patient_id", "hour"], how="left")
    df = df.merge(labs, on=["patient_id", "hour"], how="left")

    for col in ["HR", "BP", "SpO2", "respiration", "temp", "creatinine", "WBC", "lactate"]:
        if col not in df.columns:
            df[col] = np.nan

    df = _ensure_targets(df)
    return df


def _parse_args():
    parser = argparse.ArgumentParser(description="Build a real project CSV from raw eICU tables.")
    parser.add_argument("--eicu-path", required=True, help="Path to raw eICU folder")
    parser.add_argument("--output-csv", required=True, help="Output CSV path for the extracted cohort")
    parser.add_argument("--max-hours", type=int, default=72, help="Max ICU hours per stay to export")
    parser.add_argument("--patient-limit", type=int, default=None, help="Optional patient cap for faster extraction")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    df = build_real_eicu_csv(
        eicu_path=args.eicu_path,
        output_csv=args.output_csv,
        max_hours=args.max_hours,
        patient_limit=args.patient_limit,
    )
    print(f"Wrote {len(df)} rows to {args.output_csv}")
