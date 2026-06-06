"""Train and persist the stellar classification model (matches AIProjectHJ.ipynb)."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.combine import SMOTETomek
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = ROOT / "dataset" / "star_classification.csv"
MODEL_DIR = ROOT / "model"

NOISE_COLUMNS = [
    "obj_ID",
    "spec_obj_ID",
    "run_ID",
    "rerun_ID",
    "cam_col",
    "field_ID",
    "plate",
    "MJD",
    "fiber_ID",
]

FEATURE_COLUMNS = ["alpha", "delta", "u", "g", "r", "i", "z", "redshift"]


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = df.copy()
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        q1 = df_clean[col].quantile(0.25)
        q3 = df_clean[col].quantile(0.75)
        iqr = q3 - q1
        df_clean = df_clean[
            (df_clean[col] >= q1 - 1.5 * iqr) & (df_clean[col] <= q3 + 1.5 * iqr)
        ]
    return df_clean


def main() -> None:
    df = pd.read_csv(DATASET_PATH)
    df = df.drop(columns=NOISE_COLUMNS)
    df_clean = remove_outliers(df)

    x = df_clean[FEATURE_COLUMNS]
    y = df_clean["class"]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    smote_tomek = SMOTETomek(random_state=42)
    x_train_res, y_train_res = smote_tomek.fit_resample(x_train_scaled, y_train)

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(x_train_res, y_train_res)

    accuracy = model.score(x_test_scaled, y_test)
    print(f"Test accuracy: {accuracy:.4f}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_DIR / "stellar_classifier.pkl")
    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
    joblib.dump(
        {
            "feature_columns": FEATURE_COLUMNS,
            "classes": list(model.classes_),
            "accuracy": float(accuracy),
            "model_type": "RandomForestClassifier",
        },
        MODEL_DIR / "model_metadata.pkl",
    )
    print(f"Model saved to {MODEL_DIR}")


if __name__ == "__main__":
    main()
