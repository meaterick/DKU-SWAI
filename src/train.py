"""Train and evaluate a baseline model on the Kaggle lung cancer survey."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


TARGET = "LUNG_CANCER"
BINARY_FEATURES = [
    "SMOKING",
    "YELLOW_FINGERS",
    "ANXIETY",
    "PEER_PRESSURE",
    "CHRONIC DISEASE",
    "FATIGUE",
    "ALLERGY",
    "WHEEZING",
    "ALCOHOL CONSUMING",
    "COUGHING",
    "SHORTNESS OF BREATH",
    "SWALLOWING DIFFICULTY",
    "CHEST PAIN",
]
FEATURES = ["GENDER", "AGE", *BINARY_FEATURES]


def load_survey(path: Path) -> tuple[pd.DataFrame, pd.Series, dict[str, int]]:
    """Validate the documented survey schema and return numeric features and labels."""
    frame = pd.read_csv(path)
    frame.columns = frame.columns.str.strip()
    if frame.columns.duplicated().any():
        raise ValueError("공백 제거 후 중복된 열 이름이 있습니다.")
    missing = sorted(set([*FEATURES, TARGET]) - set(frame.columns))
    if missing:
        raise ValueError(f"필수 열이 없습니다: {', '.join(missing)}")

    frame = frame[[*FEATURES, TARGET]].drop_duplicates().copy()
    if frame.empty or frame.isna().any().any():
        raise ValueError("데이터가 비어 있거나 결측값이 있습니다. 원본 CSV를 확인하세요.")

    gender = frame["GENDER"].astype(str).str.strip().str.upper()
    if not gender.isin(["F", "M"]).all():
        raise ValueError("GENDER는 F 또는 M이어야 합니다.")
    frame["GENDER"] = gender.map({"F": 0, "M": 1})

    age = pd.to_numeric(frame["AGE"], errors="raise")
    if not age.between(1, 120).all():
        raise ValueError("AGE는 1~120 사이의 숫자여야 합니다.")
    frame["AGE"] = age

    for column in BINARY_FEATURES:
        values = pd.to_numeric(frame[column], errors="raise")
        if not values.isin([1, 2]).all():
            raise ValueError(f"{column}은 1(아니오) 또는 2(예)여야 합니다.")
        frame[column] = values - 1

    labels = frame[TARGET].astype(str).str.strip().str.upper()
    if not labels.isin(["NO", "YES"]).all():
        raise ValueError("LUNG_CANCER는 NO 또는 YES여야 합니다.")
    y = labels.map({"NO": 0, "YES": 1}).astype(int)
    counts = {"NO": int((y == 0).sum()), "YES": int((y == 1).sum())}
    if min(counts.values()) < 2:
        raise ValueError("계층화 분할을 위해 각 클래스에 최소 2개 행이 필요합니다.")
    return frame[FEATURES], y, counts


def train(data_path: Path, output_dir: Path, test_size: float = 0.2) -> dict:
    """Fit a baseline with training-only scaling and save evaluation artifacts."""
    if not 0 < test_size < 1:
        raise ValueError("test_size는 0과 1 사이여야 합니다.")
    X, y, counts = load_survey(data_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "source": str(data_path),
        "rows_after_deduplication": len(X),
        "class_counts": counts,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "test_size": test_size,
        "random_state": 42,
        "accuracy": accuracy_score(y_test, predictions),
        "precision_yes": precision_score(y_test, predictions, zero_division=0),
        "recall_yes": recall_score(y_test, predictions, zero_division=0),
        "f1_yes": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "confusion_matrix_no_yes": confusion_matrix(
            y_test, predictions, labels=[0, 1]
        ).tolist(),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    joblib.dump({"model": model, "features": FEATURES}, output_dir / "model.joblib")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/lung_cancer.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()
    print(json.dumps(train(args.data, args.output_dir, args.test_size), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
