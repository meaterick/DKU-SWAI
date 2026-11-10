from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
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
DERIVED_FEATURE = "SYMPTOM_COUNT"
MODEL_FEATURES = [*FEATURES, DERIVED_FEATURE]


def load_survey(path: Path) -> tuple[pd.DataFrame, pd.Series, dict[str, int]]:
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


def add_symptom_count(frame: pd.DataFrame) -> pd.DataFrame:
    featured = frame.copy()
    featured[DERIVED_FEATURE] = featured[BINARY_FEATURES].sum(axis=1)
    return featured


def evaluate_model(model, frame: pd.DataFrame, y: pd.Series) -> dict:
    predictions = model.predict(frame)
    probabilities = model.predict_proba(frame)[:, 1]
    return {
        "accuracy": float(accuracy_score(y, predictions)),
        "precision_yes": float(precision_score(y, predictions, zero_division=0)),
        "recall_yes": float(recall_score(y, predictions, zero_division=0)),
        "f1_yes": float(f1_score(y, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "confusion_matrix_no_yes": confusion_matrix(y, predictions, labels=[0, 1]).tolist(),
    }


class LungCancerTrainer:
    def __init__(self, data_path: Path, output_dir: Path, test_size: float = 0.2):
        if not 0 < test_size < 1:
            raise ValueError("test_size는 0과 1 사이여야 합니다.")
        self.data_path = data_path
        self.output_dir = output_dir
        self.test_size = test_size

    def build_baseline(self):
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        )

    def build_candidates(self) -> dict:
        return {
            "LogisticRegression": make_pipeline(
                StandardScaler(),
                LogisticRegression(C=1.0, max_iter=3000, random_state=42),
            ),
            "KNN(k=5)": make_pipeline(
                StandardScaler(),
                KNeighborsClassifier(n_neighbors=5),
            ),
            "RandomForest": RandomForestClassifier(
                n_estimators=200,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=42,
            ),
            "GradientBoosting": GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=2,
                min_samples_leaf=4,
                subsample=0.9,
                random_state=42,
            ),
        }

    def select_model(self, frame: pd.DataFrame, y: pd.Series) -> tuple[str, dict]:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scoring = {
            "accuracy": "accuracy",
            "f1": "f1",
            "roc_auc": "roc_auc",
            "recall": "recall",
        }
        results = {}
        for name, model in self.build_candidates().items():
            scores = cross_validate(model, frame, y, cv=cv, scoring=scoring, n_jobs=1)
            results[name] = {
                "accuracy_mean": float(scores["test_accuracy"].mean()),
                "accuracy_std": float(scores["test_accuracy"].std()),
                "f1_mean": float(scores["test_f1"].mean()),
                "roc_auc_mean": float(scores["test_roc_auc"].mean()),
                "recall_mean": float(scores["test_recall"].mean()),
            }
        best_name = max(
            results,
            key=lambda name: (results[name]["accuracy_mean"], results[name]["f1_mean"]),
        )
        return best_name, results

    def build_model(self, name: str):
        return self.build_candidates()[name]

    def run(self) -> dict:
        X, y, counts = load_survey(self.data_path)
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=42, stratify=y
        )

        baseline = self.build_baseline()
        baseline.fit(X_train_raw, y_train)
        baseline_metrics = evaluate_model(baseline, X_test_raw, y_test)

        X_train = add_symptom_count(X_train_raw)
        X_test = add_symptom_count(X_test_raw)
        selected_name, cv_results = self.select_model(X_train, y_train)
        model = self.build_model(selected_name)
        model.fit(X_train, y_train)
        selected_metrics = evaluate_model(model, X_test, y_test)

        metrics = {
            "source": str(self.data_path),
            "rows_after_deduplication": len(X),
            "class_counts": counts,
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "test_size": self.test_size,
            "random_state": 42,
            "features": MODEL_FEATURES,
            "feature_engineering": "SYMPTOM_COUNT = 13개 설문 응답(1=예)의 합",
            "model_selection": {
                "method": "StratifiedKFold",
                "splits": 5,
                "selection_metric": "mean accuracy (tie-break: mean F1)",
                "selected_model": selected_name,
                "candidates": cv_results,
            },
            "baseline_model": "LogisticRegression(class_weight='balanced') · 원본 15개 특성",
            "baseline": baseline_metrics,
            "selected_model": selected_name,
            **selected_metrics,
        }
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        joblib.dump(
            {"model": model, "features": MODEL_FEATURES, "selected_model": selected_name},
            self.output_dir / "model.joblib",
        )
        return metrics


def train(data_path: Path, output_dir: Path, test_size: float = 0.2) -> dict:
    return LungCancerTrainer(data_path, output_dir, test_size).run()


def main() -> None:
    parser = argparse.ArgumentParser(description="폐암 설문 데이터 분류 모델 학습")
    parser.add_argument("--data", type=Path, default=Path("data/lung_cancer.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()
    print(json.dumps(train(args.data, args.output_dir, args.test_size), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
