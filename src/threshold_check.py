from __future__ import annotations

import argparse
import json
from pathlib import Path

from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

from train import LungCancerTrainer, add_symptom_count, load_survey


def scan_thresholds(y, probabilities, thresholds: list[float]) -> list[dict]:
    rows = []
    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, predictions, labels=[0, 1]).ravel()
        rows.append(
            {
                "threshold": threshold,
                "precision": round(float(precision_score(y, predictions, zero_division=0)), 4),
                "recall": round(float(recall_score(y, predictions, zero_division=0)), 4),
                "f1": round(float(f1_score(y, predictions, zero_division=0)), 4),
                "false_negative": int(fn),
                "false_positive": int(fp),
            }
        )
    return rows


def recommend(rows: list[dict], min_recall: float = 0.95) -> float:
    safe = [row for row in rows if row["recall"] >= min_recall]
    pool = safe if safe else rows
    best = max(pool, key=lambda row: (row["f1"], row["threshold"]))
    return best["threshold"]


def check(data_path: Path, artifacts: Path, model_name: str, test_size: float = 0.2) -> dict:
    X, y, _ = load_survey(data_path)
    X_train, _, y_train, _ = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    X_train = add_symptom_count(X_train)
    model = LungCancerTrainer(data_path, artifacts, test_size).build_model(model_name)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    probabilities = cross_val_predict(model, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    thresholds = [round(0.30 + 0.05 * step, 2) for step in range(11)]
    rows = scan_thresholds(y_train, probabilities, thresholds)
    report = {
        "model": model_name,
        "method": "학습 데이터 220건 5-fold out-of-fold 확률 (평가 데이터 미사용)",
        "rule": "재현율 0.95 이상인 임계값 중 F1이 가장 큰 값",
        "recommended_threshold": recommend(rows),
        "thresholds": rows,
    }
    artifacts.mkdir(parents=True, exist_ok=True)
    (artifacts / "threshold_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="판정 임계값별 정밀도·재현율 점검")
    parser.add_argument("--data", type=Path, default=Path("data/lung_cancer.csv"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    metrics = json.loads((args.artifacts / "metrics.json").read_text(encoding="utf-8"))
    report = check(args.data, args.artifacts, metrics["selected_model"], metrics["test_size"])
    print(f"{'임계값':>6} {'정밀도':>7} {'재현율':>7} {'F1':>7} {'놓침':>4} {'오탐':>4}")
    for row in report["thresholds"]:
        print(
            f"{row['threshold']:>8.2f} {row['precision']:>9.3f} {row['recall']:>9.3f} "
            f"{row['f1']:>7.3f} {row['false_negative']:>6} {row['false_positive']:>6}"
        )
    print(f"F1 최대 임계값: {report['recommended_threshold']} (재현율 0.95 이상인 경우 중). 기본값은 0.5")


if __name__ == "__main__":
    main()
