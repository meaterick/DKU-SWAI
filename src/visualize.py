from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import ListedColormap
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split

from train import FEATURES, add_symptom_count, load_survey

DARK = "#0B3C49"
TEAL = "#0B5563"
MINT = "#7FBFB0"
CORAL = "#E76F51"
GRAY = "#5B6B6E"
PINK = "#FBE7E1"
LABELS_KO = {
    "GENDER": "성별",
    "AGE": "나이",
    "SMOKING": "흡연",
    "YELLOW_FINGERS": "누런 손가락",
    "ANXIETY": "불안",
    "PEER_PRESSURE": "또래 압력",
    "CHRONIC DISEASE": "만성 질환",
    "FATIGUE": "피로",
    "ALLERGY": "알레르기",
    "WHEEZING": "쌕쌕거림",
    "ALCOHOL CONSUMING": "음주",
    "COUGHING": "기침",
    "SHORTNESS OF BREATH": "호흡 곤란",
    "SWALLOWING DIFFICULTY": "삼킴 곤란",
    "CHEST PAIN": "가슴 통증",
    "SYMPTOM_COUNT": "증상 수(파생)",
}


def setup_font() -> None:
    installed = {font.name for font in font_manager.fontManager.ttflist}
    for name in ["Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR"]:
        if name in installed:
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False


class ResultVisualizer:
    def __init__(self, artifacts: Path, data_path: Path):
        self.artifacts = artifacts
        self.figures = artifacts / "figures"
        self.metrics = json.loads((artifacts / "metrics.json").read_text(encoding="utf-8"))
        bundle = joblib.load(artifacts / "model.joblib")
        self.model = bundle["model"]
        self.model_name = bundle["selected_model"]
        X, y, _ = load_survey(data_path)
        self.X, self.y = X, y
        X_train, X_test, _, y_test = train_test_split(
            X, y, test_size=self.metrics["test_size"], random_state=42, stratify=y
        )
        self.X_test = add_symptom_count(X_test)
        self.y_test = y_test

    def save(self, fig, name: str) -> Path:
        path = self.figures / name
        fig.tight_layout()
        fig.savefig(path, dpi=200)
        plt.close(fig)
        return path

    def data_overview(self) -> Path:
        featured = add_symptom_count(self.X)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        counts = self.metrics["class_counts"]
        axes[0].bar(["NO", "YES"], [counts["NO"], counts["YES"]], color=[MINT, TEAL])
        for index, value in enumerate([counts["NO"], counts["YES"]]):
            axes[0].text(index, value + 3, f"{value}건", ha="center", color=DARK)
        axes[0].set_title("정답 분포 (중복 제거 276건)")
        counts_range = list(range(0, 14))
        for label, color, name, shift in [(0, MINT, "NO", -0.2), (1, TEAL, "YES", 0.2)]:
            values = featured.loc[self.y == label, "SYMPTOM_COUNT"]
            ratios = [(values == count).mean() for count in counts_range]
            axes[1].bar([count + shift for count in counts_range], ratios, width=0.4, color=color, label=name)
        axes[1].set_xticks(counts_range)
        axes[1].set_title("증상 수 분포 (정답별 비율)")
        axes[1].set_xlabel("예라고 답한 설문 문항 수")
        axes[1].legend()
        return self.save(fig, "01_data_overview.png")

    def cv_comparison(self) -> Path:
        candidates = self.metrics["model_selection"]["candidates"]
        names = list(candidates)
        means = [candidates[name]["accuracy_mean"] for name in names]
        stds = [candidates[name]["accuracy_std"] for name in names]
        colors = [CORAL if name == self.model_name else MINT for name in names]
        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.bar(names, means, yerr=stds, color=colors, capsize=5)
        for bar, mean, std in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width() / 2, mean + std + 0.008, f"{mean:.3f}", ha="center", color=DARK)
        ax.set_ylim(0.7, 1.05)
        ax.set_ylabel("5-fold CV 평균 정확도")
        ax.set_title("후보 모델 비교 (학습 데이터 5-fold 교차검증)")
        return self.save(fig, "02_cv_comparison.png")

    def metric_comparison(self) -> Path:
        keys = ["accuracy", "precision_yes", "recall_yes", "f1_yes", "roc_auc"]
        names = ["정확도", "정밀도", "재현율", "F1", "ROC AUC"]
        baseline = [self.metrics["baseline"][key] for key in keys]
        selected = [self.metrics[key] for key in keys]
        positions = range(len(keys))
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar([p - 0.2 for p in positions], baseline, width=0.4, color=MINT, label="기준 로지스틱 회귀")
        ax.bar([p + 0.2 for p in positions], selected, width=0.4, color=TEAL, label=f"선택 {self.model_name}")
        for p, (a, b) in enumerate(zip(baseline, selected)):
            ax.text(p - 0.2, a + 0.005, f"{a:.3f}", ha="center", fontsize=8, color=GRAY)
            ax.text(p + 0.2, b + 0.005, f"{b:.3f}", ha="center", fontsize=8, color=DARK)
        ax.set_xticks(list(positions), names)
        ax.set_ylim(0.8, 1.03)
        ax.set_title("기준 모델과 선택 모델 비교 (평가 데이터 56건)")
        ax.legend(loc="lower right")
        return self.save(fig, "03_metric_comparison.png")

    def confusion(self) -> Path:
        matrix = self.metrics["confusion_matrix_no_yes"]
        tags = [["TN", "FP"], ["FN", "TP"]]
        fig, ax = plt.subplots(figsize=(4.8, 4.2))
        ax.imshow([[1, 0], [0, 1]], cmap=ListedColormap([PINK, TEAL]))
        for row in range(2):
            for col in range(2):
                value = matrix[row][col]
                color = "white" if row == col else DARK
                ax.text(col, row, f"{tags[row][col]}\n{value}", ha="center", va="center", fontsize=16, color=color)
        ax.set_xticks([0, 1], ["예측 NO", "예측 YES"])
        ax.set_yticks([0, 1], ["실제 NO", "실제 YES"])
        ax.set_title(f"혼동 행렬 ({self.model_name})")
        return self.save(fig, "04_confusion_matrix.png")

    def roc(self) -> Path:
        probabilities = self.model.predict_proba(self.X_test)[:, 1]
        fpr, tpr, _ = roc_curve(self.y_test, probabilities)
        fig, ax = plt.subplots(figsize=(5, 4.5))
        ax.plot(fpr, tpr, color=TEAL, linewidth=2.5, label=f"{self.model_name} (AUC {self.metrics['roc_auc']:.3f})")
        ax.plot([0, 1], [0, 1], linestyle="--", color=GRAY, label="무작위 추측 (AUC 0.5)")
        ax.set_xlabel("거짓 양성률 (FPR)")
        ax.set_ylabel("참 양성률 (재현율, TPR)")
        ax.set_title("ROC 곡선 (평가 데이터 56건)")
        ax.legend(loc="lower right")
        return self.save(fig, "05_roc_curve.png")

    def importance(self) -> Path:
        estimator = self.model[-1] if hasattr(self.model, "steps") else self.model
        if not hasattr(estimator, "feature_importances_"):
            raise ValueError("선택 모델이 feature_importances_를 제공하지 않습니다.")
        pairs = sorted(
            zip([*FEATURES, "SYMPTOM_COUNT"], estimator.feature_importances_),
            key=lambda pair: pair[1],
        )
        fig, ax = plt.subplots(figsize=(7, 5.5))
        colors = [CORAL if name == "SYMPTOM_COUNT" else TEAL for name, _ in pairs]
        ax.barh([LABELS_KO[name] for name, _ in pairs], [value for _, value in pairs], color=colors)
        ax.set_xlabel("중요도 (Gradient Boosting feature_importances_)")
        ax.set_title("입력 특성별 중요도")
        return self.save(fig, "06_feature_importance.png")

    def threshold(self) -> Path | None:
        report_path = self.artifacts / "threshold_report.json"
        if not report_path.exists():
            return None
        report = json.loads(report_path.read_text(encoding="utf-8"))
        rows = report["thresholds"]
        xs = [row["threshold"] for row in rows]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(xs, [row["precision"] for row in rows], marker="o", color=TEAL, label="정밀도")
        ax.plot(xs, [row["recall"] for row in rows], marker="o", color=CORAL, label="재현율")
        ax.plot(xs, [row["f1"] for row in rows], marker="o", color=GRAY, label="F1")
        ax.axvline(0.5, linestyle="--", color=DARK, label="기본값 0.5")
        ax.axvline(report["recommended_threshold"], linestyle=":", color=CORAL, label=f"F1 최대 {report['recommended_threshold']}")
        ax.set_xlabel("판정 임계값 (YES 확률 ≥ 임계값 → YES)")
        ax.set_title("임계값별 정밀도·재현율 (학습 데이터 5-fold 교차검증)")
        ax.legend(loc="lower left", fontsize=9)
        return self.save(fig, "07_threshold.png")

    def run(self) -> list[Path]:
        self.figures.mkdir(parents=True, exist_ok=True)
        setup_font()
        steps = [self.data_overview, self.cv_comparison, self.metric_comparison, self.confusion, self.roc, self.importance, self.threshold]
        saved = []
        for step in steps:
            path = step()
            if path is not None:
                saved.append(path)
        return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="학습 결과를 PNG 그래프로 저장")
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--data", type=Path, default=Path("data/lung_cancer.csv"))
    args = parser.parse_args()
    for path in ResultVisualizer(args.artifacts, args.data).run():
        print(f"저장: {path}")


if __name__ == "__main__":
    main()
