"""폐암 설문 데이터 기반 분류 모델 학습 스크립트 (역할별 설명용).

이 파일은 src/train.py와 완전히 같은 동작을 하면서, 팀원 세 사람이
자기 담당 코드를 찾아 설명할 수 있도록 역할 태그와 상세 주석을 붙인
복사본입니다. 코드를 실행해도 결과는 src/train.py와 같습니다.

실행 방법:
    python src/train_explained.py

역할 태그:
    [이서준] 데이터 검증·전처리, 파생 변수
    [박인찬] 모델 선택·학습·통합, CLI
    [유소현] 평가 지표·혼동 행렬·결과 정리
    [공통] run()의 전체 흐름처럼 함께 설명하는 부분

자세한 설명 순서와 예상 질문은 docs/역할별_코드설명.md에 정리되어 있습니다.
"""

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


# ===========================================================================
# [이서준] 데이터 스키마: CSV의 어떤 열을 쓰고, 어떤 값으로 바꾸는지 정의
# ===========================================================================
# TARGET은 CSV에서 맞혀야 하는 정답 열입니다. 값은 "YES" 또는 "NO"입니다.
TARGET = "LUNG_CANCER"

# 설문 문항 13개입니다. 원본 CSV에서는 1=아니오, 2=예로 기록되어 있습니다.
# load_survey()에서 0/1로 바꾼 뒤 모델 입력으로 사용합니다.
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

# 원본 입력 특성 15개입니다. GENDER, AGE + 설문 13개로 구성됩니다.
FEATURES = ["GENDER", "AGE", *BINARY_FEATURES]

# 13개 설문 응답의 합을 저장할 파생 변수 이름입니다. 값은 0~13입니다.
DERIVED_FEATURE = "SYMPTOM_COUNT"

# 실제 모델에 넣는 입력 특성 16개입니다. 원본 15개 + 파생 변수 1개입니다.
MODEL_FEATURES = [*FEATURES, DERIVED_FEATURE]


# ===========================================================================
# [이서준] 데이터 읽기·검증: CSV를 모델이 이해할 수 있는 숫자 표로 바꾸는 함수
# ===========================================================================

def load_survey(path: Path) -> tuple[pd.DataFrame, pd.Series, dict[str, int]]:
    """CSV를 읽고 값 규칙을 검사한 뒤 숫자로 변환합니다.

    매개변수:
        path: 설문 CSV 파일 경로입니다.

    반환값:
        X: 원본 입력 특성 15개로 이루어진 표입니다.
        y: 정답 레이블입니다. NO=0, YES=1로 변환되어 있습니다.
        counts: 원본 데이터의 NO/YES 개수입니다.

    처리 순서:
        1) CSV를 읽고 열 이름의 앞뒤 공백을 제거합니다.
        2) 필수 열이 모두 있는지, 중복된 열 이름은 없는지 확인합니다.
        3) 완전히 같은 행을 제거하고 결측값이 없는지 확인합니다.
        4) GENDER, AGE, 설문 13개, 정답 열을 숫자로 변환합니다.
        5) 각 클래스가 최소 2개 이상인지 확인합니다.

    이 함수가 끝나면 문자열(YES/NO, F/M)은 모두 숫자로 바뀌고,
    이후 모델은 숫자 표만 다루게 됩니다.
    """
    # 1) CSV 파일을 pandas DataFrame으로 읽습니다.
    #    DataFrame은 엑셀 표처럼 행과 열로 이루어진 자료구조입니다.
    frame = pd.read_csv(path)

    # 2) 열 이름에 붙은 공백을 제거합니다.
    #    " AGE"와 "AGE"를 같은 열로 보기 위해서입니다.
    frame.columns = frame.columns.str.strip()
    if frame.columns.duplicated().any():
        raise ValueError("공백 제거 후 중복된 열 이름이 있습니다.")

    # 3) 모델에 필요한 열이 빠짐없이 있는지 확인합니다.
    #    set 차집합으로 필요한 열 - 실제 열을 계산합니다.
    missing = sorted(set([*FEATURES, TARGET]) - set(frame.columns))
    if missing:
        raise ValueError(f"필수 열이 없습니다: {', '.join(missing)}")

    # 4) 필요한 열만 남기고, 완전히 동일한 행은 하나로 합칩니다.
    #    drop_duplicates()는 모든 열의 값이 같은 행을 제거합니다.
    #    copy()를 사용해 원본 frame과 분리된 새 표로 만듭니다.
    frame = frame[[*FEATURES, TARGET]].drop_duplicates().copy()
    if frame.empty or frame.isna().any().any():
        raise ValueError("데이터가 비어 있거나 결측값이 있습니다. 원본 CSV를 확인하세요.")

    # 5) GENDER를 숫자로 바꿉니다. F=0, M=1로 변환합니다.
    #    astype(str)로 문자열로 만든 뒤 strip/upper로 표기를 통일합니다.
    gender = frame["GENDER"].astype(str).str.strip().str.upper()
    if not gender.isin(["F", "M"]).all():
        raise ValueError("GENDER는 F 또는 M이어야 합니다.")
    frame["GENDER"] = gender.map({"F": 0, "M": 1})

    # 6) AGE를 숫자로 바꾸고 1~120 범위인지 확인합니다.
    #    errors="raise"는 숫자로 바꿀 수 없는 값이 있으면 바로 오류를 냅니다.
    age = pd.to_numeric(frame["AGE"], errors="raise")
    if not age.between(1, 120).all():
        raise ValueError("AGE는 1~120 사이의 숫자여야 합니다.")
    frame["AGE"] = age

    # 7) 설문 13개를 0/1로 바꿉니다. 원본 1(아니오)은 0, 2(예)는 1이 됩니다.
    #    for 반복문으로 13개 열을 하나씩 검사합니다.
    for column in BINARY_FEATURES:
        values = pd.to_numeric(frame[column], errors="raise")
        if not values.isin([1, 2]).all():
            raise ValueError(f"{column}은 1(아니오) 또는 2(예)여야 합니다.")
        frame[column] = values - 1

    # 8) 정답 열을 0/1로 바꿉니다. NO=0, YES=1입니다.
    #    모델은 숫자 정답을 사용하므로 문자열을 그대로 넣지 않습니다.
    labels = frame[TARGET].astype(str).str.strip().str.upper()
    if not labels.isin(["NO", "YES"]).all():
        raise ValueError("LUNG_CANCER는 NO 또는 YES여야 합니다.")
    y = labels.map({"NO": 0, "YES": 1}).astype(int)

    # 9) 클래스별 개수를 세고, 계층화 분할을 위해 각 클래스가 2개 이상인지 확인합니다.
    #    NO/YES가 각각 2개 미만이면 train_test_split(stratify=y)가 실패할 수 있습니다.
    counts = {"NO": int((y == 0).sum()), "YES": int((y == 1).sum())}
    if min(counts.values()) < 2:
        raise ValueError("계층화 분할을 위해 각 클래스에 최소 2개 행이 필요합니다.")

    # 10) 원본 15개 특성과 정답, 클래스 개수를 반환합니다.
    #     파생 변수 SYMPTOM_COUNT는 add_symptom_count()에서 따로 추가합니다.
    return frame[FEATURES], y, counts


# ===========================================================================
# [이서준] 파생 변수: 13개 설문 응답의 합을 새 특성으로 추가
# ===========================================================================

def add_symptom_count(frame: pd.DataFrame) -> pd.DataFrame:
    """13개 설문 응답의 합을 SYMPTOM_COUNT 열로 추가합니다.

    매개변수:
        frame: GENDER, AGE, 설문 13개가 숫자로 변환된 표입니다.

    반환값:
        SYMPTOM_COUNT가 추가된 새로운 표입니다.

    설문 응답은 0(아니오) 또는 1(예)이므로, 행 방향 합은
    "예"라고 답한 문항 수가 됩니다. 값은 0~13 범위입니다.

    이 함수는 정답(y)을 사용하지 않는 단순 변환이므로,
    학습/평가 데이터로 나눈 뒤 각각에 적용해도 정보 누출이 없습니다.
    """
    # 원본 표를 직접 바꾸지 않도록 복사합니다.
    featured = frame.copy()

    # axis=1은 행(row) 방향으로 합한다는 뜻입니다.
    # 즉, 한 사람의 13개 설문 응답 값을 모두 더합니다.
    featured[DERIVED_FEATURE] = featured[BINARY_FEATURES].sum(axis=1)
    return featured


# ===========================================================================
# [유소현] 평가 지표: 예측 결과를 숫자로 요약하는 함수
# ===========================================================================

def evaluate_model(model, frame: pd.DataFrame, y: pd.Series) -> dict:
    """학습된 모델로 예측하고 평가 지표를 딕셔너리로 반환합니다.

    매개변수:
        model: fit()이 끝난 분류 모델입니다.
        frame: 평가용 입력 특성입니다.
        y: 평가용 정답 레이블입니다.

    반환값:
        accuracy: 전체 예측 중 맞힌 비율
        precision_yes: YES라고 예측한 것 중 실제 YES인 비율
        recall_yes: 실제 YES 중 YES라고 예측한 비율
        f1_yes: 정밀도와 재현율의 조화평균
        roc_auc: 확률을 이용한 클래스 구분 능력
        confusion_matrix_no_yes: [[TN, FP], [FN, TP]] 형태의 혼동 행렬

    혼동 행렬의 행은 실제 값, 열은 예측 값입니다.
    labels=[0, 1]을 지정해 항상 NO, YES 순서로 고정합니다.
    """
    # predict()는 0/1 클래스 레이블을 반환합니다.
    predictions = model.predict(frame)

    # predict_proba()는 [P(NO), P(YES)]를 반환합니다.
    # ROC AUC에는 YES일 확률인 두 번째 열만 사용합니다.
    probabilities = model.predict_proba(frame)[:, 1]

    return {
        # 전체 예측 중 맞힌 비율입니다.
        "accuracy": float(accuracy_score(y, predictions)),
        # YES라고 예측한 것 중 실제 YES인 비율입니다.
        # 0으로 나누는 경우를 대비해 zero_division=0으로 처리합니다.
        "precision_yes": float(precision_score(y, predictions, zero_division=0)),
        # 실제 YES 중 YES라고 예측한 비율입니다.
        "recall_yes": float(recall_score(y, predictions, zero_division=0)),
        # 정밀도와 재현율의 조화평균입니다.
        "f1_yes": float(f1_score(y, predictions, zero_division=0)),
        # 확률을 이용해 클래스를 얼마나 잘 구분하는지 나타내는 값입니다.
        "roc_auc": float(roc_auc_score(y, probabilities)),
        # labels=[0, 1]을 지정해 행/열 순서를 NO, YES로 고정합니다.
        # 결과는 [[TN, FP], [FN, TP]] 순서입니다.
        "confusion_matrix_no_yes": confusion_matrix(y, predictions, labels=[0, 1]).tolist(),
    }


# ===========================================================================
# [박인찬] 모델 선택·학습·통합: 한 번의 실행을 관리하는 클래스
# ===========================================================================

class LungCancerTrainer:
    """데이터 분할부터 모델 선택, 평가, 파일 저장까지 한 번의 실행을 관리합니다."""

    def __init__(self, data_path: Path, output_dir: Path, test_size: float = 0.2):
        """실행에 필요한 경로와 평가 데이터 비율을 저장합니다.

        매개변수:
            data_path: 설문 CSV 경로입니다.
            output_dir: metrics.json과 model.joblib을 저장할 폴더입니다.
            test_size: 평가 데이터로 떼어 둘 비율입니다. 기본값은 0.2(20%)입니다.
        """
        # test_size가 0 또는 1이면 학습/평가 분할이 불가능하므로 미리 막습니다.
        if not 0 < test_size < 1:
            raise ValueError("test_size는 0과 1 사이여야 합니다.")
        self.data_path = data_path
        self.output_dir = output_dir
        self.test_size = test_size

    def build_baseline(self):
        """비교 기준이 되는 로지스틱 회귀 모델을 만듭니다.

        StandardScaler와 LogisticRegression을 하나의 Pipeline으로 묶습니다.
        Pipeline 안에 스케일러를 넣으면, 교차검증에서 학습 데이터에만
        평균과 표준편차를 적합하므로 데이터 누출을 막을 수 있습니다.
        class_weight="balanced"는 YES/NO 개수 차이를 보정합니다.
        """
        return make_pipeline(
            # StandardScaler는 각 특성을 평균 0, 표준편차 1로 맞춥니다.
            # 로지스틱 회귀는 특성 단위에 민감하므로 표준화가 도움이 됩니다.
            StandardScaler(),
            # class_weight="balanced"는 클래스 개수의 역수에 비례해 가중치를 줍니다.
            # YES가 많은 불균형 데이터에서 NO도 놓치지 않도록 돕습니다.
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        )

    def build_candidates(self) -> dict:
        """비교할 후보 모델 4개를 딕셔너리로 만듭니다.

        딕셔너리의 key는 모델 이름, value는 학습 전 모델 객체입니다.
        같은 이름을 결과 JSON과 화면 출력에서 그대로 사용합니다.
        """
        return {
            # 1) 선형 기준 모델입니다. 스케일러가 필요하므로 Pipeline으로 묶습니다.
            #    C는 규제 강도의 역수입니다. C가 작을수록 규제가 강해집니다.
            "LogisticRegression": make_pipeline(
                StandardScaler(),
                LogisticRegression(C=1.0, max_iter=3000, random_state=42),
            ),
            # 2) 거리 기반 모델입니다. 특성마다 단위가 다르므로 표준화가 중요합니다.
            #    k=5는 가장 가까운 이웃 5개의 다수결로 분류한다는 뜻입니다.
            "KNN(k=5)": make_pipeline(
                StandardScaler(),
                KNeighborsClassifier(n_neighbors=5),
            ),
            # 3) 배깅 방식의 트리 모델입니다. 나무 200개를 독립적으로 만듭니다.
            #    min_samples_leaf=2로 잎 노드가 너무 작아지는 것을 막아 과적합을 줄입니다.
            "RandomForest": RandomForestClassifier(
                n_estimators=200,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=42,
            ),
            # 4) 부스팅 방식의 트리 모델입니다.
            #    나무 100개를 순차적으로 보정하고, 깊이 2의 단순한 나무를 사용합니다.
            #    learning_rate=0.05는 한 번에 얼마나 크게 보정할지 정합니다.
            #    subsample=0.9는 각 단계에서 학습 데이터의 90%만 무작위로 사용합니다.
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
        """5-fold 교차검증으로 후보 4개를 비교해 가장 좋은 모델 이름을 고릅니다.

        매개변수:
            frame: 학습 데이터의 입력 특성입니다.
            y: 학습 데이터의 정답 레이블입니다.

        반환값:
            best_name: 평균 정확도가 가장 높은 모델 이름입니다.
            results: 후보별 평균/표준편차 지표를 담은 딕셔너리입니다.

        평가 데이터는 이 함수에 넘기지 않습니다. 모델 선택은 학습 데이터에서만
        수행해야 최종 평가 데이터가 선택에 영향을 주지 않습니다.
        """
        # StratifiedKFold는 각 fold에 NO/YES 비율이 비슷하게 들어가도록 나눕니다.
        # shuffle=True로 데이터 순서에 따른 편향을 줄이고, random_state로 재현합니다.
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # cross_validate에 전달할 평가 지표 이름과 sklearn 지표 이름입니다.
        scoring = {
            "accuracy": "accuracy",
            "f1": "f1",
            "roc_auc": "roc_auc",
            "recall": "recall",
        }

        results = {}

        # 후보 모델을 하나씩 꺼내 5-fold 교차검증을 수행합니다.
        for name, model in self.build_candidates().items():
            # cross_validate는 fold마다 학습/검증을 반복하고 점수 배열을 돌려줍니다.
            # n_jobs=1은 한 개의 프로세스로 순서대로 실행해 결과를 재현하기 쉽게 합니다.
            scores = cross_validate(model, frame, y, cv=cv, scoring=scoring, n_jobs=1)

            # fold마다 계산된 점수의 평균과 표준편차를 저장합니다.
            # test_*는 검증 fold에서 계산한 점수입니다.
            results[name] = {
                "accuracy_mean": float(scores["test_accuracy"].mean()),
                "accuracy_std": float(scores["test_accuracy"].std()),
                "f1_mean": float(scores["test_f1"].mean()),
                "roc_auc_mean": float(scores["test_roc_auc"].mean()),
                "recall_mean": float(scores["test_recall"].mean()),
            }

        # 평균 정확도를 1순위, 평균 F1을 2순위로 가장 높은 모델을 고릅니다.
        # max()에 tuple을 돌려주면 앞의 값이 같을 때 뒤의 값으로 비교합니다.
        best_name = max(
            results,
            key=lambda name: (results[name]["accuracy_mean"], results[name]["f1_mean"]),
        )
        return best_name, results

    def build_model(self, name: str):
        """선택된 이름에 해당하는 새 모델 객체를 반환합니다.

        select_model()에서 고른 모델을 다시 만들어 학습 데이터 전체에
        fit하기 위해 사용합니다. 후보 딕셔너리에서 같은 이름을 꺼냅니다.
        """
        return self.build_candidates()[name]

    def run(self) -> dict:
        """[공통] 데이터 읽기부터 저장까지 전체 학습 과정을 실행합니다.

        반환값:
            평가 지표와 모델 선택 결과를 담은 metrics 딕셔너리입니다.

        각 단계 앞의 역할 태그를 보면 누가 어떤 부분을 설명하면 되는지
        알 수 있습니다. run() 전체는 세 사람이 함께 설명합니다.
        """
        # -------------------------------------------------------------------
        # [이서준] 1) CSV를 읽고 검증한 뒤 원본 특성 15개와 정답을 받습니다.
        # -------------------------------------------------------------------
        X, y, counts = load_survey(self.data_path)

        # -------------------------------------------------------------------
        # [박인찬] 2) 학습 80% / 평가 20%로 나눕니다.
        # stratify=y를 사용해 두 데이터의 YES/NO 비율을 비슷하게 유지합니다.
        # random_state=42로 실행할 때마다 같은 분할이 나오게 합니다.
        # -------------------------------------------------------------------
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=42, stratify=y
        )

        # -------------------------------------------------------------------
        # [박인찬] 3-1) 기준 모델을 원본 15개 특성으로 학습합니다.
        # -------------------------------------------------------------------
        baseline = self.build_baseline()
        baseline.fit(X_train_raw, y_train)

        # -------------------------------------------------------------------
        # [유소현] 3-2) 기준 모델을 평가 데이터로 평가합니다.
        # 이 값은 선택 모델과 비교하기 위한 기준점입니다.
        # -------------------------------------------------------------------
        baseline_metrics = evaluate_model(baseline, X_test_raw, y_test)

        # -------------------------------------------------------------------
        # [이서준] 4) 학습/평가 데이터 각각에 증상 수 파생 변수를 추가합니다.
        # 이 변환은 정답을 사용하지 않으므로 분할 후에 계산해도 안전합니다.
        # -------------------------------------------------------------------
        X_train = add_symptom_count(X_train_raw)
        X_test = add_symptom_count(X_test_raw)

        # -------------------------------------------------------------------
        # [박인찬] 5) 학습 데이터에서만 5-fold CV를 수행해 모델을 선택합니다.
        # 평가 데이터는 여기서 사용하지 않습니다.
        # -------------------------------------------------------------------
        selected_name, cv_results = self.select_model(X_train, y_train)

        # -------------------------------------------------------------------
        # [박인찬] 6) 선택된 모델을 학습 데이터 전체에 다시 학습합니다.
        # -------------------------------------------------------------------
        model = self.build_model(selected_name)
        model.fit(X_train, y_train)

        # -------------------------------------------------------------------
        # [유소현] 7) 평가 데이터로 최종 성능을 계산합니다.
        # -------------------------------------------------------------------
        selected_metrics = evaluate_model(model, X_test, y_test)

        # -------------------------------------------------------------------
        # [유소현] 8) 실행 결과를 하나의 딕셔너리로 정리합니다.
        # source, rows, counts, features, model_selection, baseline, selected
        # 지표를 한곳에 모아 JSON으로 저장할 수 있게 합니다.
        # -------------------------------------------------------------------
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
            # selected_metrics의 accuracy, precision_yes, recall_yes, f1_yes,
            # roc_auc, confusion_matrix_no_yes를 최상위 키로 합칩니다.
            **selected_metrics,
        }

        # -------------------------------------------------------------------
        # [유소현] 9) 결과 폴더를 만들고 평가 지표를 JSON으로 저장합니다.
        # ensure_ascii=False로 한글을 그대로 저장하고, indent=2로 읽기 쉽게 정리합니다.
        # -------------------------------------------------------------------
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        # -------------------------------------------------------------------
        # [박인찬] 10) 나중에 예측할 때 필요한 모델, 특성 순서, 선택 모델 이름을 저장합니다.
        # -------------------------------------------------------------------
        joblib.dump(
            {"model": model, "features": MODEL_FEATURES, "selected_model": selected_name},
            self.output_dir / "model.joblib",
        )
        return metrics


# ===========================================================================
# [박인찬] 함수 형태의 실행 인터페이스와 CLI
# ===========================================================================

def train(data_path: Path, output_dir: Path, test_size: float = 0.2) -> dict:
    """LungCancerTrainer를 함수 형태로 실행하는 짧은 래퍼입니다.

    매개변수:
        data_path: 설문 CSV 경로입니다.
        output_dir: 결과를 저장할 폴더입니다.
        test_size: 평가 데이터 비율입니다.

    반환값:
        LungCancerTrainer.run()이 만든 metrics 딕셔너리입니다.
    """
    return LungCancerTrainer(data_path, output_dir, test_size).run()


def main() -> None:
    """터미널에서 실행할 때 인자를 읽고 train()을 호출합니다.

    사용 예:
        python src/train_explained.py
        python src/train_explained.py --data data/lung_cancer.csv --output-dir artifacts
    """
    parser = argparse.ArgumentParser(description="폐암 설문 데이터 분류 모델 학습")

    # --data를 생략하면 저장소에 포함된 기본 CSV를 사용합니다.
    parser.add_argument("--data", type=Path, default=Path("data/lung_cancer.csv"))

    # 결과 파일을 저장할 폴더입니다. 없으면 run()에서 자동으로 만듭니다.
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))

    # 평가 데이터 비율을 바꾸고 싶을 때 사용합니다.
    parser.add_argument("--test-size", type=float, default=0.2)

    args = parser.parse_args()

    # 실행 결과 딕셔너리를 읽기 좋은 JSON 형태로 출력합니다.
    print(json.dumps(train(args.data, args.output_dir, args.test_size), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
