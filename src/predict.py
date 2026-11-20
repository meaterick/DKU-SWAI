from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from train import BINARY_FEATURES, FEATURES, add_symptom_count

QUESTIONS = {
    "SMOKING": "흡연을 하나요?",
    "YELLOW_FINGERS": "손가락이 누렇게 변했나요?",
    "ANXIETY": "불안감을 자주 느끼나요?",
    "PEER_PRESSURE": "주변의 압박(또래 압력)을 느끼나요?",
    "CHRONIC DISEASE": "만성 질환이 있나요?",
    "FATIGUE": "피로를 자주 느끼나요?",
    "ALLERGY": "알레르기가 있나요?",
    "WHEEZING": "쌕쌕거리는 숨소리가 나나요?",
    "ALCOHOL CONSUMING": "음주를 하나요?",
    "COUGHING": "기침을 자주 하나요?",
    "SHORTNESS OF BREATH": "숨이 자주 차나요?",
    "SWALLOWING DIFFICULTY": "음식을 삼키기 어렵나요?",
    "CHEST PAIN": "가슴 통증이 있나요?",
}


def prepare_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = frame.columns.str.strip()
    missing = [column for column in FEATURES if column not in frame.columns]
    if missing:
        raise ValueError(f"필수 열이 없습니다: {', '.join(missing)}")
    frame = frame[FEATURES]
    if frame.empty or frame.isna().any().any():
        raise ValueError("입력이 비어 있거나 결측값이 있습니다.")

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
    return add_symptom_count(frame)


def risk_level(probability: float) -> str:
    if probability >= 0.8:
        return "높음"
    elif probability >= 0.5:
        return "주의"
    else:
        return "낮음"


class SurveyPredictor:
    def __init__(self, model_path: Path, threshold: float = 0.5):
        if not 0 < threshold < 1:
            raise ValueError("threshold는 0과 1 사이여야 합니다.")
        if not model_path.exists():
            raise FileNotFoundError(
                f"{model_path}가 없습니다. 먼저 python src/train.py를 실행하세요."
            )
        bundle = joblib.load(model_path)
        self.model = bundle["model"]
        self.features = bundle["features"]
        self.model_name = bundle["selected_model"]
        self.threshold = threshold

    def predict(self, raw: pd.DataFrame) -> pd.DataFrame:
        features = prepare_features(raw)
        probabilities = self.model.predict_proba(features[self.features])[:, 1]
        results = []
        for symptoms, probability in zip(features["SYMPTOM_COUNT"], probabilities):
            results.append(
                {
                    "SYMPTOM_COUNT": int(symptoms),
                    "YES_PROBABILITY": round(float(probability), 4),
                    "PREDICTION": "YES" if probability >= self.threshold else "NO",
                    "RISK_LEVEL": risk_level(float(probability)),
                }
            )
        return pd.concat(
            [raw.reset_index(drop=True), pd.DataFrame(results)], axis=1
        )

    def predict_one(self, answers: dict) -> dict:
        return self.predict(pd.DataFrame([answers])).iloc[0].to_dict()


def ask_choice(prompt: str, allowed: list[str]) -> str:
    while True:
        answer = input(f"{prompt} ").strip().upper()
        if answer in allowed:
            return answer
        print(f"  다시 입력하세요. 가능한 값: {', '.join(allowed)}")


def ask_age() -> int:
    while True:
        answer = input("나이(1~120)를 입력하세요: ").strip()
        if answer.isdigit() and 1 <= int(answer) <= 120:
            return int(answer)
        print("  1~120 사이의 숫자를 입력하세요.")


def collect_answers() -> dict:
    answers = {"GENDER": ask_choice("성별을 입력하세요 (M/F):", ["M", "F"])}
    answers["AGE"] = ask_age()
    for number, (column, question) in enumerate(QUESTIONS.items(), start=1):
        reply = ask_choice(f"[{number:2d}/13] {question} (Y/N):", ["Y", "N"])
        answers[column] = 2 if reply == "Y" else 1
    return answers


def print_result(result: dict, threshold: float) -> None:
    print("\n" + "=" * 44)
    print(f" 예측 결과        : {result['PREDICTION']}")
    print(f" 폐암 YES 확률    : {result['YES_PROBABILITY'] * 100:.1f}%")
    print(f" 위험 수준        : {result['RISK_LEVEL']}")
    print(f" 증상 수          : {result['SYMPTOM_COUNT']} / 13")
    print(f" 판정 임계값      : {threshold}")
    print("=" * 44)
    print(" ※ 학습용 모델의 참고 결과이며 의학적 진단이 아닙니다.")


def run_interactive(predictor: SurveyPredictor) -> None:
    print(f"폐암 설문 예측 ({predictor.model_name}) 15개 항목에 답하세요.")
    while True:
        result = predictor.predict_one(collect_answers())
        print_result(result, predictor.threshold)
        if ask_choice("\n한 번 더 예측할까요? (Y/N):", ["Y", "N"]) == "N":
            break


def main() -> None:
    parser = argparse.ArgumentParser(description="학습된 모델로 폐암 설문 결과 예측")
    parser.add_argument("--model", type=Path, default=Path("artifacts/model.joblib"))
    parser.add_argument("--input", type=Path, help="예측할 설문 CSV (LUNG_CANCER 열 불필요)")
    parser.add_argument("--output", type=Path, default=Path("artifacts/predictions.csv"))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--interactive", action="store_true", help="터미널에서 15문항을 직접 입력")
    args = parser.parse_args()

    predictor = SurveyPredictor(args.model, args.threshold)
    if args.interactive:
        run_interactive(predictor)
    elif args.input:
        table = predictor.predict(pd.read_csv(args.input))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.output, index=False, encoding="utf-8-sig")
        summary = table["PREDICTION"].value_counts().to_dict()
        print(table[["GENDER", "AGE", "SYMPTOM_COUNT", "YES_PROBABILITY", "PREDICTION", "RISK_LEVEL"]].to_string(index=False))
        print(json.dumps({"rows": len(table), "summary": summary, "saved": str(args.output)}, ensure_ascii=False))
    else:
        parser.error("--input CSV 경로 또는 --interactive 중 하나를 지정하세요.")


if __name__ == "__main__":
    main()
