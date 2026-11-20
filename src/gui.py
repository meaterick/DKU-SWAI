from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

from predict import QUESTIONS, SurveyPredictor

MODEL_PATH = Path(__file__).resolve().parent.parent / "artifacts" / "model.joblib"
LEVEL_COLORS = {"높음": "#E76F51", "주의": "#E9A23B", "낮음": "#0B5563"}


class PredictorApp:
    def __init__(self, root: tk.Tk, predictor: SurveyPredictor):
        self.root = root
        self.predictor = predictor
        self.gender = tk.StringVar(value="M")
        self.age = tk.StringVar(value="60")
        self.answers = {column: tk.BooleanVar(value=False) for column in QUESTIONS}
        root.title("폐암 설문 위험 예측 (SW·AI입문 4조)")
        root.resizable(False, False)
        self.build()

    def build(self) -> None:
        frame = ttk.Frame(self.root, padding=16)
        frame.grid()
        ttk.Label(frame, text="폐암 설문 위험 예측", font=("Malgun Gothic", 15, "bold")).grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 10)
        )
        ttk.Label(frame, text="성별").grid(row=1, column=0, sticky="w")
        gender_box = ttk.Frame(frame)
        gender_box.grid(row=1, column=1, sticky="w")
        ttk.Radiobutton(gender_box, text="남(M)", value="M", variable=self.gender).pack(side="left")
        ttk.Radiobutton(gender_box, text="여(F)", value="F", variable=self.gender).pack(side="left")
        ttk.Label(frame, text="나이").grid(row=2, column=0, sticky="w", pady=4)
        ttk.Spinbox(frame, from_=1, to=120, textvariable=self.age, width=6).grid(row=2, column=1, sticky="w")

        for row, (column, question) in enumerate(QUESTIONS.items(), start=3):
            ttk.Checkbutton(frame, text=question, variable=self.answers[column]).grid(
                row=row, column=0, columnspan=2, sticky="w"
            )

        last = 3 + len(QUESTIONS)
        ttk.Button(frame, text="예측하기", command=self.on_predict).grid(row=last, column=0, pady=12, sticky="w")
        ttk.Button(frame, text="초기화", command=self.on_reset).grid(row=last, column=1, pady=12, sticky="e")
        self.result = tk.Label(frame, text="항목을 선택하고 [예측하기]를 누르세요.", font=("Malgun Gothic", 12), justify="left")
        self.result.grid(row=last + 1, column=0, columnspan=2, sticky="w")
        ttk.Label(frame, text="※ 학습용 모델의 참고 결과이며 의학적 진단이 아닙니다.", foreground="#5B6B6E").grid(
            row=last + 2, column=0, columnspan=2, sticky="w", pady=(8, 0)
        )

    def on_predict(self) -> None:
        age = self.age.get().strip()
        if not age.isdigit() or not 1 <= int(age) <= 120:
            messagebox.showwarning("입력 오류", "나이는 1~120 사이의 숫자로 입력하세요.")
            return
        answers = {"GENDER": self.gender.get(), "AGE": int(age)}
        for column, variable in self.answers.items():
            answers[column] = 2 if variable.get() else 1
        result = self.predictor.predict_one(answers)
        self.result.config(
            text=(
                f"예측: {result['PREDICTION']}   |   YES 확률 {result['YES_PROBABILITY'] * 100:.1f}%\n"
                f"위험 수준: {result['RISK_LEVEL']}   |   증상 수 {result['SYMPTOM_COUNT']}/13"
            ),
            fg=LEVEL_COLORS[result["RISK_LEVEL"]],
        )

    def on_reset(self) -> None:
        self.gender.set("M")
        self.age.set("60")
        for variable in self.answers.values():
            variable.set(False)
        self.result.config(text="항목을 선택하고 [예측하기]를 누르세요.", fg="black")


def main() -> None:
    root = tk.Tk()
    try:
        predictor = SurveyPredictor(MODEL_PATH)
    except FileNotFoundError as error:
        root.withdraw()
        messagebox.showerror("모델 없음", str(error))
        return
    PredictorApp(root, predictor)
    root.mainloop()


if __name__ == "__main__":
    main()
