# 학습 코드

`train.py`는 설문 CSV를 검증하고 중복 행을 제거한 뒤, 계층화된 80:20 분할로 로지스틱 회귀 기준 모델을 학습합니다. 결과는 `artifacts/`에 저장됩니다.

저장소 루트에서 실행:

```bash
python src/train.py --data data/survey_lung_cancer.csv
```

평가 점수는 분할된 한 번의 테스트 결과입니다. 작은 데이터셋에 대한 안정적인 성능 추정이나 의료적 효용을 의미하지 않습니다.
