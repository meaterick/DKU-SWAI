# 데이터 준비

사용할 데이터: [Kaggle — Lung Cancer (mysarahmadbhat)](https://www.kaggle.com/datasets/mysarahmadbhat/lung-cancer). Kaggle 페이지에서 설문 CSV를 직접 내려받아 이 폴더에 `survey_lung_cancer.csv`로 저장합니다. CSV 파일은 Git에서 제외됩니다. 데이터의 사용 조건은 다운로드 전에 Kaggle 페이지에서 확인하세요.

학습 코드는 다음 열을 요구합니다.

- 정답: `LUNG_CANCER` (`YES`/`NO`)
- 기본 정보: `GENDER` (`F`/`M`), `AGE` (1~120 사이 숫자)
- 설문 항목: `SMOKING`, `YELLOW_FINGERS`, `ANXIETY`, `PEER_PRESSURE`, `CHRONIC DISEASE`, `FATIGUE`, `ALLERGY`, `WHEEZING`, `ALCOHOL CONSUMING`, `COUGHING`, `SHORTNESS OF BREATH`, `SWALLOWING DIFFICULTY`, `CHEST PAIN` (각각 1=아니오, 2=예)

열 이름의 앞뒤 공백은 자동으로 제거합니다. 다른 버전의 데이터셋을 사용한다면 열 구성과 값 인코딩을 먼저 확인하고 코드를 수정하세요. 원본 데이터의 출처는 [`docs/references.md`](../docs/references.md)에 기록했습니다.
