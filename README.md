# DKU-SWAI — Kaggle 폐암 설문 데이터 기반 분류 모델

단국대학교 **SW·AI입문(SW) 4조** 팀 프로젝트로, Kaggle 폐암 설문 데이터의 feature를 이용해 `LUNG_CANCER`(YES/NO)를 분류하는 모델을 학습시킵니다.

## 팀 구성 — 4조

| 구분 | 이름 | 소속 | 담당 역할 |
| --- | --- | --- | --- |
| 팀장 | 박인찬 | 퇴계혁신칼리지 (SW융합계열광역) | 모델 학습·평가, 통합·테스트, 발표 |
| 팀원 | 이서준 | 컴퓨터공학과 | 데이터 검증·전처리, --predict, 데이터·검증 발표 |
| 팀원 | 유소현 | 컴퓨터공학과 | 시각화, 자료 조사, PPT, 결과 발표 |

## 구현 개요

1. CSV의 필수 열과 값 범위를 확인하고 완전히 동일한 행을 제거
2. 설문 응답을 숫자로 변환하고 정답 비율을 유지해 학습·평가 데이터를 80:20으로 split
3. 학습 데이터에만 스케일러를 적합한 뒤, 증상 수 파생 변수를 추가하고 5-fold 계층 교차검증으로 후보 4개를 비교해 Gradient Boosting을 선택
4. 평가 데이터에서 정확도, YES 클래스 정밀도·재현율·F1, ROC AUC, 혼동 행렬을 기록

학습·평가·결과 저장 흐름은 `LungCancerTrainer` 클래스가 관리합니다.

## 역할별 코드 안내

- `src/train_explained.py`: 함수별 설명과 `[이서준]`, `[박인찬]`, `[유소현]` 역할 태그가 있는 실행 복사본
- `docs/역할별_코드설명.md`: 팀원별 설명 담당 코드, 핵심 설명 포인트, 예상 질문 정리

## 모델 개선 결과

기준 로지스틱 회귀에 증상 수 파생 변수(`SYMPTOM_COUNT`)와 5-fold 계층 교차검증 기반 모델 선택을 적용했습니다. 선택 모델은 `GradientBoosting(100, leaf=4)`입니다. 평가 데이터 56건 기준 결과는 다음과 같습니다.

| 모델 | 정확도 | YES 정밀도 | YES 재현율 | YES F1 | ROC AUC |
| --- | --- | --- | --- | --- | --- |
| 기준 로지스틱 회귀 | 0.893 | 1.000 | 0.875 | 0.933 | 0.958 |
| 선택 Gradient Boosting | 0.929 | 0.958 | 0.958 | 0.958 | 0.966 |

- 학습 데이터 5-fold CV 정확도: 선택 Gradient Boosting 0.923 (후보 4개 중 1위)
- 놓친 YES: 6건 → 2건, 대신 오탐 0건 → 2건
- 평가 데이터가 56건으로 작아 단일 분할 결과의 변동 가능성이 있습니다. 향후 임계값 조정으로 정밀도와 재현율의 균형을 확인합니다.

## 데이터 준비

- **출처:** [Kaggle — Lung Cancer (mysarahmadbhat)](https://www.kaggle.com/datasets/mysarahmadbhat/lung-cancer)
- **정답 열:** `LUNG_CANCER` (`YES`/`NO`)
- **기본 정보:** `GENDER` (`F`/`M`), `AGE` (1~120 사이 숫자)
- **설문 항목:** `SMOKING`, `YELLOW_FINGERS`, `ANXIETY`, `PEER_PRESSURE`, `CHRONIC DISEASE`, `FATIGUE`, `ALLERGY`, `WHEEZING`, `ALCOHOL CONSUMING`, `COUGHING`, `SHORTNESS OF BREATH`, `SWALLOWING DIFFICULTY`, `CHEST PAIN` (각각 1=아니오, 2=예)
- **파생 변수:** 13개 설문 응답의 합 `SYMPTOM_COUNT` (0~13). 모델 입력은 원본 15개와 파생 1개를 합친 16개입니다.

이 저장소에는 `data/lung_cancer.csv`가 포함되어 있어 별도 다운로드 없이 실행할 수 있습니다.

## 실행 방법

Git과 Python 3.10 이상을 설치한 뒤, 터미널에서 저장소를 클론합니다. 아래 명령은 현재 포함된 `data/lung_cancer.csv`를 사용합니다.

### Windows PowerShell

```powershell
git clone https://github.com/meaterick/DKU-SWAI.git
cd DKU-SWAI
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe src\train.py
```

### macOS / Linux

```bash
git clone https://github.com/meaterick/DKU-SWAI.git
cd DKU-SWAI
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python src/train.py
```

`src/train.py`는 설문 CSV를 검증하고 후보 4개를 비교한 뒤 선택 모델을 학습·평가합니다. 실행 결과는 `artifacts/metrics.json`과 `artifacts/model.joblib`에 저장됩니다. `src/train_explained.py`는 같은 로직에 함수별 설명과 주석을 붙인 복사본이며, `python src/train_explained.py`로 실행할 수 있고 결과도 동일합니다.

## 저장소 구성 원칙

```text
.
├── README.md          # 프로젝트 소개와 참고 자료
├── requirements.txt   # Python 의존성
├── src/train.py       # 실행용 코드 (주석 없음)
├── src/train_explained.py # 같은 로직에 함수별 설명·주석 포함
├── data/lung_cancer.csv # 저장소에 포함된 설문 CSV
├── artifacts/         # 실행 시 생성되는 모델·평가 결과 (Git에서 제외)
└── docs/              # 프로젝트 문서와 발표 자료
```

## 참고 자료

- 데이터: [Kaggle Lung Cancer 설문 데이터](https://www.kaggle.com/datasets/mysarahmadbhat/lung-cancer)
- 개발: [Python 공식 문서](https://docs.python.org/3/), [pandas 문서](https://pandas.pydata.org/docs/), [scikit-learn 문서](https://scikit-learn.org/stable/)
- 모델 선택: [scikit-learn Model selection](https://scikit-learn.org/stable/model_selection.html), [GradientBoostingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)
