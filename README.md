# DKU-SWAI — Kaggle 폐암 설문 데이터 기반 분류 모델

단국대학교 **SW·AI입문(SW) 4조** 팀 프로젝트로, Kaggle 폐암 설문 데이터의 feature를 이용해 `LUNG_CANCER`(YES/NO)를 분류하는 모델을 학습시킵니다.

## 팀 구성 — 4조

| 구분 | 이름 | 소속 | 담당 역할 |
| --- | --- | --- | --- |
| 팀장 | 박인찬 | 퇴계혁신칼리지 (SW융합계열광역) | 모델 학습·평가, 통합·테스트, 발표 |
| 팀원 | 이서준 | 컴퓨터공학과 | 데이터 검증·전처리, --predict, 데이터·검증 발표 |
| 팀원 | 유소현 | 컴퓨터공학과 | 시각화, 자료 조사, PPT, 결과 발표 |

## 2차 팀 프로젝트 과제 안내

4인 1조로 수행하는 2차 팀 프로젝트 과제입니다. (배점: 총 20점)

### 프로젝트 개요

- **주제:** 자유 주제
  - 실생활 문제 해결
  - 파이썬 기반 데이터 수집·처리·분석 및 시각화
  - 텍스트/GUI 게임
  - 업무 자동화 설계 및 구현
- **파이썬 중심 구현 조건:** 아래 중 하나 이상
  - 공공데이터/오픈 API 연동 (`requests` 등)
  - 데이터 분석 및 시각화 (`pandas`, `matplotlib` 등)
  - 파일 처리 또는 경량 DB 연동 (CSV, JSON, `sqlite3` 등)
  - 대화형/시각적 인터페이스 (`Streamlit`, `tkinter`, `turtle` 등)
- **파이썬 기본 문법:** 기본 자료형·자료구조(리스트, 딕셔너리 등), 조건문(`if/elif/else`), 반복문(`for/while`), 함수 정의, 클래스 적용
- **파이썬 확장 활용:** 위 구현 조건 중 1개 이상 사용
- **역할 분담:** 조원 구성원의 역할을 명시하고, 보고서에 각 조원의 구체적인 역할을 기재

### 팀원 역할 분담

현재 역할 분담은 아래 팀 구성 표와 `docs/역할별_코드설명.md`에 정리되어 있습니다.

| 담당 | 주요 역할 |
| --- | --- |
| 박인찬 (팀장) | 모델 학습·평가, 통합·테스트, 발표 |
| 이서준 | 데이터 검증·전처리, `--predict`, 데이터·검증 발표 |
| 유소현 | 시각화, 자료 조사, PPT, 결과 발표 |

### 단계별 제출 일정 및 배점

기한을 엄수해야 하며, 기한 내 제출 시 부분점수가 부과됩니다.

| 단계 | 내용 | 기한·배점 | 제출물 | 제출 방법 |
| --- | --- | --- | --- | --- |
| 1. 기획서 | 기능 정의, 실행 흐름도, 파이썬 개발 환경, 팀원 역할 분담 | 10/21(수) 23:59 · 3점 | 과제 요약서(Word/HWP) | 조장이 조원 수신처를 포함해 담당교수 메일로 제출 |
| 2. 중간 진척 | 현재 코드 구현 진행률, 핵심 파이썬 함수 동작 여부, 잔여 구현 계획 | 11/11(수) 23:59 · 3점 | 요약서, 초안 PPT, 소스코드 일부 | 조장이 조원 수신처를 포함해 담당교수 메일로 제출 |
| 3. 최종 결과물 | 최종 완성 코드, 결과 보고서 | 11/20(금) 23:59 · 4점 | 최종 결과물 전체 | 과제 채널에 ZIP 파일 업로드 |
| 4. 최종 발표 | 조별 15분 발표 | 11/24(화) 수업 시간 · 최대 10점 | 발표 자료 | 수업 시간 발표 |

담당교수 메일: `na.seungwon@dankook.ac.kr`

### 3단계 최종 제출 서류

팀장 1명만 대표 제출합니다. 제출 기한은 11/20(금) 23:59까지입니다.

- **발표 자료:** PPT 파일 1부(연동 URL 포함)
- **프로젝트 요약 보고서:** A4 2장 이내 1부 (DOC/HWP/PDF 중 1개)
- **소스코드:** `.py` 또는 `.ipynb` 소스코드와 실행에 필요한 데이터·리소스 파일 일체를 ZIP으로 제출
- **프로그램 시연 영상:** 선택 사항 (실행 오류 대비용)
- **공통 필수 기재:** 문서 말미에 데이터 출처와 참고 문헌(API 문서, 데이터셋 출처 URL 등) 표기

### 평가 방법

| 항목 | 배점 |
| --- | --- |
| 발표자료·프로그램 완성도 및 동작 | 4점 |
| 코드 구현 이해도 및 질의응답 | 4점 |
| 발표 전달력 및 협업 | 2점 |

### 현재 2차 준비물

- 요약서: `docs/DKU-SWAI_모델보고서.docx`
- 초안 PPT: `docs/DKU-SWAI_발표자료.pptx`
- 소스코드 일부: `src/train.py`, `src/train_explained.py`

`DKU-SWAI_모델보고서.docx`는 2차 중간 진척용입니다. 3단계 최종 보고서는 A4 2장 이내로 별도 작성해야 합니다.

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
├── README.md          # 프로젝트 소개·과제 안내·참고 자료
├── requirements.txt   # Python 의존성
├── src/train.py       # 실행용 코드 (주석 없음)
├── src/train_explained.py # 같은 로직에 함수별 설명·주석 포함
├── data/lung_cancer.csv # 저장소에 포함된 설문 CSV
├── artifacts/         # 실행 시 생성되는 모델·평가 결과 (Git에서 제외)
└── docs/              # 보고서·발표 자료·역할별 코드 설명
```

## 참고 자료

- 데이터: [Kaggle Lung Cancer 설문 데이터](https://www.kaggle.com/datasets/mysarahmadbhat/lung-cancer)
- 개발: [Python 공식 문서](https://docs.python.org/3/), [pandas 문서](https://pandas.pydata.org/docs/), [scikit-learn 문서](https://scikit-learn.org/stable/)
- 모델 선택: [scikit-learn Model selection](https://scikit-learn.org/stable/model_selection.html), [GradientBoostingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)
