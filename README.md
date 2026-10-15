# DKU-SWAI — Kaggle 폐암 설문 데이터 기반 분류 모델

단국대학교 **SW·AI입문(SW) 2차 팀 프로젝트** Kaggle의 폐암 설문 데이터에 있는 Features을 이용해 `LUNG_CANCER`(YES/NO)를 분류하는 모델을 만들고 평가합니다.

## 팀 구성 — 4조

| 구분 | 이름 | 소속 | 담당 역할 |
| --- | --- | --- | --- |
| 팀장 | 박인찬 | 퇴계혁신칼리지 (SW융합계열광역) | 프로그램 구현,발표 |
| 팀원 | 이서준 | 컴퓨터공학과 | 프로그램 구조 설계,발표 |
| 팀원 | 유소현 | 컴퓨터공학과 | 자료 조사, PPT, 발표 |

## 과제 요건

- **배점:** 총 20점.
- **파이썬 기본 문법:** 자료형·자료구조(리스트·딕셔너리 등), 조건문(`if/elif/else`), 반복문(`for/while`), 함수와 클래스를 적용해야 합니다.
- **확장 활용:** 아래 중 하나 이상을 사용해야 합니다.
  - 공공데이터/오픈 API 연동 (`requests` 등)
  - 데이터 분석 및 시각화 (`pandas`, `matplotlib` 등)
  - 파일 처리 또는 경량 DB 연동 (CSV, JSON, `sqlite3` 등)
  - 대화형/시각적 인터페이스 (Streamlit, tkinter, turtle 등)
- **제출 코드 형태:** Python 소스코드(`.py`/`.ipynb`) 또는 Streamlit 앱.
- **보고서:** 팀원별 구체적인 역할,활동을 명시하고, 문서 말미에 데이터 출처와 참고 문헌을 표기합니다.

## 일정과 제출물

1·2단계는 팀장이 다른 팀원 2명을 참조에 포함해 담당 교수(`na.seungwon@dankook.ac.kr`)에게 메일로 제출합니다.

| 단계 | 기한·배점 | 제출물과 제출 방법 | 준비 사항 |
| --- | --- | --- | --- |
| 1. 기획서 | 10/21(수) 23:59 · 3점 | 과제 요약서(Word/HWP), 팀장 메일 제출 | 기능 정의, 실행 흐름도, 개발 환경, 구체적 역할 분담, 출처 표기 |
| 2. 중간 진척 | 11/11(수) 23:59 · 3점 | 요약서·초안 PPT·소스코드 일부, 팀장 메일 제출 | 구현 진행률, 핵심 함수 동작, 남은 구현 계획, 출처 표기 |
| 3. 최종 결과물 | 11/20(금) 23:59 · 4점 | 팀장 1명이 과제 채널에 ZIP 업로드 | 아래 최종 제출물 목록과 실행 검증 |
| 4. 최종 발표 | 11/24(화) 수업 시간 · 최대 10점 | 조별 15분 발표 | 파트 분배, 리허설, 시연 및 질의응답 준비 |

**최종 ZIP 준비:** 발표 PPT 1부(연동 URL 포함), A4 2장 이내의 요약 보고서 1부(DOC/HWP/PDF 중 하나), 소스코드와 실행에 필요한 데이터·리소스 파일을 포함합니다. 시연 영상은 선택 사항입니다. 보고서에는 팀원별 실제 역할을, 모든 문서 말미에는 데이터 출처와 참고 문헌을 적습니다. 제출 전 다른 PC에서 ZIP을 풀어 실행을 확인합니다.

발표 평가는 자료·프로그램 완성도 및 동작 4점, 코드 이해도·질의응답 4점, 전달력·협업 2점입니다. 팀원 모두 자신이 맡은 코드를 설명할 수 있어야합니다.

## 현재 구현과 개발 계획

1. CSV의 필수 열과 값 범위를 확인하고 완전히 동일한 행을 제거
2. 설문 응답을 숫자로 변환하고 정답 비율을 유지해 학습·평가 데이터를 80:20으로 split
3. 학습 데이터로만 스케일러를 적합한 뒤, 클래스 불균형을 고려한 로지스틱 회귀 기준 모델을 학습
4. 평가 데이터에서 정확도, YES 클래스 정밀도·재현율·F1, ROC AUC, 혼동 행렬을 기록

학습·평가·결과 저장 흐름은 `LungCancerTrainer` 클래스로 관리합니다.

## 데이터 준비

- **출처:** [Kaggle — Lung Cancer (mysarahmadbhat)](https://www.kaggle.com/datasets/mysarahmadbhat/lung-cancer)
- **정답 열:** `LUNG_CANCER` (`YES`/`NO`)
- **기본 정보:** `GENDER` (`F`/`M`), `AGE` (1~120 사이 숫자)
- **설문 항목:** `SMOKING`, `YELLOW_FINGERS`, `ANXIETY`, `PEER_PRESSURE`, `CHRONIC DISEASE`, `FATIGUE`, `ALLERGY`, `WHEEZING`, `ALCOHOL CONSUMING`, `COUGHING`, `SHORTNESS OF BREATH`, `SWALLOWING DIFFICULTY`, `CHEST PAIN` (각각 1=아니오, 2=예)

이 저장소에는 `data/lung_cancer.csv`가 포함되어 있어 별도 다운로드 없이 실행할 수 있습니다. 다른 버전의 설문 CSV를 사용하려면 Kaggle 페이지의 이용 조건을 확인해 내려받고, 실행 시 `--data`에 파일 경로를 지정합니다. 실제 제출물에는 사용한 데이터셋 버전과 다운로드 날짜를 적습니다. 열 이름의 앞뒤 공백은 코드가 제거하지만, 다른 버전의 데이터라면 열과 인코딩을 먼저 확인해야 합니다.

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

다른 CSV를 사용한다면 마지막 명령에 `--data 경로/파일.csv`를 추가합니다. `src/train.py`는 설문 CSV를 검증하고 학습·평가합니다. 실행 결과는 `artifacts/metrics.json`과 `artifacts/model.joblib`에 저장됩니다. 입력에 필수 열이 없거나 값 인코딩이 다르면 오류를 냅니다. 평가 점수는 한 번 분할한 테스트 데이터의 결과입니다.

## 저장소 구성 원칙

```text
.
├── README.md          # 프로젝트·과제 안내와 참고 자료를 한곳에 정리
├── requirements.txt   # Python 의존성
├── src/train.py       # 기준 모델 학습·평가 코드
├── data/lung_cancer.csv # 저장소에 포함된 설문 CSV
├── artifacts/         # 실행 시 생성되는 모델·평가 결과 (Git에서 제외)
└── docs/              # 실제 제출용 기획서·보고서·발표 자료만 보관
```

현재 `docs/`에 실제 제출물이 없으므로 빈 폴더를 유지하기 위한 안내 파일을 두지 않습니다. 제출물이 준비되면 단계별 파일을 `docs/`에 추가합니다. 

## 참고 자료

- 데이터: [Kaggle Lung Cancer 설문 데이터](https://www.kaggle.com/datasets/mysarahmadbhat/lung-cancer)
- 개발: [Python 공식 문서](https://docs.python.org/3/), [pandas 문서](https://pandas.pydata.org/docs/), [scikit-learn 문서](https://scikit-learn.org/stable/)
