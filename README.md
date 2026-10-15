# DKU-SWAI — Kaggle 폐암 설문 데이터 기반 분류 모델

단국대학교 **SW·AI입문(SW)** 2차 팀 프로젝트입니다. Kaggle의 폐암 설문 데이터에서 나이, 생활 습관, 증상 등의 항목으로 `LUNG_CANCER`(YES/NO)를 예측하는 기초 분류 모델을 학습하고 평가합니다.

> 이 프로젝트는 수업용 데이터 분석 실습입니다. 모델 출력은 실제 폐암 진단이나 의료 판단에 사용할 수 없습니다.

- 과제 공지: [`docs/assignment.md`](docs/assignment.md)
- 데이터 출처: [`docs/references.md`](docs/references.md)

## 팀 구성 — 4조

우리 팀은 예외적으로 3인 1조입니다. 역할은 프로그램 함수 구현, 구조 설계, 자료 조사·발표 준비로 나누어 진행하며, 최종 보고서에는 실제 수행한 세부 역할을 기록합니다.

## 프로젝트 목표와 진행 흐름

1. [Kaggle Lung Cancer 데이터셋](https://www.kaggle.com/datasets/mysarahmadbhat/lung-cancer)의 CSV를 내려받습니다.
2. 데이터 열과 값 범위를 확인하고, 완전히 동일한 행을 제거합니다.
3. 입력 항목을 숫자로 변환한 뒤 `LUNG_CANCER`를 정답으로 사용합니다.
4. 정답 비율을 유지해 학습·평가 데이터를 나눕니다. 스케일링은 학습 데이터로만 적합합니다.
5. 클래스 불균형을 고려한 로지스틱 회귀 기준 모델을 학습합니다.
6. 보류한 평가 데이터에서 정확도, YES 클래스 정밀도·재현율·F1, ROC AUC, 혼동 행렬을 기록합니다.

현재는 **기준 모델 코드와 실행 절차**를 준비한 단계입니다. 실제 데이터로 학습한 결과와 점수는 아직 저장소에 게시하지 않았습니다. 작은 설문 데이터의 점수만으로 임상 성능을 주장하지 않습니다.

## 실행 방법

Python 3.10 이상을 권장합니다. 저장소 루트에서 다음 명령을 실행합니다.

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python src/train.py --data data/survey_lung_cancer.csv
```

Kaggle에서 내려받은 설문 CSV를 `data/survey_lung_cancer.csv`로 저장하세요. 원본 파일명이 다르면 이름을 바꾸거나 `--data`에 해당 경로를 지정하면 됩니다. 개인 데이터와 데이터셋 파일은 Git에 올리지 않습니다.

실행하면 `artifacts/metrics.json`과 `artifacts/model.joblib`이 생성됩니다. CSV 열 이름의 앞뒤 공백은 허용하지만 필수 열이 없거나 인코딩 값이 다르면 오류를 내도록 했습니다. 파일 형식과 인코딩은 [`data/README.md`](data/README.md)를 확인하세요.

## 과제 요구 사항과 다음 작업

- 기본 문법: 함수, 조건문, 반복문, 자료구조를 학습 코드에 사용했습니다. **클래스 적용**은 후속 구현에서 추가해야 합니다.
- 확장 활용: CSV 파일 처리와 `pandas`를 이용한 데이터 분석을 사용합니다. 시각화는 후속 작업입니다.
- 다음 단계: 실제 CSV의 품질·클래스 분포 확인, 탐색적 시각화, 모델 비교, 학습 결과 기록, 시연 방식 결정.

| 단계 | 내용 | 기한 | 배점 | 제출 방법 |
| --- | --- | --- | --- | --- |
| 1단계 | 프로젝트 기획서 | 10/21(수) 23:59 | 3점 | 팀장 → 교수 메일 (팀원 참조) |
| 2단계 | 중간 진척 보고서 | 11/11(수) 23:59 | 3점 | 팀장 → 교수 메일 (팀원 참조) |
| 3단계 | 최종 결과물 | 11/20(금) 23:59 | 4점 | 과제 채널에 zip 업로드 |
| 4단계 | 최종 발표 (15분) | 11/24(화) 수업 시간 | 최대 10점 | 발표 |

제출 연락처와 단계별 세부 안내는 [`docs/assignment.md`](docs/assignment.md), 단계별 체크리스트는 `docs/`의 해당 폴더를 확인하세요.

## 저장소 구조

```text
.
├── README.md
├── requirements.txt
├── src/
│   └── train.py                # 기준 모델 학습 및 평가
├── data/
│   └── README.md               # CSV 준비 방법 (CSV는 Git에서 제외)
├── docs/
│   ├── assignment.md           # 과제 공지 정리
│   ├── references.md           # 데이터 출처와 참고 문헌
│   ├── 1_proposal/
│   ├── 2_progress/
│   ├── 3_final/
│   └── 4_presentation/
└── artifacts/                  # 실행 시 생성 (Git에서 제외)
```
