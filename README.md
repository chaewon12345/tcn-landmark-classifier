# 자폐아동의 반복행동 감지 서비스
자페 아동의 스트레스 관련 상황을 분석하고 감정 조절을 도울 수 있도록 웹캠을 통해 자폐 아동의 반복 행동을 인식하는 알고리즘입니다.

## TCN Landmark Classifier

MediaPipe Pose로 추출한 신체 랜드마크 시퀀스를 TCN(Temporal Convolutional Network) 기반 모델로 분류하는 실시간 반복 행동 인식합니다.

## 주요 기능

- 웹캠 영상에서 MediaPipe Pose로 33개 신체 랜드마크(x, y, z, visibility)를 실시간 추출
- 300프레임(약 10초 내외) 시퀀스를 누적하여 학습된 TCN 모델로 행동 분류
- 분류 결과와 신뢰도(confidence)를 화면에 실시간 표시

분류 클래스:
- `body` — 몸 관련 반복 행동
- `hand` — 손 관련 반복 행동
- `head` — 머리 관련 반복 행동

## Training Pipeline

본 모델은 MediaPipe Pose를 통해 추출한 신체 랜드마크 시퀀스를 입력으로 받아 TCN(Temporal Convolutional Network) 기반 분류 모델로 자폐 아동의 반복 행동 유형을 분류하도록 학습되었습니다.

### 1. Dataset 구성

반복 행동 데이터를 크롤링, 직접 촬영하는 방식으로 수집했습니다.
학습 데이터 부족 문제를 완화하고 모델의 일반화 성능을 향상시키기 위해 OpenCV 기반 영상 증강(Horizontal Flip, Brightness Adjustment, Scaling)을 적용했습니다.

| Class | Description |
|------|-------------|
| body | 몸 전체 움직임과 관련된 반복 행동 |
| hand | 손 움직임과 관련된 반복 행동 |
| head | 머리 움직임과 관련된 반복 행동 |

총 270개의 행동 영상 데이터를 사용했으며, 각 클래스별 90개의 영상으로 구성했습니다.


## 요구 사항

- Python 3.10
- macOS (Apple Silicon 기준, Metal GPU 가속 지원)
- 웹캠

## 설치

Conda 가상환경 생성 및 활성화:

```bash
conda create -n tcn-landmark-classifier python=3.10
conda activate tcn-landmark-classifier
```

필수 패키지 설치 (버전 호환성 검증 완료된 조합):

```bash
pip install tensorflow-macos==2.16.2 tensorflow-metal==1.2.0
pip install mediapipe==0.10.14
pip install "numpy<2.0.0,>=1.23.5"
pip install "ml-dtypes>=0.5.0"
pip install "opencv-contrib-python==4.10.0.84"
```

> ⚠️ **버전 관련 주의사항**
> - `mediapipe`는 0.10.3x 이상 최신 버전에서 `mp.solutions`가 정상 동작하지 않는 알려진 버그가 있어 `0.10.14`로 고정했습니다.
> - `opencv-contrib-python` 5.x / 최신 4.11+ 버전은 `numpy>=2`를 요구해 `tensorflow`(`numpy<2.0.0` 요구)와 충돌하므로 `4.10.0.84`로 고정했습니다.
> - `ml-dtypes`는 `tensorflow`가 요구하는 `~=0.3.1`보다 `mediapipe`가 내부적으로 사용하는 `jax`가 요구하는 `>=0.5.0`을 우선해야 임포트 에러가 발생하지 않습니다. `pip install` 시 tensorflow 관련 dependency 충돌 경고가 뜨지만 무시해도 됩니다.

## 사용 방법

1. 학습된 모델 파일(`tcn_landmark_classifier.keras`)을 프로젝트 루트에 위치시킵니다.
2. 실시간 추론 스크립트를 실행합니다.

```bash
python realtime_predict.py
```

3. 웹캠 화면이 열리고 신체 랜드마크가 초록색으로 표시됩니다.
4. 화면 상단에 `Frames: N/300` 카운터가 누적되며, 300프레임이 채워지면 분류 결과와 신뢰도(`%`)가 표시됩니다.
5. `Q` 키를 누르면 종료됩니다.

## 설정값 (`realtime_predict.py` 상단)

| 변수 | 값 | 설명 |
|---|---|---|
| `MODEL_PATH` | `tcn_landmark_classifier.keras` | 학습된 모델 파일 경로 |
| `SEQUENCE_LENGTH` | `300` | 추론에 사용할 프레임 시퀀스 길이 |
| `FEATURE_DIM` | `132` | 프레임당 특징 차원 (33개 랜드마크 × 4) |
| `CLASS_NAMES` | `["body", "hand", "head"]` | 분류 클래스 순서 (학습 시 label encoder 순서와 반드시 일치해야 함) |


## 프로젝트 구조

```
tcn-landmark-classifier/
├── realtime_predict.py       # 실시간 웹캠 추론 스크립트
├── tcn_landmark_classifier.keras  # 학습된 모델 (별도 관리, .gitignore 대상)
├── .gitignore
└── README.md
```
