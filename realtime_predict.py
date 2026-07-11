import cv2
import mediapipe as mp
import numpy as np
from collections import deque
from tensorflow.keras.models import load_model

# ==========================
# 설정
# ==========================
MODEL_PATH = "tcn_landmark_classifier.keras"

SEQUENCE_LENGTH = 300
FEATURE_DIM = 132

# 학습할 때 label encoder를 저장하지 않았다면
# 순서는 직접 수정해야 한다.
CLASS_NAMES = ["body", "hand", "head"]

# ==========================
# 모델 로드
# ==========================
print("모델 로드 중...")
model = load_model(MODEL_PATH)
print("완료!")

# ==========================
# MediaPipe
# ==========================
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

sequence = deque(maxlen=SEQUENCE_LENGTH)

# ==========================
# 카메라
# ==========================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()

print("Q를 누르면 종료")

# ==========================
# 실시간 추론
# ==========================
while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = pose.process(rgb)

    if result.pose_landmarks:

        mp_draw.draw_landmarks(
            frame,
            result.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

        feature = []

        for lm in result.pose_landmarks.landmark:
            feature.extend([
                lm.x,
                lm.y,
                lm.z,
                lm.visibility
            ])

        feature = np.array(feature, dtype=np.float32)

        sequence.append(feature)

        cv2.putText(
            frame,
            f"Frames : {len(sequence)}/{SEQUENCE_LENGTH}",
            (20,40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,255,255),
            2
        )

        if len(sequence) == SEQUENCE_LENGTH:

            x = np.array(sequence)
            x = np.expand_dims(x, axis=0)

            pred = model.predict(x, verbose=0)[0]

            idx = np.argmax(pred)

            label = CLASS_NAMES[idx]
            confidence = pred[idx]

            cv2.putText(
                frame,
                f"{label} ({confidence*100:.1f}%)",
                (20,80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,255,0),
                2
            )

    cv2.imshow("Autism Repetitive Behavior Detection", frame)

    key = cv2.waitKey(1)

    if key & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()