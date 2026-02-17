import os
import cv2
import numpy as np
import joblib
import mediapipe as mp
import time
from collections import deque, Counter

base_dir = os.path.dirname(os.path.dirname(__file__))
model_path = os.path.join(base_dir, "models", "naruto_seal_model.pkl")

model = joblib.load(model_path)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

def normalize_hand(hand_vector):
    if np.sum(hand_vector) == 0:
        return hand_vector

    hand_vector = hand_vector.reshape(21, 3)
    wrist = hand_vector[0]
    hand_vector = hand_vector - wrist

    scale = np.linalg.norm(hand_vector[9])
    if scale != 0:
        hand_vector = hand_vector / scale

    return hand_vector.flatten()

prediction_buffer = deque(maxlen=20)
seal_start_time = None
confirmed_seal = None

CONFIDENCE_THRESHOLD = 0.80
HOLD_TIME = 0.8

cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        h, w, _ = frame.shape

        if result.multi_hand_landmarks and len(result.multi_hand_landmarks) >= 1:

            all_points = []
            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                for lm in hand_landmarks.landmark:
                    all_points.append([lm.x * w, lm.y * h])

            all_points = np.array(all_points)
            x_min, y_min = np.min(all_points, axis=0)
            x_max, y_max = np.max(all_points, axis=0)

            area = (x_max - x_min) * (y_max - y_min)

            if area > 12000:

                left = np.zeros(63)
                right = np.zeros(63)

                for idx, hand_landmarks in enumerate(result.multi_hand_landmarks):
                    coords = []
                    for lm in hand_landmarks.landmark:
                        coords.extend([lm.x, lm.y, lm.z])

                    if idx == 0:
                        left = np.array(coords)
                    elif idx == 1:
                        right = np.array(coords)

                left = normalize_hand(left)
                right = normalize_hand(right)

                feature_vector = np.concatenate([left, right]).reshape(1, -1)

                probs = model.predict_proba(feature_vector)[0]
                best_index = np.argmax(probs)
                confidence = probs[best_index]
                prediction = model.classes_[best_index]

                if confidence > CONFIDENCE_THRESHOLD:
                    prediction_buffer.append(prediction)

                    most_common = Counter(prediction_buffer).most_common(1)[0][0]

                    if confirmed_seal != most_common:
                        seal_start_time = time.time()
                        confirmed_seal = most_common

                    if time.time() - seal_start_time > HOLD_TIME:
                        cv2.putText(
                            frame,
                            f"Seal: {most_common}",
                            (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2
                        )
                else:
                    prediction_buffer.clear()
                    confirmed_seal = None
                    seal_start_time = None

            else:
                prediction_buffer.clear()
                confirmed_seal = None
                seal_start_time = None

        else:
            prediction_buffer.clear()
            confirmed_seal = None
            seal_start_time = None

        cv2.imshow("Naruto Seal Recognition (Stable Mode)", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()