import os
import cv2
import numpy as np
import joblib
import mediapipe as mp
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

prediction_buffer = deque(maxlen=15)

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

        left = None
        right = None

        if result.multi_hand_landmarks and result.multi_handedness:
            for idx, hand_landmarks in enumerate(result.multi_hand_landmarks):
                lr_label = result.multi_handedness[idx].classification[0].label

                coords = []
                for lm in hand_landmarks.landmark:
                    coords.extend([lm.x, lm.y, lm.z])

                if lr_label == "Left":
                    left = np.array(coords)
                elif lr_label == "Right":
                    right = np.array(coords)

                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        if left is not None and right is not None:

            left = normalize_hand(left)
            right = normalize_hand(right)

            feature_vector = np.concatenate([left, right]).reshape(1, -1)

            prediction = model.predict(feature_vector)[0]
            prediction_buffer.append(prediction)

            most_common = Counter(prediction_buffer).most_common(1)[0][0]

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

            cv2.putText(
                frame,
                "Show BOTH hands for Naruto Seal",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        cv2.imshow("Naruto Two-Hand Dynamic Recognition", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()