import cv2
import mediapipe as mp
import joblib
import os

base_dir = os.path.dirname(os.path.dirname(__file__))
model_path = os.path.join(base_dir, "models", "naruto_seal_model.pkl")

model = joblib.load(model_path)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as hands:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]

            landmarks = []
            for lm in hand.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            prediction = model.predict([landmarks])[0]

            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            cv2.putText(
                frame,
                prediction,
                (10,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,255,0),
                2
            )

        cv2.imshow("Naruto Seal Recognition", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()