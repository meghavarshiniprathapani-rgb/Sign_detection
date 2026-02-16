import cv2
import mediapipe as mp
import numpy as np  # noqa: F401

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

def normalize_lighting(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    l = cv2.equalizeHist(l)

    lab = cv2.merge((l, a, b))
    frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    frame = cv2.convertScaleAbs(frame, alpha=1.1, beta=-20)

    return frame

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.4,
    min_tracking_confidence=0.4
) as hands:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        frame = normalize_lighting(frame)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            for hand in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            status = "PAIR" if len(result.multi_hand_landmarks) >= 2 else "SINGLE"
            color = (0,255,0) if status=="PAIR" else (0,200,255)
        else:
            status = "NO HAND"
            color = (0,0,255)

        cv2.putText(frame, status, (10,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.imshow("Naruto Hand Tracking (Lighting Fixed)", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()