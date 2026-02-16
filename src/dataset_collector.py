import cv2
import mediapipe as mp
import csv
import os

seals = ["tiger","boar","ram","snake","dog","bird","horse","monkey"]
current_seal = 0

base_dir = os.path.dirname(os.path.dirname(__file__))
dataset_dir = os.path.join(base_dir, "dataset")

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

            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            cv2.putText(
                frame,
                f"Seal: {seals[current_seal]}",
                (10,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,255,0),
                2
            )

        cv2.imshow("Dataset Collector", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("s") and result.multi_hand_landmarks:
            file_path = os.path.join(dataset_dir, f"{seals[current_seal]}.csv")
            with open(file_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(landmarks)
            print("saved", seals[current_seal])

        if key == ord("n"):
            current_seal = (current_seal + 1) % len(seals)

        if key == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()