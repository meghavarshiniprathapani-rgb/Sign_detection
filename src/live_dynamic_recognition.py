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

def extract_angles(hand_vector):
    if np.sum(hand_vector) == 0:
        return np.zeros(5)

    try:
        hand_vector = hand_vector.reshape(21, 3)
    except Exception:
        return np.zeros(5)
        
    angles = []
    
    # Finger bases: Thumb=2, Index=5, Middle=9, Ring=13, Pinky=17
    finger_bases = [2, 5, 9, 13, 17]
    # Finger PIPs (first joint): Thumb=3, Index=6, Middle=10, Ring=14, Pinky=18
    finger_pips = [3, 6, 10, 14, 18]
    wrist = hand_vector[0]

    for base, pip in zip(finger_bases, finger_pips):
        v1 = hand_vector[base] - wrist
        v2 = hand_vector[pip] - hand_vector[base]
        
        # Calculate angle between v1 and v2
        cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        angles.append(angle)
        
    return np.array(angles)

def extract_relative_features(left_raw, right_raw):
    # Base scale on the size of the overall left hand bounding width/height approx
    # or distance between wrist and middle MCP (index 9)
    scale = np.linalg.norm(left_raw[9] - left_raw[0])
    if scale == 0:
        scale = 1.0
        
    # Key contact points in Naruto Seals: Wrists, Thumbs, Index, Middle, Ring, Pinky
    # Indices: 0, 4, 8, 12, 16, 20
    contact_points = [0, 4, 8, 12, 16, 20]
    
    rel_features = []
    for pt in contact_points:
        # Scale-invariant distance between the identical fingertips of both hands
        dist = np.linalg.norm(left_raw[pt] - right_raw[pt]) / scale
        rel_features.append(dist)
        
    return np.array(rel_features)

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

prediction_buffer = deque(maxlen=30)

cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as hands:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        left = np.zeros(63)
        right = np.zeros(63)
        prediction_made = False

        if result.multi_hand_landmarks:
            # Draw landmarks for ALL detected hands, regardless of validity rules
            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Apply rules for classification: exactly 2 hands
            if len(result.multi_hand_landmarks) == 2:
                # Remove distance restriction because seals naturally involve close proximity/touching
                # Instead, sort by X coordinate to consistently assign "left side" vs "right side"
                # This fixes the bug where MediaPipe gets Left/Right confused when palms face or cross over
                
                hands_data = []
                for hand_landmarks in result.multi_hand_landmarks:
                    coords = []
                    for lm in hand_landmarks.landmark:
                        coords.extend([lm.x, lm.y, lm.z])
                    hands_data.append((hand_landmarks.landmark[0].x, np.array(coords)))
                
                # Sort: lower x value means it's on the left side of the screen
                hands_data.sort(key=lambda x: x[0])
                
                left = hands_data[0][1]
                right = hands_data[1][1]
                
                left_angles = extract_angles(left)
                right_angles = extract_angles(right)
                
                # Compute spatial relationship before destroying it via normalization
                left_raw = left.reshape(21, 3)
                right_raw = right.reshape(21, 3)
                rel_features = extract_relative_features(left_raw, right_raw)

                left = normalize_hand(left)
                right = normalize_hand(right)

                feature_vector = np.concatenate([left, right, left_angles, right_angles, rel_features]).reshape(1, -1)

                probabilities = model.predict_proba(feature_vector)[0]
                max_prob = np.max(probabilities)
                prediction = model.classes_[np.argmax(probabilities)]
                
                cv2.putText(
                    frame,
                    f"Raw Prob: {max_prob:.2f} | Pred: {prediction}",
                    (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2
                )
                
                # Confidence Threshold (85% as specified)
                if max_prob > 0.85:
                    prediction_buffer.append(prediction)
                    prediction_made = True

        if not prediction_made:
            prediction_buffer.append("...")

        if len(prediction_buffer) > 0:
            most_common, count = Counter(prediction_buffer).most_common(1)[0]
            # Requires at least 20/30 matching frames (majority vote smoothing + hold time)
            if count >= 20 and most_common != "...":
                display_seal = most_common
            else:
                display_seal = "..."
        else:
            display_seal = "..."

        cv2.putText(
            frame,
            f"Seal: {display_seal}",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        
        hands_detected = len(result.multi_hand_landmarks) if result.multi_hand_landmarks else 0
        cv2.putText(
            frame,
            f"Hands Detected: {hands_detected}",
            (10, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )

        cv2.imshow("Naruto Dynamic Seal Recognition", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()