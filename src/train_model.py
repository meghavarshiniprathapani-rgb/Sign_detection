import os
import cv2
import numpy as np
import joblib
import mediapipe as mp

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

base_dir = os.path.dirname(os.path.dirname(__file__))
image_root = os.path.join(base_dir, "dataset")
model_path = os.path.join(base_dir, "models", "naruto_seal_model.pkl")

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.7
)

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

X = []
y = []

print("Extracting landmarks with advanced augmentation...\n")

for label in os.listdir(image_root):
    label_dir = os.path.join(image_root, label)

    if not os.path.isdir(label_dir):
        continue

    for img_name in os.listdir(label_dir):
        img_path = os.path.join(label_dir, img_name)
        image = cv2.imread(img_path)

        if image is None:
            continue

        augmented_images = []

        augmented_images.append(image)

        flipped = cv2.flip(image, 1)
        augmented_images.append(flipped)

        h, w = image.shape[:2]

        matrix = cv2.getRotationMatrix2D((w//2, h//2), 10, 1.0)
        rotated = cv2.warpAffine(image, matrix, (w, h))
        augmented_images.append(rotated)

        bright = cv2.convertScaleAbs(image, alpha=1.1, beta=20)
        augmented_images.append(bright)

        scale = 1.1
        zoom_matrix = cv2.getRotationMatrix2D((w//2, h//2), 0, scale)
        zoomed = cv2.warpAffine(image, zoom_matrix, (w, h))
        augmented_images.append(zoomed)

        for aug_img in augmented_images:

            rgb = cv2.cvtColor(aug_img, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            left = np.zeros(63)
            right = np.zeros(63)

            if result.multi_hand_landmarks and result.multi_handedness:
                for idx, hand_landmarks in enumerate(result.multi_hand_landmarks):
                    # In real life, camera flips or hand crossing can confuse "Left" vs "Right".
                    # A robust method is just sorting by wrist x-coordinate if we have 2 hands
                    pass
                
                # If we have exactly 2 hands, sort them by x-coordinate: left screen vs right screen
                if len(result.multi_hand_landmarks) == 2:
                    hands_data = []
                    for hand_landmarks in result.multi_hand_landmarks:
                        coords = []
                        for lm in hand_landmarks.landmark:
                            coords.extend([lm.x, lm.y, lm.z])
                        hands_data.append((hand_landmarks.landmark[0].x, np.array(coords)))
                    
                    # Sort by X coordinate: index 0 is left-most on screen, index 1 is right-most
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

            # If there's only 1 hand, rel_features won't be defined. Only process if 2 hands.
            if result.multi_hand_landmarks and len(result.multi_hand_landmarks) == 2:
                feature_vector = np.concatenate([left, right, left_angles, right_angles, rel_features])

                if np.sum(feature_vector) != 0:
                    X.append(feature_vector)
                    y.append(label)

hands.close()

X = np.array(X)
y = np.array(y)

print("Total augmented samples:", len(X))

if len(X) == 0:
    print("No valid hand landmarks detected. Training aborted.")
    exit()

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("rf", RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        random_state=42,
        n_jobs=-1,
        verbose=0
    ))
])

print("\nTraining model...\n")
model.fit(X_train, y_train)

train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)

print("\nTraining Accuracy:", round(train_acc * 100, 2), "%")
print("Validation Accuracy:", round(test_acc * 100, 2), "%")

print("\nClassification Report:\n")
print(classification_report(y_test, model.predict(X_test)))

print("Confusion Matrix:\n")
print(confusion_matrix(y_test, model.predict(X_test)))

os.makedirs(os.path.dirname(model_path), exist_ok=True)
joblib.dump(model, model_path)

print("\nModel saved to:", model_path)
print("Training complete.")