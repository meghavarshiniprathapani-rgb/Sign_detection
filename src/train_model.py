import os
import cv2
import numpy as np
import joblib
import mediapipe as mp

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

base_dir = os.path.dirname(os.path.dirname(__file__))
image_root = os.path.join(base_dir, "dataset", "test")
model_path = os.path.join(base_dir, "models", "naruto_seal_model.pkl")

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5
)

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
                    lr_label = result.multi_handedness[idx].classification[0].label

                    coords = []
                    for lm in hand_landmarks.landmark:
                        coords.extend([lm.x, lm.y, lm.z])

                    if lr_label == "Left":
                        left = np.array(coords)
                    else:
                        right = np.array(coords)

            left = normalize_hand(left)
            right = normalize_hand(right)

            feature_vector = np.concatenate([left, right])

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
    ("mlp", MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation="relu",
        solver="adam",
        learning_rate_init=0.0008,
        alpha=0.0005,
        max_iter=350,
        random_state=42,
        verbose=True
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