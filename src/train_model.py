import os
import glob
import cv2
import numpy as np
from sklearn.svm import SVC
import joblib

base_dir = os.path.dirname(os.path.dirname(__file__))
train_dir = os.path.join(base_dir, "dataset", "train")
model_path = os.path.join(base_dir, "models", "naruto_seal_model.pkl")

X = []
y = []

classes = os.listdir(train_dir)

for label in classes:
    class_dir = os.path.join(train_dir, label)
    if not os.path.isdir(class_dir):
        continue

    for img_path in glob.glob(os.path.join(class_dir, "*")):
        img = cv2.imread(img_path)
        if img is None:
            continue

        img = cv2.resize(img, (64, 64))
        img = img.flatten()

        X.append(img)
        y.append(label)

X = np.array(X)
y = np.array(y)

model = SVC(kernel="rbf", probability=True)
model.fit(X, y)

os.makedirs(os.path.dirname(model_path), exist_ok=True)
joblib.dump(model, model_path)

print("model trained and saved to models/")