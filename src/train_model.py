import os
import glob
import pandas as pd
from sklearn.svm import SVC
import joblib

base_dir = os.path.dirname(os.path.dirname(__file__))
dataset_dir = os.path.join(base_dir, "dataset")
model_path = os.path.join(base_dir, "models", "naruto_seal_model.pkl")

X = []
y = []

for file in glob.glob(os.path.join(dataset_dir, "*.csv")):
    label = os.path.basename(file).replace(".csv","")
    data = pd.read_csv(file, header=None)

    X.extend(data.values)
    y.extend([label]*len(data))

model = SVC(kernel="rbf", probability=True)
model.fit(X,y)

os.makedirs(os.path.dirname(model_path), exist_ok=True)
joblib.dump(model, model_path)

print("model trained and saved to models/")