import time
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from feature_extractor import extract_features

CSV_PATH = "PhiUSIIL_Phishing_URL_Dataset.csv"
OUTPUT_PATH = "phishing_model.pkl"

print("Loading dataset...")
df = pd.read_csv(CSV_PATH)

print("Computing features (using the same function app.py will use)...")
t0 = time.time()
X = np.array([extract_features(u) for u in df["URL"]])
y = df["label"].values
groups = df["Domain"].values
print(f"  done in {time.time()-t0:.1f}s - {X.shape[0]} rows, {X.shape[1]} features")

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=groups))
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

print("Training with regularized settings (deliberately less flexible, to avoid memorizing narrow rules)...")
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    min_samples_leaf=50,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1,
)
t0 = time.time()
model.fit(X_train, y_train)
print(f"  done in {time.time()-t0:.1f}s")

pred = model.predict(X_test)

print()
print("=== Held-out test performance (domain-grouped, honest split) ===")
print("Accuracy: ", accuracy_score(y_test, pred))
print("Precision:", precision_score(y_test, pred))
print("Recall:   ", recall_score(y_test, pred))
print("F1:       ", f1_score(y_test, pred))

joblib.dump(model, OUTPUT_PATH)
print()
print(f"Saved {OUTPUT_PATH} - restart app.py to use the new model.")