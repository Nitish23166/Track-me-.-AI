"""
Train XGBoost + Random Forest on V4 features (generalised & phone-aware).

Key improvements over V3:
  • Position/angle-invariant features (solvePnP head pose, normalised distances)
  • Dedicated phone-usage features
  • Reduced environment-dependent image features
  • Handles class imbalance with SMOTE + class weights
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from xgboost import XGBClassifier
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES_FILE = os.path.join(BASE_DIR, "features", "m_dataset_features_v4.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

TEST_SIZE = 0.2
RANDOM_STATE = 42
N_FOLDS = 5

# ─── Load ─────────────────────────────────────────────────────────────────────
print("=" * 60)
print("  V4 Training — Generalised & Phone-Aware Features")
print("=" * 60)
print(f"\nLoading features from {FEATURES_FILE} ...")
df = pd.read_csv(FEATURES_FILE)
df = df.fillna(0.0)

print(f"  Samples : {len(df)}")
print(f"  Classes : {df['label'].value_counts().to_dict()}")

X = df.drop(columns=["label", "filename"], errors="ignore")
y = df["label"]

le = LabelEncoder()
y_enc = le.fit_transform(y)
class_names = le.classes_
n_classes = len(class_names)
print(f"  Labels  : {dict(zip(class_names, le.transform(class_names)))}")
print(f"  Features: {X.shape[1]}")

# ─── Handle class imbalance ──────────────────────────────────────────────────
try:
    from imblearn.over_sampling import SMOTE
    sm = SMOTE(random_state=RANDOM_STATE)
    X_res, y_res = sm.fit_resample(X, y_enc)
    print(f"\n  SMOTE resampled: {len(X_res)} samples (from {len(X)})")
    use_smote = True
except ImportError:
    print("\n  imblearn not installed — using raw data (install with: pip install imbalanced-learn)")
    X_res, y_res = X.values, y_enc
    use_smote = False

X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_res,
)
print(f"  Train: {len(X_train)}, Test: {len(X_test)}")

# Compute class weights for models that support it
from collections import Counter
counts = Counter(y_train)
total = sum(counts.values())
class_weight_dict = {c: total / (n_classes * cnt) for c, cnt in counts.items()}


# ─── Evaluation helper ────────────────────────────────────────────────────────

def evaluate(model, name):
    print(f"\n{'=' * 60}\n  {name}\n{'=' * 60}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cv = cross_val_score(
        model, X_train, y_train,
        cv=StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE),
        scoring="accuracy",
    )
    print(f"\n  Accuracy : {acc:.4f} ({acc*100:.2f}%)")
    print(f"  {N_FOLDS}-Fold CV : {cv.mean():.4f} ± {cv.std():.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=class_names)}")
    cm = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix:\n{cm}")
    return acc, y_pred, cm


def save_cm(cm, names, title, path):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap=plt.cm.Blues)
    ax.set_title(title, fontsize=14)
    plt.colorbar(im)
    t = np.arange(len(names))
    ax.set_xticks(t); ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticks(t); ax.set_yticklabels(names)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    ax.set_ylabel("True"); ax.set_xlabel("Predicted")
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
    print(f"  Saved: {path}")


def save_fi(model, col_names, title, path, top_n=30):
    imp = model.feature_importances_
    idx = np.argsort(imp)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(idx)), imp[idx[::-1]])
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([col_names[i] for i in idx[::-1]])
    ax.set_xlabel("Importance"); ax.set_title(title)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
    print(f"  Saved: {path}")


# ─── Feature column names ────────────────────────────────────────────────────
if isinstance(X_res, pd.DataFrame):
    col_names = X_res.columns.tolist()
elif isinstance(X, pd.DataFrame):
    col_names = X.columns.tolist()
else:
    col_names = [f"f{i}" for i in range(X_train.shape[1])]


# ─── XGBoost ─────────────────────────────────────────────────────────────────
xgb = XGBClassifier(
    n_estimators=600, max_depth=8, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0,
    gamma=0.1,
    eval_metric="mlogloss", random_state=RANDOM_STATE, n_jobs=-1,
)
xgb_acc, _, xgb_cm = evaluate(xgb, "XGBoost (V4 — Generalised)")
joblib.dump(xgb, os.path.join(MODELS_DIR, "xgb_behaviour_model_v4.pkl"))
save_cm(xgb_cm, class_names, "XGBoost V4 — Confusion Matrix",
        os.path.join(MODELS_DIR, "xgb_v4_confusion_matrix.png"))
save_fi(xgb, col_names, "XGBoost V4 — Top 30 Features",
        os.path.join(MODELS_DIR, "xgb_v4_feature_importance.png"))

# ─── Random Forest ────────────────────────────────────────────────────────────
rf = RandomForestClassifier(
    n_estimators=600, max_depth=None,
    min_samples_split=3, min_samples_leaf=1,
    max_features="sqrt", class_weight="balanced",
    random_state=RANDOM_STATE, n_jobs=-1,
)
rf_acc, _, rf_cm = evaluate(rf, "Random Forest (V4 — Generalised)")
joblib.dump(rf, os.path.join(MODELS_DIR, "rf_behaviour_model_v4.pkl"))
save_cm(rf_cm, class_names, "Random Forest V4 — Confusion Matrix",
        os.path.join(MODELS_DIR, "rf_v4_confusion_matrix.png"))
save_fi(rf, col_names, "Random Forest V4 — Top 30 Features",
        os.path.join(MODELS_DIR, "rf_v4_feature_importance.png"))

# ─── Save label encoder ──────────────────────────────────────────────────────
joblib.dump(le, os.path.join(MODELS_DIR, "label_encoder_v4.pkl"))

# ─── Summary ─────────────────────────────────────────────────────────────────
print(f"\n{'=' * 60}")
print(f"  SUMMARY  (V4 — {X.shape[1]} generalised features)")
print(f"{'=' * 60}")
print(f"  XGBoost       : {xgb_acc * 100:.2f}%")
print(f"  Random Forest : {rf_acc * 100:.2f}%")
best = "XGBoost" if xgb_acc >= rf_acc else "Random Forest"
print(f"  Best          : {best}")
print(f"  SMOTE used    : {use_smote}")
print(f"\n  Models saved to: {MODELS_DIR}")

# Phone-specific feature importances
print(f"\n  Phone-Specific Feature Importances (XGBoost):")
phone_cols = [c for c in col_names if "phone" in c.lower()]
xgb_imp = xgb.feature_importances_
for c in phone_cols:
    idx = col_names.index(c)
    print(f"    {c:35s} : {xgb_imp[idx]:.4f}")
