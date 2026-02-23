import json
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from sklearn.calibration import CalibratedClassifierCV

# 1) Load data
df = pd.read_csv("transfers_train.csv")

target = "transfer_success"
drop_cols = [target, "target_team", "target_league"]
X = df.drop(columns=drop_cols)
y = df[target]

feature_names = list(X.columns)

# 2) Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 3) Model + tuning (small grid, fast)
base = RandomForestClassifier(random_state=42, n_jobs=-1)

param_grid = {
    "n_estimators": [300, 600],
    "max_depth": [8, 12, None],
    "min_samples_leaf": [2, 4],
    "max_features": ["sqrt", 0.6],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(
    estimator=base,
    param_grid=param_grid,
    scoring="roc_auc",
    cv=cv,
    n_jobs=-1,
    verbose=1
)

grid.fit(X_train, y_train)
best = grid.best_estimator_

# 4) Calibrate probabilities (more trustworthy output)
calibrated = CalibratedClassifierCV(best, method="isotonic", cv=3)
calibrated.fit(X_train, y_train)

# 5) Evaluate
proba = calibrated.predict_proba(X_test)[:, 1]
pred = (proba >= 0.5).astype(int)

auc = roc_auc_score(y_test, proba)
acc = accuracy_score(y_test, pred)

print("\nBEST PARAMS:", grid.best_params_)
print(f"TEST AUC: {auc:.3f}")
print(f"TEST ACC: {acc:.3f}")
print("\nREPORT:\n", classification_report(y_test, pred))

# 6) Feature importance (from best RF before calibration)
importances = pd.Series(best.feature_importances_, index=feature_names).sort_values(ascending=False)
print("\nTOP 12 FEATURES:\n", importances.head(12))

# 7) Save artifacts
joblib.dump(calibrated, "model.pkl")
pd.DataFrame({"feature": importances.index, "importance": importances.values}).to_csv("feature_importance.csv", index=False)

metrics = {
    "best_params": grid.best_params_,
    "test_auc": float(auc),
    "test_accuracy": float(acc),
    "n_rows": int(len(df)),
    "n_features": int(len(feature_names)),
    "features": feature_names
}
with open("metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

print("\nSaved: model.pkl, feature_importance.csv, metrics.json")

