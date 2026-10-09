import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, balanced_accuracy_score,
    classification_report, confusion_matrix,
    ConfusionMatrixDisplay
)

# 1. DATASET LOADING
df = pd.read_csv("kag_risk_factors_cervical_cancer.csv")
df = df.replace("?", pd.NA)

for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Remove two columns with extensive missing data, if present
df = df.drop(columns=[
    "STDs: Time since first diagnosis",
    "STDs: Time since last diagnosis"
], errors="ignore")

# 2. TARGET AND INPUT FEATURES
target = "Biopsy"
excluded = ["Biopsy", "Hinselmann", "Schiller", "Citology"]

X = df.drop(columns=excluded, errors="ignore")
y = df[target]

# Remove rows with missing target labels
valid = y.notna()
X = X.loc[valid]
y = y.loc[valid].astype(int)

# Remove entirely empty input columns
X = X.dropna(axis=1, how="all")

print("Dataset shape:", df.shape)
print("Input features:", X.shape[1])
print("Target distribution:\n", y.value_counts())

# 3. TRAIN-TEST SPLIT BEFORE IMPUTATION
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 4. BASELINE MODEL
baseline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        class_weight="balanced", max_iter=2000, random_state=42
    ))
])

baseline.fit(X_train, y_train)
baseline_pred = baseline.predict(X_test)

# 5. HYPERPARAMETER TUNING
tuning_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        max_iter=2000, random_state=42
    ))
])

grid = GridSearchCV(
    tuning_pipeline,
    {
        "classifier__C": [0.01, 0.1, 1, 10],
        "classifier__class_weight": ["balanced", None]
    },
    scoring="f1",
    cv=5,
    n_jobs=-1
)

grid.fit(X_train, y_train)
tuned = grid.best_estimator_
tuned_pred = tuned.predict(X_test)

print("\nBest parameters:", grid.best_params_)

# 6. FEATURE IMPORTANCE
coefs = abs(tuned.named_steps["classifier"].coef_[0])
importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": coefs
}).sort_values("Importance", ascending=False)

print("\nTop 10 features:")
print(importance.head(10).to_string(index=False))

# 7. FEATURE REDUCTION: SELECT TOP 10 FEATURES
top_features = importance.head(min(10, len(importance)))["Feature"].tolist()

reduced = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        class_weight="balanced", max_iter=2000, random_state=42
    ))
])

reduced.fit(X_train[top_features], y_train)
reduced_pred = reduced.predict(X_test[top_features])

print("\nSelected features:", top_features)

# 8. MODEL COMPARISON
models = {
    "Baseline": baseline_pred,
    "Tuned": tuned_pred,
    "Top-10 Features": reduced_pred
}

results = []

for name, pred in models.items():
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Balanced Accuracy": balanced_accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1-score": f1_score(y_test, pred, zero_division=0)
    })

results_df = pd.DataFrame(results)
print("\nMODEL COMPARISON")
print(results_df.round(3).to_string(index=False))

print("\nTUNED MODEL CLASSIFICATION REPORT")
print(classification_report(
    y_test, tuned_pred, zero_division=0
))

print("\nTUNED MODEL CONFUSION MATRIX")
print(confusion_matrix(y_test, tuned_pred))

# 9. VISUALIZATIONS
importance.head(10).sort_values("Importance").plot(
    x="Feature", y="Importance", kind="barh",
    legend=False, title="Top 10 Feature Importance"
)
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.close()

ConfusionMatrixDisplay.from_predictions(
    y_test, tuned_pred, values_format="d"
)
plt.title("Tuned Model Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()

results_df.set_index("Model")[
    ["Precision", "Recall", "F1-score"]
].plot(kind="bar", ylim=(0, 1), title="Model Metric Comparison")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("model_comparison.png")
plt.close()

results_df.to_csv("model_comparison.csv", index=False)

print("\nSaved: feature_importance.png")
print("Saved: confusion_matrix.png")
print("Saved: model_comparison.png")
print("Saved: model_comparison.csv")
print("\nPROJECT RUN COMPLETED")