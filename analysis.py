"""
Portfolio Project: Diagnostic Data Analysis
Dataset: Breast Cancer Wisconsin (Diagnostic) Dataset (scikit-learn / UCI ML Repository)

Author: Caspal Okelo
Purpose: Demonstrate exploratory data analysis and a simple predictive
model on real diagnostic laboratory data, combining clinical/lab
domain knowledge with Python data analysis skills.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_curve, roc_auc_score
)

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 120

# ---------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------
data = datasets.load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["diagnosis"] = data.target
df["diagnosis_label"] = df["diagnosis"].map({0: "Malignant", 1: "Benign"})

print("Dataset shape:", df.shape)
print(df["diagnosis_label"].value_counts())
print(df.describe().T[["mean", "std", "min", "max"]].head(10))

# ---------------------------------------------------------------
# 2. Class balance chart
# ---------------------------------------------------------------
plt.figure(figsize=(5, 4))
ax = sns.countplot(data=df, x="diagnosis_label", hue="diagnosis_label",
                    palette={"Malignant": "#c0392b", "Benign": "#2980b9"}, legend=False)
ax.set_title("Diagnosis Distribution\n(569 breast tissue biopsy samples)")
ax.set_xlabel("")
ax.set_ylabel("Number of samples")
for p in ax.patches:
    ax.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width()/2, p.get_height()),
                ha="center", va="bottom", fontsize=11)
plt.tight_layout()
plt.savefig("01_class_distribution.png")
plt.close()

# ---------------------------------------------------------------
# 3. Key feature comparison: mean radius, mean texture, mean concavity
# ---------------------------------------------------------------
key_features = ["mean radius", "mean texture", "mean concavity", "mean area"]
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
for ax, feat in zip(axes.flatten(), key_features):
    sns.boxplot(data=df, x="diagnosis_label", y=feat, hue="diagnosis_label",
                palette={"Malignant": "#c0392b", "Benign": "#2980b9"}, legend=False, ax=ax)
    ax.set_title(feat.title())
    ax.set_xlabel("")
fig.suptitle("Distribution of Key Diagnostic Measurements by Class", y=1.02, fontsize=14)
plt.tight_layout()
plt.savefig("02_key_features_boxplots.png", bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------
# 4. Correlation heatmap (top features)
# ---------------------------------------------------------------
top_corr_features = df.corr(numeric_only=True)["diagnosis"].abs().sort_values(ascending=False).index[1:11]
plt.figure(figsize=(9, 7))
corr = df[top_corr_features].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, square=True, cbar_kws={"shrink": 0.8})
plt.title("Correlation Between Top 10 Diagnostically-Relevant Features")
plt.tight_layout()
plt.savefig("03_correlation_heatmap.png")
plt.close()

# ---------------------------------------------------------------
# 5. Model: Logistic Regression to predict diagnosis
# ---------------------------------------------------------------
X = df[data.feature_names]
y = df["diagnosis"]  # 0 = Malignant, 1 = Benign

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=5000, random_state=42)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

acc = accuracy_score(y_test, y_pred)
print(f"\nModel accuracy on held-out test set: {acc:.3f}")
print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=["Malignant", "Benign"]))

# ---------------------------------------------------------------
# 6. Confusion matrix chart
# ---------------------------------------------------------------
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4.5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Malignant", "Benign"], yticklabels=["Malignant", "Benign"])
plt.title(f"Confusion Matrix\nModel Accuracy: {acc:.1%}")
plt.xlabel("Predicted diagnosis")
plt.ylabel("Actual diagnosis")
plt.tight_layout()
plt.savefig("04_confusion_matrix.png")
plt.close()

# ---------------------------------------------------------------
# 7. ROC curve
# ---------------------------------------------------------------
fpr, tpr, _ = roc_curve(y_test, y_proba)
auc = roc_auc_score(y_test, y_proba)
plt.figure(figsize=(5.5, 5))
plt.plot(fpr, tpr, color="#2980b9", linewidth=2, label=f"ROC curve (AUC = {auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve: Predicting Benign vs. Malignant")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("05_roc_curve.png")
plt.close()

# ---------------------------------------------------------------
# 8. Feature importance (top coefficients)
# ---------------------------------------------------------------
coef_df = pd.DataFrame({
    "feature": data.feature_names,
    "coefficient": model.coef_[0]
}).sort_values("coefficient", key=abs, ascending=False).head(10)

plt.figure(figsize=(8, 6))
colors = ["#2980b9" if c > 0 else "#c0392b" for c in coef_df["coefficient"]]
plt.barh(coef_df["feature"], coef_df["coefficient"], color=colors)
plt.axvline(0, color="black", linewidth=0.8)
plt.title("Top 10 Features Influencing Diagnosis Prediction\n(Blue = pushes toward Benign, Red = pushes toward Malignant)")
plt.xlabel("Model coefficient (standardized)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("06_feature_importance.png")
plt.close()

print("\nAll charts saved. Analysis complete.")
