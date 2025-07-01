import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve,  precision_score, recall_score, f1_score


# === Step 1: Load Real-World Dataset ===
df_real = pd.read_csv("data/thursday_ransomware.csv", low_memory=False)
df_filtered = df_real[df_real['Label'].isin(['Benign', 'Infiltration'])].copy()
df_filtered['label'] = df_filtered['Label'].apply(lambda x: 1 if x == 'Infilteration' else 0)

df_filtered = df_filtered[[
    'Flow Duration',
    'Tot Fwd Pkts',
    'Tot Bwd Pkts',
    'label'
]].replace([np.inf, -np.inf], np.nan).dropna()

df_filtered.rename(columns={
    'Flow Duration': 'cpu_usage',
    'Tot Fwd Pkts': 'file_changes',
    'Tot Bwd Pkts': 'network_activity'
}, inplace=True)

X = df_filtered.drop(columns=['label'])
y = df_filtered['label']

# === Step 2: Train Random Forest ===
rf_model = RandomForestClassifier()
rf_model.fit(X, y)
y_rf_pred = rf_model.predict(X)
y_rf_probs = rf_model.predict_proba(X)[:, 1]
rf_cm = confusion_matrix(y, y_rf_pred)
rf_auc = roc_auc_score(y, y_rf_probs)

# === Step 3: Train Isolation Forest ===
iso_model = IsolationForest(contamination=0.2, random_state=42)
iso_model.fit(X[y == 0])  # Train only on benign
y_iso_scores = iso_model.decision_function(X)
y_iso_pred = iso_model.predict(X)
y_iso_pred = np.where(y_iso_pred == -1, 1, 0)
iso_cm = confusion_matrix(y, y_iso_pred)
iso_auc = roc_auc_score(y, y_iso_scores)

# === Step 4: Print Reports ===
print("\n=== Random Forest (Supervised) ===")
print(classification_report(y, y_rf_pred))
print("AUC Score:", rf_auc)

print("\n=== Isolation Forest (Unsupervised) ===")
print(classification_report(y, y_iso_pred))
print("AUC Score:", iso_auc)

# === Step 5: Visual Comparison ===
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Blues', ax=axs[0])
axs[0].set_title("Random Forest Confusion Matrix")
axs[0].set_xlabel("Predicted")
axs[0].set_ylabel("Actual")

sns.heatmap(iso_cm, annot=True, fmt='d', cmap='Greens', ax=axs[1])
axs[1].set_title("Isolation Forest Confusion Matrix")
axs[1].set_xlabel("Predicted")
axs[1].set_ylabel("Actual")

plt.tight_layout()
plt.savefig("results/real_comparison_conf_matrix.png")
plt.show()

# === Step 6: ROC Curves ===
fpr_rf, tpr_rf, _ = roc_curve(y, y_rf_probs)
fpr_iso, tpr_iso, _ = roc_curve(y, y_iso_scores)

plt.figure(figsize=(8, 6))
plt.plot(fpr_rf, tpr_rf, label=f"Random Forest AUC = {rf_auc:.2f}", color='orange')
plt.plot(fpr_iso, tpr_iso, label=f"Isolation Forest AUC = {iso_auc:.2f}", color='green')
plt.plot([0, 1], [0, 1], linestyle='--', color='grey')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves - Real Data")
plt.legend()
plt.tight_layout()
plt.savefig("results/real_model_comparison_roc.png")
plt.show()

# === STEP 10: Graphical Comparison Bar Chart ===
metrics_labels = ['Precision', 'Recall', 'F1-Score']
isolation_scores = [0.61, 0.42, 0.50]  # Replace with your actual Isolation Forest scores
rf_scores = [0.83, 0.55, 0.66]         # Replace with your actual Random Forest scores

x = np.arange(len(metrics_labels))
width = 0.35

plt.figure(figsize=(8, 6))
plt.bar(x - width/2, isolation_scores, width, label='Isolation Forest', color='skyblue')
plt.bar(x + width/2, rf_scores, width, label='Random Forest', color='lightgreen')
plt.xticks(x, metrics_labels)
plt.ylim(0, 1.05)
plt.ylabel("Score")
plt.title("Performance Comparison: Isolation Forest vs Random Forest (Real-World Data)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("results/model_comparison_bar_chart.png")
plt.show()
