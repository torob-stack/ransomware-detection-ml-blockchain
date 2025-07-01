from blockchain_logger import record_log
# === Step 1: Imports ===
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

# === Step 2: Load the Real-World Dataset ===
df_real = pd.read_csv("data/thursday_ransomware.csv", low_memory=False)
print("✅ Dataset loaded. Shape:", df_real.shape)
print("Unique labels:", df_real['Label'].unique())

# === Step 3: Filter for Benign and Infilteration (ransomware) ===
df_filtered = df_real[df_real['Label'].isin(['Benign', 'Infilteration'])].copy()
df_filtered['label'] = df_filtered['Label'].apply(lambda x: 1 if x == 'Infilteration' else 0)

print("Ransomware samples:", df_filtered['label'].sum())
print("Normal samples:", (df_filtered['label'] == 0).sum())

# === Step 4: Select and Rename Features ===
df_filtered = df_filtered[[
    'Flow Duration',       # proxy for CPU usage
    'Tot Fwd Pkts',        # proxy for file changes
    'Tot Bwd Pkts',        # proxy for network activity
    'label'
]]

df_filtered.rename(columns={
    'Flow Duration': 'cpu_usage',
    'Tot Fwd Pkts': 'file_changes',
    'Tot Bwd Pkts': 'network_activity'
}, inplace=True)

# Drop rows with invalid values
df_filtered = df_filtered.replace([np.inf, -np.inf], np.nan).dropna()

# === Step 5: Train and Evaluate Random Forest ===
X = df_filtered.drop(columns=['label'])
y = df_filtered['label']

model = RandomForestClassifier()
model.fit(X, y)
y_pred = model.predict(X)

# === Optional: Blockchain logging for detected ransomware cases ===
# === Blockchain Logging (Controlled) ===
#log_limit = 5
#logged = 0
#
#for i, (actual, predicted) in enumerate(zip(y, y_pred)):
#    if predicted == 1 and logged < log_limit:
#        msg = f"Ransomware detected at index {i}"
#        sys_id = f"ML-System-{i}"
#        record_log(msg, sys_id)
#        logged += 1
#if logged == 0:
#    print("ℹ️ No ransomware detections logged to blockchain.")
#else:
#    print(f"{logged} ransomware detection(s) logged to blockchain.")

# === Optional: Blockchain Logging for High-Confidence Ransomware Detections ===

# Get probability scores for class 1 (ransomware)
y_probs = model.predict_proba(X)[:, 1]

log_limit = 5   # Prevents spamming the blockchain
logged = 0
confidence_threshold = 0.85  # Minimum confidence to trigger blockchain log

for i, (pred, prob) in enumerate(zip(y_pred, y_probs)):
    if pred == 1 and prob >= confidence_threshold and logged < log_limit:
        msg = f"High-confidence ransomware detected (p={prob:.2f})"
        sys_id = f"ML-System-{i}"
        record_log(msg, sys_id)
        logged += 1

if logged == 0:
    print("ℹ️ No high-confidence ransomware detections logged to blockchain.")
else:
    print(f"{logged} high-confidence detection(s) logged to blockchain.")

print("\n=== Classification Report (Real-World Ransomware) ===")
print(classification_report(y, y_pred))

print("=== Confusion Matrix ===")
cm = confusion_matrix(y, y_pred)
print(cm)

# === Step 6: Confusion Matrix Heatmap ===
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', xticklabels=['Normal', 'Ransomware'], yticklabels=['Normal', 'Ransomware'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Real Data")
plt.tight_layout()
plt.savefig("results/real_confusion_matrix.png")
plt.show()

# === Step 7: ROC Curve and AUC ===
if len(np.unique(y)) > 1:
    y_probs = model.predict_proba(X)[:, 1]
    fpr, tpr, thresholds = roc_curve(y, y_probs)
    auc_score = roc_auc_score(y, y_probs)

    print(f"AUC Score: {auc_score:.2f}")

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}", color='darkorange')
    plt.plot([0, 1], [0, 1], linestyle='--', color='grey')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Real Ransomware Detection")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/real_roc_curve.png")
    plt.show()
else:
    print("❗ ROC Curve skipped: only one class detected in labels.")

# === Step 8: Comparison Chart - Simulated vs Real Data ===

# Replace these with your actual values from both tests
simulated_scores = {
    "Precision": 0.81,
    "Recall":    0.73,
    "F1-Score":  0.77,
    "AUC":       1.00
}

real_scores = {
    "Precision": 0.83,
    "Recall":    0.55,
    "F1-Score":  0.66,
    "AUC":       0.90
}

# Convert to lists
metrics = list(simulated_scores.keys())
sim_values = list(simulated_scores.values())
real_values = list(real_scores.values())

# Plotting
x = np.arange(len(metrics))
width = 0.35

plt.figure(figsize=(8, 6))
plt.bar(x - width/2, sim_values, width, label='Simulated Data', color='skyblue')
plt.bar(x + width/2, real_values, width, label='Real Data', color='orange')

plt.ylabel('Score')
plt.ylim(0, 1.05)
plt.title('Model Performance: Simulated vs Real-World Data')
plt.xticks(x, metrics)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("results/sim_vs_real_comparison.png")
plt.show()
