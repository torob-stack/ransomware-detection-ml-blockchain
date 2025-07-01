# === STEP 1: Imports ===
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score

# === STEP 2: Prepare Data Folder ===
os.makedirs('data', exist_ok=True)
os.makedirs('results', exist_ok=True)

# === STEP 3: Generate Simulated Dataset ===
np.random.seed(42)

normal = pd.DataFrame({
    'file_changes': np.random.poisson(3, 100),
    'cpu_usage': np.random.normal(20, 5, 100),
    'network_activity': np.random.normal(100, 15, 100),
    'label': 0
})

ransomware = pd.DataFrame({
    'file_changes': np.random.poisson(25, 20),
    'cpu_usage': np.random.normal(90, 10, 20),
    'network_activity': np.random.normal(300, 20, 20),
    'label': 1
})

df = pd.concat([normal, ransomware], ignore_index=True)
df.to_csv('data/simulated_ransomware.csv', index=False)
print("Dataset created and saved to data/simulated_ransomware.csv")

# === STEP 4: Train Model Multiple Times ===
X = df.drop(columns=['label'])
y_true = df['label']

runs = 5
results = []

for i in range(runs):
    model = IsolationForest(contamination=0.15, random_state=i)
    model.fit(X)
    y_pred = model.predict(X)
    y_pred = [0 if p == 1 else 1 for p in y_pred]

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    results.append((precision, recall, f1))

# === STEP 5: Print Average Scores ===
avg_precision = np.mean([r[0] for r in results])
avg_recall = np.mean([r[1] for r in results])
avg_f1 = np.mean([r[2] for r in results])

print("\n=== Average Metrics Over 5 Runs ===")
print(f"Average Precision: {avg_precision:.2f}")
print(f"Average Recall:    {avg_recall:.2f}")
print(f"Average F1 Score:  {avg_f1:.2f}")

import seaborn as sns

# Create confusion matrix from last model run (run 5)
cm = confusion_matrix(y_true, y_pred)
labels = ['Normal', 'Ransomware']

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Isolation Forest (Last Run)")
plt.tight_layout()
plt.savefig("results/confusion_matrix_heatmap.png")
plt.show()


# === STEP 6: Visualize Data Points (Scatter Plot) ===
# Use last prediction (from run 5)
colors = ['blue' if p == 0 else 'red' for p in y_pred]

plt.figure(figsize=(8, 6))
plt.scatter(X['file_changes'], X['cpu_usage'], c=colors, alpha=0.7)
plt.xlabel("File Changes")
plt.ylabel("CPU Usage (%)")
plt.title("Anomaly Detection: Normal (blue) vs Ransomware (red)")
plt.grid(True)
plt.tight_layout()
plt.savefig("results/scatter_plot.png")
plt.show()

# === STEP 7: Visualize Model Metrics Over Runs (Line Graph) ===
metrics = list(zip(*results))
labels = ['Precision', 'Recall', 'F1-Score']
colors = ['green', 'orange', 'blue']

plt.figure(figsize=(8, 6))
for i, metric in enumerate(metrics):
    plt.plot(range(1, runs + 1), metric, marker='o', label=labels[i], color=colors[i])

plt.title("Model Performance Across Multiple Runs")
plt.xlabel("Run Number")
plt.ylabel("Score")
plt.ylim(0, 1.05)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/performance_across_runs.png")
plt.show()

# === STEP 8: Train and Evaluate Random Forest Classifier (Supervised) ===
from sklearn.ensemble import RandomForestClassifier

print("\n=== RANDOM FOREST CLASSIFIER (Supervised) ===")

# Train Random Forest using known labels
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X, y_true)
rf_pred = rf_model.predict(X)

# Evaluate Random Forest
rf_precision = precision_score(y_true, rf_pred)
rf_recall = recall_score(y_true, rf_pred)
rf_f1 = f1_score(y_true, rf_pred)

print("\nRandom Forest Classification Report:")
print(classification_report(y_true, rf_pred))

# Confusion Matrix for Random Forest
rf_cm = confusion_matrix(y_true, rf_pred)

# Plot Confusion Matrix Heatmap - Random Forest
plt.figure(figsize=(6, 5))
sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Greens', xticklabels=['Normal', 'Ransomware'], yticklabels=['Normal', 'Ransomware'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Random Forest Classifier")
plt.tight_layout()
plt.savefig("results/confusion_matrix_rf_heatmap.png")
plt.show()

# === STEP 9: Print Side-by-Side Comparison Table ===
print("\n=== Model Comparison Table ===")
print("{:<25} {:<15} {:<15} {:<15}".format("Metric", "Isolation Forest", "Random Forest", "Notes"))
print("-" * 70)
print("{:<25} {:<15.2f} {:<15.2f} {}".format("Precision", avg_precision, rf_precision, "Higher = fewer false alarms"))
print("{:<25} {:<15.2f} {:<15.2f} {}".format("Recall", avg_recall, rf_recall, "Higher = catches more attacks"))
print("{:<25} {:<15.2f} {:<15.2f} {}".format("F1-Score", avg_f1, rf_f1, "Balance of precision and recall"))

# === STEP 10: Graphical Comparison Chart ===
metrics_labels = ['Precision', 'Recall', 'F1-Score']
isolation_scores = [avg_precision, avg_recall, avg_f1]
rf_scores = [rf_precision, rf_recall, rf_f1]

x = np.arange(len(metrics_labels))
width = 0.35

plt.figure(figsize=(8, 6))
plt.bar(x - width/2, isolation_scores, width, label='Isolation Forest', color='skyblue')
plt.bar(x + width/2, rf_scores, width, label='Random Forest', color='lightgreen')
plt.xticks(x, metrics_labels)
plt.ylim(0, 1.05)
plt.ylabel("Score")
plt.title("Performance Comparison: Isolation Forest vs Random Forest")
plt.legend()
plt.tight_layout()
plt.savefig("results/model_comparison_bar_chart.png")
plt.show()

# === STEP 11: ROC Curve and AUC for Random Forest ===
# Get prediction probabilities (not just 0 or 1)
from sklearn.metrics import roc_curve, roc_auc_score

rf_probs = rf_model.predict_proba(X)[:, 1]  # Probability of class 1 (ransomware)

# Compute ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_true, rf_probs)
auc_score = roc_auc_score(y_true, rf_probs)

# Print AUC
print(f"\nRandom Forest AUC (Area Under Curve): {auc_score:.2f}")

# Plot ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"Random Forest (AUC = {auc_score:.2f})", color='darkorange')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')  # baseline
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title("ROC Curve - Random Forest")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.savefig("results/roc_curve_rf.png")
plt.show()
