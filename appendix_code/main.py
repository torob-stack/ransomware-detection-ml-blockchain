import pandas as pd
import numpy as np
import os

# Create a folder to save data
os.makedirs('data', exist_ok=True)

# Set seed for reproducibility
np.random.seed(42)

# Simulate normal behavior (100 samples)
normal = pd.DataFrame({
    'file_changes': np.random.poisson(3, 100),          # around 3 file changes
    'cpu_usage': np.random.normal(20, 5, 100),          # ~20% CPU
    'network_activity': np.random.normal(100, 15, 100), # ~100 packets/sec
    'label': 0                                           # 0 = normal
})

# Simulate ransomware behavior (20 samples)
ransomware = pd.DataFrame({
    'file_changes': np.random.poisson(25, 20),          # high file changes
    'cpu_usage': np.random.normal(90, 10, 20),          # high CPU usage
    'network_activity': np.random.normal(300, 20, 20),  # high network usage
    'label': 1                                           # 1 = ransomware
})

# Combine both into one dataset
data = pd.concat([normal, ransomware], ignore_index=True)

# Save the dataset
data.to_csv('data/simulated_ransomware.csv', index=False)

print("Dataset created and saved to data/simulated_ransomware.csv")

#----------------------------------------------------------------------------------------------

from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix

# Load the dataset
df = pd.read_csv('data/simulated_ransomware.csv')

# Separate features and labels
X = df.drop(columns=['label'])
y_true = df['label']

# Create and train Isolation Forest
model = IsolationForest(contamination=0.15, random_state=42)
model.fit(X)

# Make predictions (-1 = anomaly, 1 = normal)
y_pred = model.predict(X)

# Convert predictions to match label format (1 = ransomware, 0 = normal)
y_pred = [0 if p == 1 else 1 for p in y_pred]

# Evaluation
print("\n=== Confusion Matrix ===")
print(confusion_matrix(y_true, y_pred))

print("\n=== Classification Report ===")
print(classification_report(y_true, y_pred))
