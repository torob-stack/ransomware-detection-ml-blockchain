import pandas as pd
import numpy as np

# === Step 1: Load CSV and Clean Headers ===
file_path = "data/thursday_ransomware.csv"
df_real = pd.read_csv(file_path, low_memory=False)

# Strip whitespace from column names
df_real.columns = df_real.columns.str.strip()

# Show all column names
print("\n📋 Available columns:")
print(df_real.columns.tolist())

# === Step 2: Check for expected columns ===
required_columns = ['Label', 'Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts']
missing = [col for col in required_columns if col not in df_real.columns]

if missing:
    print(f"\n❌ Missing expected columns: {missing}")
    print("Check the dataset or column formatting.")
    exit()
else:
    print("\n✅ All required columns found.")

# === Step 3: Filter for ransomware vs benign ===
df_filtered = df_real[df_real['Label'].isin(['Benign', 'Infilteration'])].copy()
df_filtered['label'] = df_filtered['Label'].apply(lambda x: 1 if x == 'Infilteration' else 0)

# Select and rename relevant features
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

print(f"\n✅ Filtered dataset shape: {df_filtered.shape}")
print(df_filtered.head())

# Optional: Save cleaned data
df_filtered.to_csv("data/cleaned_ransomware_data.csv", index=False)
print("\n💾 Saved cleaned data to 'data/cleaned_ransomware_data.csv'")
