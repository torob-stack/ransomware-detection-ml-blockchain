# Resilient Ransomware Detection and Logging Using Machine Learning and Blockchain

This project implements a dual-layered defense framework that combines:
- **Machine Learning (ML)** for early ransomware detection
- **Blockchain technology** for tamper-proof forensic logging

Developed as a final year project at Keele University, this proof-of-concept (PoC) demonstrates how integrating intelligent threat detection with immutable evidence storage can enhance cyber resilience and incident response.#
pdf contains full project write up. 


## Features

- **Machine Learning Detection**:
  - Uses **Random Forest** (supervised) and **Isolation Forest** (unsupervised) models.
  - Trained and evaluated on both synthetic and real-world data (CSE-CIC-IDS2018).
  - Behavioural features: CPU usage, file changes, and network activity.

- **Blockchain Logging**:
  - Ethereum smart contract logs detections immutably.
  - Uses **Hardhat** local blockchain and **web3.py** for integration.
  - Ensures forensic logs are preserved even if the local system is compromised.


## Datasets

- **Synthetic**: Simulated benign and ransomware activities.
- **Real**: CSE-CIC-IDS2018 dataset, focusing on the "Infiltration" subset. struggled to set this dataset up as it was originally was lost due to broken hardive.



