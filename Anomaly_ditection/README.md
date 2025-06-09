# Fallacy-Aware Anomaly Detection using ECG5000 (NumPy Only)

This project implements a Fallacy-Aware Anomaly Detection model for ECG time-series classification using the ECG5000 dataset. The model is built entirely with NumPy and features a fallacy-based decision mechanism that adapts over time based on predictive confidence and error streaks.

## Highlights

- Dataset: [ECG5000 from UCR Time Series Archive](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/)
- Model: Lightweight anomaly detector using sigmoid activation, selective PReLU-inspired randomization, and a fallacy-aware logic.
- Fallacy Points (FP): Tracks overconfidence patterns and penalizes consecutive false positives, helping the model maintain caution during prediction.
- Pure NumPy: No machine learning frameworks are used; the model is fully self-contained and interpretable.

## Dataset Overview

The ECG5000 dataset consists of time-series data representing heartbeat signals. Originally a 5-class classification dataset, it is adapted here as a binary anomaly detection task:

- **Class 1** → Normal heartbeat (label `0`)
- **Classes 2 to 5** → Abnormal heartbeat (label `1`)

This binary format simplifies the problem into detecting whether a heartbeat is normal or anomalous.

## Files and Structure

| File/Folder            | Description                                      |
|------------------------|--------------------------------------------------|
| `anomaly.py`           | Main NumPy-based model implementation            |
| `ECG5000/`             | Folder containing the dataset files              |
| ├── `ECG5000_TRAIN.txt`| Training data from UCR repository                |
| └── `ECG5000_TEST.txt` | Test data from UCR repository                    |
| `README.md`            | Project overview and documentation               |

Make sure the dataset files are placed inside a folder named `ECG5000` in the same directory as `anomaly.py`.

## Model Logic and Novelty

The model uses a fallacy-aware mechanism designed to simulate human-like reasoning. It maintains an internal Fallacy Point (FP) score that:

- Increases after correct predictions, modeling rising confidence
- Penalizes streaks of false positives to enforce caution
- Influences the prediction process via controlled probabilistic behavior

This approach introduces a lightweight cognitive bias simulation, especially useful in high-stakes domains where overconfidence may lead to critical errors—such as medical diagnostics or monitoring systems.

## Training and Performance

- **Epochs**: 20
- **Final Accuracy**: ~80%

While this performance is lower than many modern deep learning models, the fallacy-aware approach offers interpretability and risk-aware adjustments that traditional models lack. It demonstrates that even simple models can be enhanced to act with a level of decision sensitivity and self-correction.

## How to Run

1. Download the ECG5000 dataset from the UCR archive.
2. Place the two files `ECG5000_TRAIN.txt` and `ECG5000_TEST.txt` inside a folder named `ECG5000`.
3. Ensure the folder is located in the same directory as `anomaly.py`.
4. Run the script:

```bash
python anomaly.py
