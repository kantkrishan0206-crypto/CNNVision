# CNNVision — Experiment Summary

This notebook project trains a CNN on CIFAR-10 (64x64 resized) to perform image classification.

## What the project does
- Prepares dataset folder structure, computes EDA (class distribution, pixel stats).
- Defines CNNModel and trains with AMP on GPU (if available).
- Saves best checkpoint and metrics.
- Evaluates the model (confusion matrix, classification report).
- Exports demo predictions, model export (TorchScript/ONNX), and a run manifest for reproducibility.

## Artifacts (see `notebooks/outputs`)
- classification_report: C:\Users\Dell\CNNVision-1\notebooks\notebooks\outputs\classification_report.csv (exists=True)
- metrics: C:\Users\Dell\CNNVision-1\notebooks\notebooks\outputs\metrics.json (exists=True)
- run_manifest: C:\Users\Dell\CNNVision-1\notebooks\notebooks\outputs\run_manifest_20251009T142411Z.json (exists=True)
- classification_csv: C:\Users\Dell\CNNVision-1\notebooks\notebooks\outputs\classification_report.csv (exists=True)
- predictions_csv: C:\Users\Dell\CNNVision-1\notebooks\notebooks\outputs\predictions_demo.csv (exists=True)
- class_counts: C:\Users\Dell\CNNVision-1\notebooks\notebooks\outputs\class_counts.json (exists=True)
- pixel_stats: C:\Users\Dell\CNNVision-1\notebooks\notebooks\outputs\pixel_stats.json (exists=True)
- requirements: C:\Users\Dell\CNNVision-1\notebooks\notebooks\outputs\requirements.txt (exists=True)

## Model exports
- TorchScript: C:\Users\Dell\CNNVision-1\notebooks\models\cnn_model_ts.pt (exists=True)
- ONNX: C:\Users\Dell\CNNVision-1\notebooks\models\cnn_model.onnx (exists=False)

## How to reproduce
1. Create venv and install dependencies (see requirements.txt).
2. Run cells 1..10 in order in this notebook with the kernel set to the project venv.

