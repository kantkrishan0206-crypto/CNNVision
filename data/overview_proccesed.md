Data preprocessing for processed images
You’ll get a production-grade pipeline that converts data/raw images into data/processed with deterministic resizing, optional augmentation for train, integrity checks, manifests, and normalization stats. It uses the real CIFAR‑10 data you ingested earlier and runs end-to-end with clear CLI commands.

What this pipeline does
- Resizes images to your target size
- Applies train-only augmentations (optional)
- Preserves split/class directory structure
- Writes new manifests under data/processed/manifests
- Computes mean and std from the training set (saved to JSON)
- Verifies counts and class coverage
- Deterministic filenames with audit trail

