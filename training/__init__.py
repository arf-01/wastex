"""
WasteX Retraining Pipeline
===========================

Background retraining system that:

1. Reads training data from the active dataset version (via ``VersionEntry``).
2. Two-stage fine-tunes an InceptionV3-based classifier (Kaggle pipeline).
3. Evaluates on the test split (confusion matrix, classification report).
4. Saves the new model + metrics under ``models/versions/<run_name>/``.
5. Optionally promotes the new model to "active" for live inference.

Package layout
--------------
config.py     – ``TrainingConfig`` dataclass, paths, hyperparameter defaults.
data.py       – Dataset loading: VersionEntry → tf.data.Dataset pipelines.
train.py      – Model building and two-stage training (FC head → fine-tune).
evaluate.py   – Test evaluation, confusion matrix, metrics persistence.
runner.py     – End-to-end orchestrator (data → train → evaluate → save).
tasks.py      – Background thread launcher and status helpers.
"""