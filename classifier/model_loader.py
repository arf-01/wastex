"""
Model loader and inference utilities for waste classification.

Loads a pre-trained InceptionV3 Keras model (outputting raw logits) and
provides helper functions for preprocessing, energy-based OOD detection,
and final prediction.

Architecture : InceptionV3  (input 299×299 RGB, /255 normalisation)
OOD method   : Energy score  –  energy = −T · logsumexp(logits / T)
               Images with energy > threshold OR max softmax < 0.7
               are flagged as Out-of-Distribution.
"""

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import tensorflow as tf
from PIL import Image
from django.conf import settings

logger = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────────────────────────

MODEL_PATH: Path = Path(settings.BASE_DIR) / "models" / "logits_mdl.keras"
INPUT_SIZE: Tuple[int, int] = (299, 299)      # InceptionV3 expected input
ENERGY_THRESHOLD: float = -4.338604            # Calibrated on validation set
SOFTMAX_CONFIDENCE_MIN: float = 0.7            # Minimum softmax confidence

# ── Model loading (singleton – loaded once at import time) ──────────────────

try:
    model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
    logger.info("Loaded classification model from %s", MODEL_PATH)
except Exception:
    logger.exception("Failed to load classification model from %s", MODEL_PATH)
    raise


# ── Preprocessing ───────────────────────────────────────────────────────────

def preprocess_image(image_path: str) -> np.ndarray:
    """Read an image file and return a (1, 299, 299, 3) float32 array in [0, 1].

    Args:
        image_path: Absolute path to the image on disk.

    Returns:
        Batch-ready numpy array with shape (1, 299, 299, 3).
    """
    img = Image.open(image_path).convert("RGB")
    img = img.resize(INPUT_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


# ── Energy & softmax helpers ────────────────────────────────────────────────

def calculate_energy(logits: np.ndarray, T: float = 1.0) -> np.ndarray:
    """Compute the energy score for a batch of logit vectors.

    Energy = −T · logsumexp(logits / T)

    Lower (more negative) energy → higher confidence the sample is in-distribution.

    Args:
        logits: Raw model logits, shape (batch, num_classes).
        T:      Temperature scaling factor (default 1.0).

    Returns:
        1-D numpy array of energy scores, one per sample.
    """
    tensor = tf.convert_to_tensor(logits, dtype=tf.float32)
    energy = -T * tf.reduce_logsumexp(tensor / T, axis=1)
    return energy.numpy()


def calculate_softmax_probs(logits: np.ndarray) -> np.ndarray:
    """Convert raw logits to softmax probability distribution.

    Args:
        logits: Raw model logits, shape (batch, num_classes).

    Returns:
        Numpy array of probabilities with the same shape.
    """
    tensor = tf.convert_to_tensor(logits, dtype=tf.float32)
    return tf.nn.softmax(tensor, axis=1).numpy()


# ── Main inference entry-point ──────────────────────────────────────────────

def get_logits(image_path: str) -> Tuple[np.ndarray, float, bool]:
    """Run inference on a single image and determine OOD status.

    Pipeline:
        1. Preprocess image → (1, 299, 299, 3) tensor
        2. Forward pass → raw logits
        3. Compute energy score & softmax probabilities
        4. Flag as OOD if energy > threshold  OR  max(softmax) < 0.7

    Args:
        image_path: Absolute path to the image file.

    Returns:
        Tuple of:
            logits  – 1-D array of raw logit values (num_classes,)
            energy  – Scalar energy score
            ood     – True if the image is Out-of-Distribution
    """
    img = preprocess_image(image_path)
    logits = model.predict(img, verbose=0)

    energy = calculate_energy(logits)[0]
    probs = calculate_softmax_probs(logits)[0]

    ood = (energy > ENERGY_THRESHOLD) or (np.max(probs) < SOFTMAX_CONFIDENCE_MIN)

    return logits[0], float(energy), bool(ood)

