"""
Two-stage InceptionV3 training — mirrors the original Kaggle pipeline.

Stage 1 — FC head training
    Freeze entire InceptionV3 base.  Build classification head:
    GlobalAveragePooling2D → Dense(1024, relu, l2) → Dropout(0.5) → Dense(N) [logits].
    Compile with CategoricalCrossentropy(from_logits=True), Adam(1e-4).
    Train with early stopping (patience 8) + checkpoint on val_accuracy.

Stage 2 — Late-layer fine-tuning
    Load best FC checkpoint.  Unfreeze last 60 base layers (skip BatchNorm).
    Compile with Adam(1e-5).  Train again with same callbacks.

Both stages output **raw logits** (no softmax) for energy-based OOD.
"""

from __future__ import annotations

import logging
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

from .config import INCEPTION_WEIGHTS_PATH, ORIGINAL_MODEL_PATH, TrainingConfig

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Model building
# ═══════════════════════════════════════════════════════════════════════════

def build_model(num_classes: int, config: TrainingConfig) -> tf.keras.Model:
    """Build an InceptionV3 classifier with frozen base + logits head.

    Architecture (matches the Kaggle notebook exactly)::

        Input(299,299,3)
          → InceptionV3(include_top=False, frozen)
          → GlobalAveragePooling2D
          → Dense(1024, relu, l2=1e-4)
          → Dropout(0.5)
          → Dense(num_classes)          ← raw logits, no softmax

    Parameters
    ----------
    num_classes : int
        Number of output classes.
    config : TrainingConfig
        Provides ``image_size`` and ``base_model`` to decide weight source.

    Returns
    -------
    tf.keras.Model
        Compiled model ready for stage-1 training.
    """
    h, w = config.image_size

    # ── Build or load base ──────────────────────────────────────────
    if config.base_model == "scratch" or config.base_model is None:
        # Fresh InceptionV3 backbone with ImageNet weights
        weights_arg = None
        base = InceptionV3(
            input_shape=(h, w, 3),
            include_top=False,
            weights=weights_arg,
        )
        # Load ImageNet weights from local file if available, else Keras default
        if INCEPTION_WEIGHTS_PATH.exists():
            base.load_weights(str(INCEPTION_WEIGHTS_PATH))
            logger.info("Loaded ImageNet weights from %s", INCEPTION_WEIGHTS_PATH)
        else:
            # Re-create with 'imagenet' so Keras auto-downloads
            base = InceptionV3(
                input_shape=(h, w, 3),
                include_top=False,
                weights="imagenet",
            )
            logger.info("Using Keras-downloaded ImageNet weights")
    else:
        # Fine-tune from an existing .keras model — extract the base
        existing = load_model(str(config.base_model), compile=False)
        # The InceptionV3 base is the first layer in our Model(inputs, outputs) wrapper
        base = existing.layers[1] if len(existing.layers) > 2 else existing
        logger.info("Loaded base from existing model: %s", config.base_model)

    # Freeze entire base for stage 1
    base.trainable = False

    # ── Classification head ─────────────────────────────────────────
    inputs = Input(shape=(h, w, 3))
    x = base(inputs)
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation="relu", kernel_regularizer=l2(1e-4))(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, name="logits")(x)  # raw logits — no softmax

    model = tf.keras.Model(inputs, outputs, name="InceptionV3")

    logger.info(
        "Built InceptionV3 model: %d classes, base frozen (%d layers)",
        num_classes, len(base.layers),
    )
    return model


# ═══════════════════════════════════════════════════════════════════════════
# Stage 1 — FC head training
# ═══════════════════════════════════════════════════════════════════════════

def train_stage1(
    model: tf.keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    config: TrainingConfig,
    output_dir: Path,
) -> tf.keras.Model:
    """Train the classification head with frozen base.

    Compiles with Adam(lr_fc), CategoricalCrossentropy(from_logits=True).
    Checkpoints on val_accuracy, early-stops on val_loss.

    Returns the best model loaded from the checkpoint.
    """
    ckpt_path = output_dir / "fc_best.keras"

    model.compile(
        optimizer=Adam(learning_rate=config.learning_rate_fc),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=["accuracy", Precision(name="precision"), Recall(name="recall")],
    )

    callbacks = [
        ModelCheckpoint(
            filepath=str(ckpt_path),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=config.early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    logger.info("═══ STAGE 1: TRAINING CLASSIFIER HEAD ═══")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # Load the best checkpoint
    best_model = load_model(str(ckpt_path), compile=False)
    logger.info("Stage 1 complete — loaded best FC checkpoint from %s", ckpt_path)
    return best_model


# ═══════════════════════════════════════════════════════════════════════════
# Stage 2 — Late-layer fine-tuning
# ═══════════════════════════════════════════════════════════════════════════

def train_stage2(
    model: tf.keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    config: TrainingConfig,
    output_dir: Path,
) -> tf.keras.Model:
    """Fine-tune the last N layers of the InceptionV3 base.

    Unfreezes the last ``config.unfreeze_layers`` layers, skipping
    BatchNormalization if ``config.skip_bn_unfreeze`` is True.
    Compiles with Adam(lr_ft).

    Returns the best model loaded from the checkpoint.
    """
    ckpt_path = output_dir / "ft_best.keras"

    # Find the InceptionV3 base layer inside the model
    base_layer = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and "inception" in layer.name.lower():
            base_layer = layer
            break

    if base_layer is None:
        logger.warning("Could not find InceptionV3 base layer — skipping stage 2")
        return model

    # First freeze everything in the base
    for layer in base_layer.layers:
        layer.trainable = False

    # Unfreeze the last N layers (skip BatchNorm)
    unfrozen = 0
    for layer in base_layer.layers[-config.unfreeze_layers:]:
        if config.skip_bn_unfreeze and "batch_normalization" in layer.name.lower():
            continue
        layer.trainable = True
        unfrozen += 1

    logger.info(
        "Unfroze %d / %d layers in base (last %d, skipping BN=%s)",
        unfrozen, len(base_layer.layers),
        config.unfreeze_layers, config.skip_bn_unfreeze,
    )

    model.compile(
        optimizer=Adam(learning_rate=config.learning_rate_ft),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=["accuracy", Precision(name="precision"), Recall(name="recall")],
    )

    callbacks = [
        ModelCheckpoint(
            filepath=str(ckpt_path),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=config.early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    logger.info("═══ STAGE 2: FINE-TUNING LAST %d LAYERS ═══", config.unfreeze_layers)
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # Load the best checkpoint
    best_model = load_model(str(ckpt_path), compile=False)
    logger.info("Stage 2 complete — loaded best FT checkpoint from %s", ckpt_path)
    return best_model
