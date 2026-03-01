# Models Directory

## Structure

```
models/
├── logits_mdl.keras       # Original shipped InceptionV3 model (logits output)
├── classes.txt            # Current class names (one per line, matches model output order)
└── versions/              # Training run artefacts (auto-created by the pipeline)
    └── model_v2_YYYYMMDD_HHMMSS/
        ├── model.keras           # Final trained model
        ├── best_model.keras      # Best checkpoint (lowest val_loss)
        ├── classes.txt           # Class list for this specific model
        ├── metrics.json          # Test accuracy, F1, per-class stats
        ├── comparison.json       # Delta vs previous model + recommendation
        ├── config.json           # Training config snapshot
        ├── training_log.json     # Epoch-level loss & accuracy
        ├── training_log.csv      # Same data, CSV format
        └── model_summary.txt     # Keras model.summary() output
```

## classes.txt Format

One class name per line, matching the output order of your model:

```
Cardboard
Food Organics
Glass
Metal
Paper
Plastic
Textile Trash
Vegetation
```

## Notes

- The `logits_mdl.keras` model outputs **raw logits** (not softmax probabilities).
- Energy-based OOD detection is applied on top: `energy = -logsumexp(logits)`.
- The retraining pipeline automatically saves new models under `versions/`.
