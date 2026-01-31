# Model Directory

Place your trained `.keras` model file here.

## Example

```
models/
├── model.keras       # Your trained Keras model
└── classes.txt       # Class names (one per line)
```

## classes.txt Format

One class name per line, matching the output order of your model:

```
class_0_name
class_1_name
class_2_name
```

The application will automatically load the first `.keras` file it finds.
