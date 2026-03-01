---
marp: true
math: katex
theme: default
paginate: true
size: 16:9
style: |
  section {
    font-size: 22px;
    padding: 40px 60px;
  }
  h1 {
    color: #1a5276;
    font-size: 36px;
    border-bottom: 3px solid #2ecc71;
    padding-bottom: 10px;
  }
  h2 {
    color: #2c3e50;
    font-size: 30px;
  }
  pre {
    font-size: 13.5px;
    line-height: 1.35;
    background: #1e1e2e;
    color: #cdd6f4;
    border-radius: 8px;
    padding: 14px 18px;
  }
  code {
    font-family: 'Cascadia Code', 'Fira Code', monospace;
  }
  table {
    font-size: 18px;
    margin: 0 auto;
  }
  th {
    background: #2c3e50;
    color: white;
    padding: 6px 14px;
  }
  td {
    padding: 5px 14px;
  }
  .columns {
    display: flex;
    gap: 30px;
  }
  .col {
    flex: 1;
  }
  blockquote {
    border-left: 4px solid #2ecc71;
    padding: 8px 16px;
    background: #eafaf1;
    font-size: 19px;
  }
  footer {
    font-size: 14px;
    color: #7f8c8d;
  }
---

<!-- _class: lead -->

# ♻️ WasteX — Retraining & Versioning Pipeline

### Automated Waste Classification with Continuous Improvement

**InceptionV3 · Energy-Based OOD · Delta Versioning · Background Training**

---

# 📋 Table of Contents

1. **System Overview** — High-level architecture
2. **Classification Flow** — Upload to inference to OOD routing
3. **Energy-Based OOD Detection** — How we catch unknowns
4. **Operator Review Workflow** — Label and stage OOD images
5. **Delta-Based Dataset Versioning** — Lightweight version control
6. **Background Training** — Thread-based retraining
7. **Evaluation & Comparison** — Metrics, reports, promotion
8. **Model Promotion** — Hot-swap the serving model
9. **Continuous Improvement Loop** — The full cycle



---

# 1. System Overview

```
 +--------+     +-----------+     +-------+     +-------------+
 | Upload |---->|InceptionV3|---->| OOD?  |---->| In-Dist:    |
 | Image  |     | Inference |     |Router |  NO | Count+1     |
 +--------+     +-----------+     +---+---+     | Delete file |
                                      |         +-------------+
                                      | YES
                                      v
                +----------+     +----------+     +-----------+
                |  Save    |---->| Operator |---->| Stage for |
                |  OOD Img |     |  Labels  |     | Version   |
                +----------+     +----------+     +-----+-----+
                                                        |
                                    +----------+        |
                                    | Promote  |<-------+
                                    | Model    |  Retrain (background)
                                    +----------+
```

---

# 2. Classification Flow

```
 +-------------+    +----------+    +----------+    +-----------+
 |POST         |--->|Validate  |--->|Save to   |--->|Preprocess |
 |/classify/   |    |type+size |    |temp path |    |299x299    |
 |(image file) |    |check     |    |Django    |    |norm [0,1] |
 +-------------+    +----------+    +----------+    +-----+-----+
                                                          |
                                                          v
 +-------------+    +----------+    +-----------+   +-----------+
 |Return JSON: |<---|  Branch  |<---|Energy +   |<--|model      |
 |logits,energy|    |  on OOD  |    |Softmax    |   |.predict() |
 |ood, class   |    |  result  |    |OOD check  |   |raw logits |
 +-------------+    +-----+----+    +-----------+   +-----------+
                     |         |
                     v         v
              +---------+ +---------+
              |YES: Save| |NO: Count|
              |OOD Image| |TrashCnt |
              |to DB    | |Del file |
              +---------+ +---------+
```

---

# 3. Energy-Based OOD Detection

### How We Catch Unknown Samples

<div class="columns">
<div class="col">

**Energy Score Formula:**
$$E(\mathbf{x}) = -T \cdot \ln \sum_{i=1}^{C} e^{f_i(\mathbf{x})/T}$$

Where $f_i(\mathbf{x})$ are raw logits, $T = 1.0$

**Decision Rule — flag as OOD if:**

| Condition | Threshold |
|-----------|-----------|
| Energy > threshold | **-4.338604** |
| max(softmax) < min | **0.70** |

Either condition triggers OOD.

</div>
<div class="col">

```
                Raw Logits [f1..fC]
                    |
            +-------+-------+
            |               |
            v               v
       +---------+     +---------+
       | Energy  |     | Softmax |
       |-logsumexp     | max(p)  |
       +----+----+     +----+----+
            |               |
            v               v
       +---------+     +---------+
       |> -4.34? |     |< 0.70?  |
       +----+----+     +----+----+
            |               |
            +-------+-------+
                    |
                    v
              +-----+-----+
              |  OR gate   |
              +-----+-----+
              |           |
             YES          NO
              v            v
            [OOD]       [In-Dist]
```

</div>
</div>

---

# 4. Operator Review Workflow

```
 +---------------+   +--------------+   +--------------+   +----------+
 | OOD Image     |-->| GET          |-->| POST         |-->| POST     |
 | saved to DB   |   | /api/ood/    |   | /api/ood/    |   | /api/ood/|
 | reviewed=False|   | Operator     |   | <id>/review/ |   | <id>/    |
 |               |   | sees list    |   | Mark seen    |   | label/   |
 +---------------+   +--------------+   +--------------+   +-----+----+
                                                                  |
                      +-------------------------------------------+
                      |
                      v
                +--------------+         +-----------------+
                | Image is now |-------->| GET /api/dataset|
                | "staged"     |         | /staged/        |
                | label is set |         | View all staged |
                | added=False  |         | ready for v2    |
                +--------------+         +-----------------+
```

---

# 5. Delta-Based Dataset Versioning

### No File Duplication — Lightweight DB Rows Only

<div class="columns">
<div class="col">

**Traditional Versioning:**
- Copy all files into new folder
- 10k images x 3 versions = 30k files
- Disk usage: **3x**

**Delta-Based (WasteX):**
- Copy DB rows only (INSERT...SELECT)
- Add new rows for staged OOD images
- Physical files stay in place
- Disk usage: **1x + delta**

</div>
<div class="col">

```
 +-------------+  fork   +-------------+  fork   +-------------+
 | v1 (seed)   |-------->| v2          |-------->| v3          |
 | 10,217 imgs |  copy   | 10,267 imgs |  copy   | 10,297 imgs |
 | datasets/v1 |  rows   | +50 OOD     |  rows   | +30 OOD     |
 +-------------+  only   +-------------+  only   +-------------+

   Physical files:          New files:              New files:
   datasets/v1/...          media/uploads/...       media/uploads/...
   (original data)          (50 OOD images)         (30 OOD images)

   DB rows: 10,217          DB rows: 10,267         DB rows: 10,297
                            (inherited + new)       (inherited + new)

   ZERO files duplicated across versions!
```

</div>
</div>

> **Key insight:** `VersionEntry` rows point to physical paths. Creating a version copies rows, not bytes.

---

# 5b. Version Creation Flow

```
 +-----------------+    +----------------+    +------------------+
 | POST /api/      |--->| Create         |--->| Bulk copy        |
 | dataset/        |    | DatasetVersion |    | VersionEntry     |
 | create-version/ |    | row in DB      |    | rows from parent |
 | {name, parent}  |    | parent --> v1   |    | INSERT...SELECT  |
 +-----------------+    +----------------+    | (no file IO)     |
                                              +--------+---------+
                                                       |
                                                       v
 +-----------------+    +----------------+    +------------------+
 | Optionally      |<---| Refresh cached |<---| For each staged  |
 | activate:       |    | stats:         |    | OOD image:       |
 | POST /api/      |    | total_images   |    | Create new       |
 | dataset/        |    | class_counts   |    | VersionEntry     |
 | set-active/     |    | splits list    |    | Mark added=True  |
 +-----------------+    +----------------+    +------------------+
```

---

# 5c. Database Schema (Versioning)

<div class="columns">
<div class="col">

```
  +------------------+
  | DatasetVersion   |
  +------------------+
  | id               |
  | name       "v1"  |
  | parent_id  (FK)  |
  | is_active  bool  |
  | total_images     |
  | class_counts {}  |
  | splits       []  |
  | notes            |
  | created_at       |
  +------------------+
       |  1
       |
       | has many
       v  *
  +------------------+
  | VersionEntry     |
  +------------------+
  | version_id (FK)  |
  | physical_path    |
  | split            |
  | class_label      |
  | filename         |
  | file_size        |
  | source_image(FK) |
  | added_at         |
  +------------------+
```

</div>
<div class="col">

```
  +------------------+
  | Image            |
  +------------------+
  | image     (file) |
  | filename         |
  | top_prediction   |
  | confidence       |
  | all_predictions  |
  | reviewed    bool |
  | assigned_label   |
  | added_to_dset    |
  | dataset_version  |
  +------------------+

  +------------------+
  | DatasetClass     |
  +------------------+
  | id               |
  | name   "Plastic" |
  | introduced_in FK |
  | created_at       |
  +------------------+

  +------------------+
  | TrashCounter     |
  +------------------+
  | class_name       |
  | total_count      |
  | recorded_at      |
  +------------------+
```

</div>
</div>

---

# 6. Background Training

### Thread-Based Retraining with Lock Protection

```
 +-------------------+     +------------------+
 | POST /api/        |---->| Acquire          |
 | training/start/   |     | threading lock   |
 | {dataset_version, |     | (1 run at a time)|
 |  epochs, ...}     |     +--------+---------+
 +-------------------+          |           |
                              locked     acquired
                                |           |
                                v           v
                         +----------+ +------------------+
                         |Return 409| |Spawn daemon      |
                         |conflict  | |thread            |
                         +----------+ +--------+---------+
                                               |
          +------------------------------------+
          |
          v
 +------------------+------------------+------------------+------------------+------------------+
 |1. Load dataset   |2. Build         |3. Stage 1        |4. Stage 2        |5. Evaluate       |
 |  train/val/test  |  InceptionV3    |  FC head only    |  Fine-tune 60    |  on test set     |
 |  and labels      |  from ImageNet  |  lr=1e-4         |  layers, lr=1e-5 |  compare w/ prev |
 +------------------+------------------+------------------+------------------+---------+--------+
           |
           v
 +---------------------+     +------------------------+
 | 6. Save artefacts   |---->| 7. Release lock        |
 | model.keras         |     | Update TrainingRun     |
 | metrics.json        |     | status = "completed"   |
 | confusion_matrix    |     | record in DB           |
 +---------------------+     +------------------------+
```

---



# 7. Evaluation & Comparison

```
 +-----------------+    +------------------+    +------------------+
 | evaluate_model()|    | model.evaluate   |    | Collect          |
 | training/       |--->| (test_ds)        |--->| y_true vs y_pred |
 | evaluate.py     |    | loss, accuracy,  |    | from all test    |
 |                 |    | precision,recall |    | batches          |
 +-----------------+    +------------------+    +--------+---------+
                                                         |
                   +------------------+------------------+
                   |                  |                  |
                   v                  v                  v
            +-----------+     +------------+     +------------+
            | confusion |     | classific. |     | metrics    |
            | _matrix   |     | _report    |     | .json      |
            | .png      |     | .txt       |     | (all nums) |
            +-----------+     +------------+     +------------+
                                    |
                                    v
                        +---------------------+
                        |compare_with_previous|
                        | Load prev metrics   |
                        | Compute deltas      |
                        | Recommend:          |
                        | promote/keep/review |
                        | --> comparison.json  |
                        +---------------------+
```

---

# 7b. Latest Training Results

<div class="columns">
<div class="col">

**Run:** `model_v1_20260224_162046`
**Dataset:** v1 (10,217 images)
**Duration:** ~2h 48m (CPU)

| Metric | Score |
|--------|------:|
| **Accuracy** | **90.38%** |
| **Macro F1** | **91.26%** |
| Precision | 91.55% |
| Recall | 90.05% |
| Loss | 0.4372 |

</div>
<div class="col">

**Per-Class F1 Scores:**

| Class | F1 |
|-------|---:|
| Vegetation | 97.67% |
| Textile Trash | 95.45% |
| Food Organics | 92.73% |
| Glass | 92.45% |
| Paper | 89.72% |
| Metal | 88.07% |
| Plastic | 87.27% |
| Cardboard | 86.73% |

</div>
</div>

---

# 8. Model Promotion

```
 +--------------+     +----------------+     +-----------------+
 | Training     |---->| Operator       |---->| POST /api/      |
 | completed    |     | reviews        |     | training/       |
 | status=done  |     | GET /api/      |     | promote/        |
 | active=False |     | training/      |     | {run_name:...}  |
 |              |     | status/        |     |                 |
 +--------------+     +----------------+     +--------+--------+
                                                      |
                                                      v
 +-------------------+     +------------------+     +------------------+
 | Model now serving |<----| Hot-reload model |<----| Deactivate all   |
 | live inference    |     | into memory      |     | other runs       |
 | No restart needed |     | model_loader     |     | Activate this    |
 +-------------------+     | .load_model()    |     | is_active = True |
                            +------------------+     +------------------+
```

> **Auto-promote:** Set `auto_promote: true` in config — if better, it promotes automatically.

---

# 9. Continuous Improvement Loop

```
       +--------+    +--------+    +--------+    +----------+
       | Deploy |    | Serve  |    |Detect  |    | Save OOD |
  +--->| Model  |--->|Requests|--->|  OOD   |--->| Image    |
  |    +--------+    +--------+    +--------+    +-----+----+
  |                                                     |
  |                                                     v
  |    +--------+    +--------+    +--------+    +----------+
  |    |Promote |    |Evaluate|    |Retrain |    | Operator |
  +----| if     |<---|  and   |<---| (bkgnd |<---| Labels   |
       |better  |    |Compare |    | thread)|    | + Stage  |
       +--------+    +--------+    +--------+    +----------+
                                       ^
                                       |
                                 +-----+------+
                                 |Create new  |
                                 |version     |
                                 |(delta-based|
                                 | v1 --> v2) |
                                 +------------+
```

> Every OOD image feeds back into training. The loop runs continuously.

---





<!-- _class: lead -->

# 🔄 Summary: The WasteX Loop

```
 +--------+   +--------+   +------+   +------+   +-------+   +-------+
 |Upload  |-->|Classify|-->| OOD? |-->| Save |-->| Label |-->|Create |
 |Image   |   |Infer.  |   |Check |   | Image|   |  it   |   |Version|
 +--------+   +---+----+   +--+---+   +------+   +-------+   +---+---+
                  |            |                                   |
                  | NO         | YES (above path)                  |
                  v            |                                   v
            +---------+        |            +--------+   +----------+
            |Count +1 |        |            |Promote |<--|Retrain + |
            |Del file |        |            |if      |   |Evaluate  |
            +---------+        |            |better  |   |(backgrnd)|
                               |            +---+----+   +----------+
                               |                |
                               |      +---------+
                               |      | Hot-reload
                               +----->| model
                                      +---------+
```

**Every OOD image makes the system smarter.**
**No manual file management. No downtime.**

---

<!-- _class: lead -->

#  Thank You

**WasteX Retraining Pipeline**



