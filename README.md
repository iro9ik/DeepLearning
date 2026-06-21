# Intelligent CRM & BI Engine — Deep Learning Notebook

**Supervisor:** Zineb H. 
**Framework:** PyTorch 2.x — Google Colab (CPU/GPU compatible)

---

## Overview

This notebook implements an **Intelligent CRM & Business Intelligence Engine** built on three distinct deep learning modules, each targeting a specific business function within a Customer Relationship Management pipeline. The three parts share a unified environment (global seed, device management, data split conventions) and follow academic, production-quality coding standards throughout.

---

## Global Environment

All parts share the following setup:

- **Device:** auto-selects CUDA GPU if available, falls back to CPU
- **Reproducibility:** fixed seed (`42`) across `random`, `numpy`, and `torch`
- **Data splits:** Train / Validation / Test across all experiments
- **Batch normalisation, dropout, and early stopping** are applied consistently

---

## Part 1 — Predictive Lead Scoring (MLP on Tabular Data)

### Business Context
The model predicts whether a prospect's income exceeds $50K/year using demographic and financial attributes. High-probability predictions are surfaced as **high-conversion leads** at the top of the CRM pipeline.

### Dataset
**Adult Income dataset** (UCI/Kaggle) — 32,561 training rows, 16,281 test rows across 15 features including age, education, occupation, hours per week, and capital gain/loss.

### Preprocessing Pipeline
1. Drop `fnlwgt` (census weight, irrelevant to scoring)
2. Strip whitespace and impute `'?'` missing values with the column mode
3. Binary encode the target (`<=50K` → 0, `>50K` → 1)
4. One-hot encode all categorical features
5. Stratified 70 / 15 / 15 train-val-test split
6. `StandardScaler` fit **only on train**, applied to numerical columns

### Model Architecture — `LeadScoringMLP`
A funnel-shaped Multi-Layer Perceptron (24,769 trainable parameters):
Input (108 features)

→ Linear(108 → 128) + BatchNorm + ReLU + Dropout(0.35)

→ Linear(128 → 64)  + BatchNorm + ReLU + Dropout(0.35)

→ Linear(64 → 32)   + BatchNorm + ReLU + Dropout(0.35)

→ Linear(32 → 1)    [logit → BCEWithLogitsLoss]

Two implementations are provided side-by-side:
- **Implementation A:** `nn.Sequential` — compact, readable
- **Implementation B:** Custom `nn.Module` subclass — extensible, recommended for production

### Training
- **Optimizer:** Adam with weight decay
- **Loss:** `BCEWithLogitsLoss` (handles class imbalance implicitly)
- **Weight initialisation compared:** Xavier Uniform vs. Kaiming He — Xavier yields the best F1
- **Early stopping** on validation F1 with model checkpointing (`state_dict` serialisation)

### Results
Best model (Xavier init) achieves an **F1-score of ~0.674** on the held-out test set. Checkpoint is saved to `./checkpoints/lead_scoring_xavier.pt` and round-trip verified.

---

## Part 2 — Invoice & Form Digitisation (CNN)

### Business Context
Handwritten digit recognition for OCR-based invoice and form field extraction. Each digit class (0–9) maps directly to a field value parsed from a scanned document in the CRM document pipeline.

### Dataset
**MNIST** — 60,000 training images, 5,000 validation, 5,000 test (28×28 grayscale, normalised to mean=0.1307, std=0.3081).

### Theoretical Background Covered
- 2D cross-correlation, padding, and stride
- Max pooling vs. Average pooling vs. Adaptive Average pooling
- Explainable AI via first-layer feature map visualisation

### Model Architecture — `LeNetCRM`
A LeNet-inspired CNN adapted with BatchNorm for stability:
Input (1 × 28 × 28)

→ Conv2d(1→6, k=5, pad=2) + BatchNorm2d + ReLU + AvgPool(2)

→ Conv2d(6→16, k=5)       + BatchNorm2d + ReLU + AvgPool(2)

→ Flatten

→ Linear(256 → 120) + ReLU

→ Linear(120 → 84)  + ReLU

→ Linear(84 → 10)   [logits over 10 digit classes]

### Training
- **Loss:** CrossEntropyLoss
- **Optimizer:** Adam
- Training and validation curves plotted per epoch
- Feature map visualisations extracted from the first convolutional layer for XAI inspection

### Results
The model achieves strong multi-class classification performance evaluated with accuracy, weighted precision, recall, and F1-score, with a full 10×10 confusion matrix visualised.

---

## Part 3 — Customer Sentiment Analysis (Bidirectional LSTM)

### Business Context
Classifies customer support ticket reviews as **positive** or **negative**. Sentiment predictions are used to prioritise CRM ticket queues, flag at-risk accounts, and generate automated response recommendations.

### Dataset
Customer support ticket dataset (Kaggle) — ~2,189 reviews with sentiment labels, ticket type, priority, and satisfaction scores. Dataset is balanced (~50/50 positive/negative).

### Text Preprocessing Pipeline
1. Lowercase, strip HTML tags, keep alphanumeric characters
2. Simple whitespace tokenisation
3. Build vocabulary from tokens with `min_freq ≥ 2` → vocabulary size: **1,699 tokens**
4. Special tokens: `<PAD>` (idx=0), `<UNK>` (idx=1)
5. Truncate/right-pad sequences to fixed `MAX_LEN`
6. Stratified 70 / 15 / 15 train-val-test split

### Embeddings — GloVe-50 (Transfer Learning)
Training embeddings from scratch on ~1,500 samples is insufficient (results in F1 ≈ 0.16). The fix is loading **GloVe 6B 50-dimensional** pre-trained vectors (trained on 6 billion tokens):

| Approach | Embedding Init | Expected F1 |
|---|---|---|
| From scratch (broken baseline) | Random | ~0.16 |
| **This implementation** | **GloVe-50, fine-tuned** | **~0.65–0.75** |
| Production best practice | DistilBERT / BERT | ~0.85+ |

GloVe coverage on the project vocabulary: **96.1%** (1,632 / 1,699 tokens).

### Model Architecture — `BiLSTMSentiment`
A bidirectional LSTM with GloVe-initialised embeddings (269,527 trainable parameters):
Input tokens

→ Embedding(1699, 50) [GloVe init, fine-tuned]

→ pack_padded_sequence  [ignores padding during LSTM computation]

→ BiLSTM(50 → 128 × 2, bidirectional=True)

→ Dropout(0.5)

→ Linear(256 → 1)  [logit → BCEWithLogitsLoss]

Bidirectionality captures both left-to-right and right-to-left context — important for negations ("not at all happy") and long-form complaints.

### Theoretical Discussion Included
A detailed comparison of sequence models is provided in the notebook:

| Property | Vanilla RNN | LSTM | GRU |
|---|---|---|---|
| Gates | 0 | 3 | 2 |
| Cell state | ✗ | ✓ | ✗ |
| Long-range memory | ✗ | ✓✓ | ✓ |
| Parameters | Fewest | Most | Intermediate |
| Training speed | Fastest | Slowest | Middle |

### Training
- **Loss:** `BCEWithLogitsLoss`
- **Optimizer:** Adam
- `pack_padded_sequence` / `pad_packed_sequence` to mask padding tokens
- Early stopping on validation F1

### Results
Model evaluated with accuracy, precision, recall, and F1 on the held-out test set with a confusion matrix. A **live inference demo** is included — raw text strings are tokenised, encoded, and scored in real time.

---

## Repository Structure
.

├── Deep_Learning_.ipynb      # Main notebook (all 3 parts)

├── checkpoints/

│   └── lead_scoring_xavier.pt   # Part 1 best model checkpoint

├── data/

│   ├── part1/                   # Adult Income dataset

│   ├── part2/                   # MNIST dataset (auto-downloaded)

│   └── glove/

│       └── glove.6B.50d.txt     # GloVe vectors (auto-downloaded ~170 MB)

---

## Requirements
torch>=2.0

torchvision

numpy

pandas

matplotlib

seaborn

scikit-learn

Install with:
```bash
pip install torch torchvision numpy pandas matplotlib seaborn scikit-learn
```

---

## How to Run

1. Open `Deep_Learning_.ipynb` in Google Colab or Jupyter.
2. Run the **Global Setup** cell first — it sets seeds, configures the device, and defines shared plot defaults.
3. Run **Part 1**, **Part 2**, and **Part 3** cells in order. Each part is self-contained after the global setup.
4. GloVe vectors (~170 MB) and MNIST are downloaded automatically on first run.
5. The Adult Income dataset can be obtained from Kaggle (`uciml/adult-census-income`).

---

## Key Design Decisions

- **Xavier init beats Kaiming He** on the Adult Income tabular task — the ReLU-specific advantage of He init does not compensate for its sensitivity on this dataset.
- **GloVe transfer learning is non-negotiable** for Part 3: the dataset (~2,100 samples) is too small for scratch embeddings to learn word semantics.
- **`pack_padded_sequence`** is used in the BiLSTM to correctly ignore padding tokens, preventing the model from learning spurious patterns from zero-padded positions.
- All checkpoints store only the `state_dict`, not the full model object, following PyTorch production best practices.
