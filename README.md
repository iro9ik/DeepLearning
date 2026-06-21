# Intelligent CRM & BI Engine — Deep Learning Notebook

**Author:** Ahmed Abounaoum  
**Supervisor:** Zineb H.  
**Class:** 4DIAM G4  
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
