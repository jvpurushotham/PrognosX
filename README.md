# PrognosX — Industrial Predictive Maintenance & RUL Intelligence

> **An end-to-end predictive maintenance platform that predicts machine degradation, estimates Remaining Useful Life (RUL), detects near-term failure risk, and estimates future failure probability from multivariate sensor time-series data.**

---

## Overview

**PrognosX** is an end-to-end machine learning system for **predictive maintenance and equipment health monitoring**.

Instead of waiting for a machine to fail:

```text
Machine → Failure ❌ → Emergency Maintenance
```

PrognosX aims to predict degradation before failure:

```text
Sensor Data
     ↓
Data Validation
     ↓
Time-Series Processing
     ↓
Feature Engineering
     ↓
Machine Health Analysis
     ↓
┌────────────────────────────────────┐
│                                    │
│  Failure Risk                      │
│  Remaining Useful Life (RUL)       │
│  Survival Probability              │
│                                    │
└────────────────────────────────────┘
     ↓
Explainable Predictions
     ↓
FastAPI
     ↓
Docker
     ↓
CI/CD
```

The final system is designed to answer three operational questions:

> **Will the machine fail soon?**

> **How much useful life remains?**

> **What is the probability of failure at different future horizons?**

---

# Project Objectives

PrognosX contains three major predictive-maintenance tasks.

## 1. Remaining Useful Life Prediction

Predict the number of operating cycles remaining before failure.

```text
Current State
     ↓
ML Model
     ↓
Estimated RUL
```

Example:

```text
Engine ID: 1042

Current Cycle: 150
Predicted RUL: 42 cycles
```

---

## 2. Failure Risk Prediction

Convert the RUL problem into a near-term failure-risk problem.

For example:

```text
RUL <= 30 cycles
        ↓
High Failure Risk
```

The model can estimate:

```text
Probability of failure within:

10 cycles
20 cycles
30 cycles
50 cycles
```

Example:

```text
Failure within 30 cycles → 78%
```

---

## 3. Survival Analysis

Estimate future failure probability over multiple time horizons.

Example:

```text
Current Engine State

10 cycles → 12%
20 cycles → 28%
30 cycles → 47%
40 cycles → 69%
50 cycles → 86%
```

This provides a richer maintenance signal than a simple:

```text
FAIL / NO FAIL
```

---

# Real-World Problem

Industrial equipment contains multiple sensors that continuously measure operating conditions.

Examples:

```text
Temperature
Pressure
RPM
Flow
Voltage
Current
Vibration
Load
```

As equipment degrades, sensor behavior can change.

A simplified degradation pattern may look like:

```text
Healthy
  │
  │
  ▼
Normal Operation
  │
  │
  ▼
Early Degradation
  │
  │
  ▼
Advanced Degradation
  │
  │
  ▼
Failure
```

The objective is to detect this degradation early enough to allow preventive maintenance.

---

# Core Project Idea

Traditional maintenance:

```text
Machine
   ↓
Failure
   ↓
Repair
```

PrognosX:

```text
Machine
   ↓
Sensor Monitoring
   ↓
Health Estimation
   ↓
Degradation Detection
   ↓
Failure Risk
   ↓
RUL Prediction
   ↓
Maintenance Decision
```

---

# Dataset — NASA C-MAPSS

## Primary Dataset

This project uses the **NASA C-MAPSS Jet Engine Simulated Data** as its primary benchmark dataset.

C-MAPSS stands for:

> **Commercial Modular Aero-Propulsion System Simulation**

The dataset consists of multiple multivariate time series representing a fleet of simulated aircraft engines.

Each engine:

```text
Starts in a normal state
        ↓
Experiences degradation
        ↓
Develops a fault
        ↓
Eventually fails
```

The training trajectories run until failure, while the test trajectories stop before failure. NASA also provides the true RUL values corresponding to the final observed point of each test trajectory.

### Official Source

NASA Open Data Portal:

**CMAPSS Jet Engine Simulated Data**

---

# Dataset Structure

Each row contains **26 columns**:

```text
Column 1
→ Unit / Engine Number

Column 2
→ Operating Cycle

Columns 3–5
→ Operational Settings

Columns 6–26
→ Sensor Measurements
```

Therefore:

```text
26 Columns
│
├── 1 Engine ID
├── 1 Cycle
├── 3 Operational Settings
└── 21 Sensor Measurements
```

---

# Dataset Characteristics

The dataset contains:

* Multiple engines
* Multivariate time-series measurements
* Different initial wear levels
* Manufacturing variation
* Operational settings
* Sensor noise
* Run-to-failure trajectories
* Multiple operating conditions
* Multiple fault modes

The initial wear and manufacturing variation should **not automatically be treated as faults**. They represent normal differences between engines.

---

# C-MAPSS Dataset Variants

The project will progressively evaluate the system across all four subsets.

| Dataset   | Train Engines | Test Engines | Operating Conditions | Fault Modes |
| --------- | ------------: | -----------: | --------------------: | ----------- |
| **FD001** |           100 |          100 |                     1 | 1           |
| **FD002** |           260 |          259 |                     6 | 1           |
| **FD003** |           100 |          100 |                     1 | 2           |
| **FD004** |           248 |          249 |                     6 | 2           |

### FD001

```text
1 operating condition
1 fault mode
HPC degradation
```

This will be the **initial development dataset**.

### FD002

```text
6 operating conditions
1 fault mode
HPC degradation
```

This tests robustness against different operating conditions.

### FD003

```text
1 operating condition
2 fault modes

HPC degradation
Fan degradation
```

This tests multiple degradation mechanisms.

### FD004

```text
6 operating conditions
2 fault modes

HPC degradation
Fan degradation
```

This is the final and most challenging benchmark.

Dataset characteristics are based on NASA's current C-MAPSS dataset description.

---

# Why Start With FD001?

The project will not immediately begin with FD004.

Instead:

```text
FD001
  ↓
Baseline System
  ↓
FD002
  ↓
Operating-Condition Robustness
  ↓
FD003
  ↓
Multiple Fault Modes
  ↓
FD004
  ↓
Final Generalized System
```

This allows the project to evolve from a controlled experiment into a robust predictive-maintenance system.

---

# RUL Target Generation

C-MAPSS training trajectories contain complete run-to-failure histories.

Suppose:

```text
Engine 1
Maximum operating cycle = 192
```

Then:

```text
Cycle 1   → RUL = 191
Cycle 2   → RUL = 190
Cycle 3   → RUL = 189
...
Cycle 180 → RUL = 12
Cycle 181 → RUL = 11
...
Cycle 192 → RUL = 0
```

Therefore:

```text
RUL = Final Failure Cycle - Current Cycle
```

This creates the supervised learning target.

---

# Machine Learning Problem

The primary learning problem becomes:

```text
X = Sensor History + Operating Conditions
y = Remaining Useful Life
```

Example:

```text
Input:

Sensor 1
Sensor 2
...
Sensor 21
Operating Setting 1
Operating Setting 2
Operating Setting 3
Historical Features

             ↓

          ML Model

             ↓

Predicted RUL = 42 cycles
```

This is a **regression problem**.

---

# Failure Prediction

C-MAPSS is primarily an RUL dataset.

However, a secondary classification task can be derived from the RUL target.

For example:

```text
RUL <= 30
      ↓
Failure Risk = HIGH
```

while:

```text
RUL > 30
      ↓
Failure Risk = LOW
```

This allows the project to answer:

```text
Will this engine fail within the next N cycles?
```

Multiple horizons can be investigated:

```text
10-cycle failure risk
20-cycle failure risk
30-cycle failure risk
50-cycle failure risk
```

---

# Survival Analysis

Survival analysis will be implemented as an advanced extension.

The objective is to estimate:

```text
P(T <= t)
```

where:

```text
T = time/cycles until failure
t = future horizon
```

Potential models:

```text
Cox Proportional Hazards
Random Survival Forest
Gradient Boosted Survival Models
```

The project will carefully construct validation/censoring scenarios from the run-to-failure trajectories rather than treating ordinary RUL labels as conventional censored survival observations.

---

# Feature Engineering

Raw sensor values alone may not capture degradation sufficiently.

PrognosX will create temporal and degradation-oriented features.

## Rolling Features

```text
Rolling Mean
Rolling Standard Deviation
Rolling Minimum
Rolling Maximum
```

Example:

```text
Sensor 11

Current Value
Rolling Mean (5)
Rolling Std (5)
Rolling Min (5)
Rolling Max (5)
```

---

## Lag Features

```text
Sensor(t-1)
Sensor(t-5)
Sensor(t-10)
```

---

## Trend Features

```text
Slope
Rate of Change
Percentage Change
Linear Trend
```

---

## Degradation Features

```text
Cumulative Change
Moving Average
Exponential Moving Average
Degradation Rate
Health Index
```

---

# Sensor Analysis

EDA will focus on identifying:

```text
Which sensors degrade before failure?

Which sensors are nearly constant?

Which sensors are noisy?

Which sensors are highly correlated?

Which sensors respond to operating conditions?

Which sensors provide the strongest predictive signal?
```

This is more useful than performing only generic exploratory data analysis.

---

# Model Development

## Stage 1 — Baseline Models

### RUL Regression

```text
Linear Regression
Random Forest Regressor
XGBoost Regressor
LightGBM
```

---

## Stage 2 — Deep Learning

```text
LSTM
GRU
Temporal CNN
Transformer
```

These models will operate on sequences of historical sensor observations.

Example:

```text
Cycle t-29
Cycle t-28
Cycle t-27
...
Cycle t-1
Cycle t
```

---

# Model Comparison

Models will be compared systematically rather than selecting a model arbitrarily.

Example:

| Model                  | Task     | Metric     |
| ----------------------- | -------- | ---------- |
| Linear Regression       | RUL      | RMSE / MAE |
| Random Forest           | RUL      | RMSE / MAE |
| XGBoost                 | RUL      | RMSE / MAE |
| LightGBM                | RUL      | RMSE / MAE |
| LSTM                    | RUL      | RMSE / MAE |
| GRU                     | RUL      | RMSE / MAE |
| Transformer              | RUL      | RMSE / MAE |
| Cox Model                | Survival | C-index    |
| Random Survival Forest  | Survival | C-index    |

---

# Evaluation Strategy

## RUL Metrics

Primary metrics:

```text
MAE
RMSE
R²
```

The project will also consider the **NASA/C-MAPSS asymmetric scoring function**, where late RUL predictions can be penalized differently from early predictions.

This is important because:

```text
Predicting too late
```

can have a different operational consequence from:

```text
Predicting too early
```

---

# Failure Classification Metrics

For failure-risk prediction:

```text
Precision
Recall
F1 Score
ROC-AUC
PR-AUC
Confusion Matrix
```

Recall and PR-AUC will receive particular attention because failure events can be treated as higher-cost events.

---

# Survival Metrics

For survival models:

```text
Concordance Index
Brier Score
Calibration
Survival Curves
```

---

# Avoiding Data Leakage

Time-series leakage is one of the most important issues in this project.

The dataset must **not** be randomly split row-by-row.

Incorrect:

```python
train_test_split(df)
```

because the same engine could appear in both training and validation.

Instead:

```text
Engine-level split

Training Engines
       ↓
Validation Engines
       ↓
Test Engines
```

The same engine trajectory should never unintentionally appear across train and validation sets.

---

# Realistic Validation Strategy

C-MAPSS training trajectories run until failure, while test trajectories stop before failure.

Therefore, the project will simulate this situation during validation.

Example:

```text
Original Training Engine

Cycle 1 → Cycle 2 → ... → Cycle 200 → FAILURE
```

Create a simulated observation point:

```text
Cycle 1 → Cycle 2 → ... → Cycle 150
```

The model must predict:

```text
RUL = 50 cycles
```

This better represents the actual inference scenario.

---

# Explainable AI

A predictive-maintenance system should not only say:

```text
Failure Risk = HIGH
```

It should also explain:

```text
Why?
```

PrognosX will use:

```text
SHAP
Permutation Importance
Feature Importance
Partial Dependence
```

Example:

```text
Prediction

Failure Risk: HIGH
RUL: 18 cycles

Top contributing signals:

Sensor 11 → Strong degradation
Sensor 4  → Increasing trend
Sensor 7  → Abnormal behavior
Sensor 2  → High recent variance
```

---

# System Architecture

```text
                  ┌──────────────────────┐
                  │   C-MAPSS Dataset    │
                  │ FD001 → FD004        │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │   Data Ingestion     │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │   Data Validation    │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Transformation       │
                  │ Cleaning / Scaling   │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Feature Engineering  │
                  │ Rolling / Lag /      │
                  │ Trend / Degradation  │
                  └──────────┬───────────┘
                             │
             ┌───────────────┼────────────────┐
             │               │                │
             ▼               ▼                ▼
      ┌─────────────┐ ┌─────────────┐ ┌──────────────┐
      │ Failure     │ │ RUL         │ │ Survival     │
      │ Prediction  │ │ Prediction  │ │ Analysis     │
      └──────┬──────┘ └──────┬──────┘ └──────┬───────┘
             │               │                │
             └───────────────┼────────────────┘
                             ▼
                  ┌──────────────────────┐
                  │ Explainability       │
                  │ SHAP / Importance    │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Model Registry       │
                  │ Versioning           │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Inference Pipeline   │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ FastAPI              │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Docker               │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ GitHub Actions       │
                  │ CI/CD                │
                  └──────────────────────┘
```

---

# Repository Structure

```text
PrognosX/
│
├── .github/
│   └── workflows/
│       └── ci-cd.yml
│
├── data/
│   ├── raw/
│   │   └── CMAPSS/
│   │
│   ├── processed/
│   │
│   └── external/
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_sensor_analysis.ipynb
│   ├── 04_rul_generation.ipynb
│   ├── 05_feature_engineering.ipynb
│   ├── 06_baseline_models.ipynb
│   ├── 07_xgboost_model.ipynb
│   ├── 08_lstm_gru_model.ipynb
│   ├── 09_survival_analysis.ipynb
│   ├── 10_model_comparison.ipynb
│   └── 11_shap_analysis.ipynb
│
├── src/
│   ├── __init__.py
│   │
│   ├── config/
│   │   └── config.py
│   │
│   ├── data/
│   │   ├── ingestion.py
│   │   ├── validation.py
│   │   └── transformation.py
│   │
│   ├── features/
│   │   └── feature_engineering.py
│   │
│   ├── models/
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── predict.py
│   │
│   ├── pipelines/
│   │   ├── training-pipeline.py
│   │   └── inference-pipeline.py
│   │
│   ├── api/
│   │   └── app.py
│   │
│   ├── utils/
│   │   ├── logger.py
│   │   └── exception.py
│   │
│   └── entity/
│       └── config_entity.py
│
├── models/
│   ├── failure/
│   ├── rul/
│   └── survival/
│
├── reports/
│   ├── figures/
│   ├── metrics/
│   └── experiments/
│
├── logs/
│
├── tests/
│   ├── test_ingestion.py
│   ├── test_validation.py
│   ├── test_features.py
│   ├── test_models.py
│   ├── test_pipeline.py
│   └── test_api.py
│
├── requirements.txt
├── setup.py
├── Dockerfile
├── docker-compose.yml
├── README.md
└── .env
```

---

# Directory Responsibilities

## `data/raw/`

Contains the original downloaded C-MAPSS files.

The raw dataset should remain unchanged.

```text
data/raw/CMAPSS/
```

---

## `data/processed/`

Contains:

```text
Cleaned datasets
RUL-labeled datasets
Engine-level splits
Feature datasets
Sequence datasets
```

---

## `notebooks/`

Used for experimentation and visualization.

The final production logic should gradually move from notebooks into `src/`.

---

## `src/data/`

### `ingestion.py`

Responsible for:

```text
Dataset loading
File extraction
Reading raw files
Data registration
```

### `validation.py`

Checks:

```text
Missing values
Duplicates
Data types
Sensor ranges
Engine IDs
Cycle ordering
Unexpected values
```

### `transformation.py`

Handles:

```text
Cleaning
Scaling
Normalization
Train/validation preparation
Sequence preparation
```

---

# `src/features/`

`feature_engineering.py` will generate:

```text
Rolling statistics
Lag features
Trend features
Rate-of-change features
Degradation indicators
Health indicators
```

---

# `src/models/`

### `train.py`

Model training.

### `evaluate.py`

Model evaluation.

### `predict.py`

Production prediction.

---

# Training Pipeline

```text
Raw Data
   ↓
Ingestion
   ↓
Validation
   ↓
Transformation
   ↓
RUL Generation
   ↓
Feature Engineering
   ↓
Train/Validation Split
   ↓
Model Training
   ↓
Evaluation
   ↓
Model Selection
   ↓
Model Saving
```

---

# Inference Pipeline

```text
New Sensor Data
       ↓
Validation
       ↓
Preprocessing
       ↓
Feature Engineering
       ↓
Trained Model
       ↓
Prediction
       ↓
Risk Calculation
       ↓
API Response
```

---

# FastAPI

The production inference service will expose:

```text
GET  /health

POST /predict

POST /predict/rul

POST /predict/failure

POST /predict/survival

GET  /model/info
```

Example:

```json
{
  "machine_id": "1042",
  "failure_probability": 0.78,
  "predicted_rul": 42,
  "risk_level": "HIGH"
}
```

---

# Docker

The application will be containerized using:

```text
Docker
Docker Compose
```

The container will include:

```text
FastAPI
ML models
Python dependencies
Configuration
Inference pipeline
```

---

# CI/CD

GitHub Actions will automate:

```text
Git Push
   ↓
Install Dependencies
   ↓
Run Tests
   ↓
Run Linting
   ↓
Build Docker Image
   ↓
Integration Tests
   ↓
Publish Artifact/Image
   ↓
Deploy
```

Workflow:

```text
.github/workflows/ci-cd.yml
```

---

# Testing Strategy

Testing will cover:

```text
Data ingestion
Data validation
RUL generation
Feature engineering
Model inference
Pipeline execution
API endpoints
```

Example:

```text
Unit Tests
     +
Integration Tests
     +
API Tests
```

---

# Reporting

The `reports/` directory will contain:

```text
RUL prediction plots
Actual vs Predicted RUL
Sensor degradation plots
Feature importance
SHAP plots
Confusion matrices
ROC curves
Precision-Recall curves
Survival curves
Model comparison
Experiment results
```

---

# Final Project Roadmap

## Phase 1 — Project Foundation

* [x] Create GitHub repository
* [x] Create repository structure
* [x] Configure Python environment
* [x] Create `requirements.txt`
* [x] Configure logging
* [x] Configure project settings

---

## Phase 2 — Dataset Acquisition

* [ ] Download C-MAPSS
* [ ] Store raw files
* [x] Understand FD001–FD004
* [x] Document dataset metadata
* [x] Implement ingestion
* [x] Implement validation

---

## Phase 3 — FD001 Exploration

* [x] Load FD001
* [x] Assign column names
* [x] Identify engines
* [x] Analyze cycle distributions
* [x] Analyze sensor distributions
* [x] Identify constant sensors
* [x] Analyze sensor correlations
* [x] Plot degradation trajectories

---

## Phase 4 — RUL Engineering

* [ ] Calculate training RUL
* [ ] Validate RUL generation
* [ ] Create RUL datasets
* [ ] Understand test RUL format
* [ ] Build engine-level train/validation split

---

## Phase 5 — Feature Engineering

* [ ] Rolling mean
* [ ] Rolling standard deviation
* [ ] Rolling min/max
* [ ] Lag features
* [ ] Rate of change
* [ ] Sensor trends
* [ ] Degradation indicators
* [ ] Feature selection

---

## Phase 6 — Baseline Models

Train:

```text
Linear Regression
Random Forest
XGBoost
LightGBM
```

* [ ] Establish baseline RMSE
* [ ] Calculate MAE
* [ ] Calculate R²
* [ ] Generate prediction plots
* [ ] Compare models

---

## Phase 7 — Advanced RUL Models

Implement:

```text
LSTM
GRU
Temporal CNN
Transformer
```

* [ ] Create sequences
* [ ] Train models
* [ ] Tune hyperparameters
* [ ] Compare with XGBoost
* [ ] Analyze inference performance

---

## Phase 8 — Failure Risk Modeling

Derive classification targets:

```text
Failure within 10 cycles
Failure within 20 cycles
Failure within 30 cycles
Failure within 50 cycles
```

Train:

```text
Logistic Regression
Random Forest
XGBoost
LightGBM
```

Evaluate:

```text
Precision
Recall
F1
ROC-AUC
PR-AUC
```

---

## Phase 9 — Survival Analysis

Implement:

```text
Cox Proportional Hazards
Random Survival Forest
```

Generate:

```text
Survival probability
Failure probability
Hazard estimates
Risk curves
```

Evaluate:

```text
C-index
Brier Score
Calibration
```

---

## Phase 10 — Explainability

Implement:

```text
SHAP
Permutation Importance
Feature Importance
```

Answer:

```text
Which sensors drive degradation?

Which sensors drive failure risk?

What features influence RUL?
```

---

## Phase 11 — Production Pipeline

Move notebook logic into:

```text
src/data/
src/features/
src/models/
src/pipelines/
```

Create:

```text
training-pipeline.py
inference-pipeline.py
```

---

## Phase 12 — API

Build FastAPI.

Implement:

```text
/health
/predict
/predict/rul
/predict/failure
/predict/survival
/model/info
```

---

## Phase 13 — Containerization

* [ ] Create Dockerfile
* [ ] Create docker-compose.yml
* [ ] Build image
* [ ] Run API inside container
* [ ] Test containerized inference

---

## Phase 14 — CI/CD

* [ ] GitHub Actions
* [ ] Automated tests
* [ ] Linting
* [ ] Docker build
* [ ] Integration tests
* [ ] Image publishing
* [ ] Deployment

---

## Phase 15 — FD002

Evaluate the same pipeline under:

```text
Multiple operating conditions
```

Focus on:

```text
Condition normalization
Feature robustness
Model generalization
```

---

## Phase 16 — FD003

Evaluate:

```text
Multiple fault modes
```

Analyze:

```text
HPC degradation
Fan degradation
```

---

## Phase 17 — FD004

Final benchmark:

```text
Multiple operating conditions
+
Multiple fault modes
```

This becomes the project's final generalized evaluation.

---

# Final System

The completed PrognosX platform should produce something like:

```text
╔══════════════════════════════════════════════╗
║             PROGNOSX HEALTH REPORT           ║
╠══════════════════════════════════════════════╣
║ Engine ID:             1042                  ║
║ Current Cycle:         150                   ║
║                                              ║
║ Predicted RUL:         42 cycles             ║
║ Failure Risk:          78%                   ║
║ Risk Level:            HIGH                  ║
║                                              ║
║ Failure Probability                         ║
║                                              ║
║ 10 cycles:             18%                   ║
║ 20 cycles:             41%                   ║
║ 30 cycles:             61%                   ║
║ 50 cycles:             86%                   ║
║                                              ║
║ Top Risk Factors:                            ║
║                                              ║
║ 1. Sensor 11 — Degradation trend             ║
║ 2. Sensor 4  — Increased variance            ║
║ 3. Sensor 7  — Abnormal operating pattern    ║
║                                              ║
║ Recommendation:                              ║
║ Schedule preventive inspection               ║
╚══════════════════════════════════════════════╝
```

---

# Future Dashboard

A future dashboard can visualize:

```text
Fleet Health
Individual Engine Health
RUL Distribution
Failure Risk
Sensor Trends
Degradation Curves
Survival Curves
Model Explanations
Maintenance Alerts
```

Possible stack:

```text
React / Next.js
       +
FastAPI
       +
Plotly
```

---

# Advanced Research Extensions

After the core system is complete, additional experiments can include:

### 1. Multitask Learning

Predict simultaneously:

```text
RUL
+
Failure Risk
+
Health Score
```

---

### 2. Attention-Based Models

Use:

```text
Transformer
Temporal Attention
Sensor Attention
```

to determine which historical time steps and sensors are most important.

---

### 3. Health Index Modeling

Construct a continuous:

```text
Machine Health Index
```

such as:

```text
1.00 → Healthy
0.80 → Normal
0.60 → Degrading
0.30 → Critical
0.00 → Failure
```

---

### 4. Model Ensemble

Combine:

```text
XGBoost
+
LSTM
+
Survival Model
```

to produce a combined health estimate.

---

### 5. Uncertainty Estimation

Instead of:

```text
RUL = 42
```

produce:

```text
Predicted RUL = 42 cycles

95% Prediction Interval:
35–51 cycles
```

This makes the system more realistic for maintenance decision-making.

---

### 6. Data Drift Monitoring

Monitor whether production sensor distributions change:

```text
Training Distribution
        ↓
Production Distribution
        ↓
Drift Detection
```

Potential techniques:

```text
PSI
KS Test
Distribution Monitoring
```

---

# Technology Stack

## Programming

```text
Python
SQL
Bash
```

## Data Processing

```text
Pandas
NumPy
SciPy
```

## Machine Learning

```text
Scikit-learn
XGBoost
LightGBM
```

## Deep Learning

```text
PyTorch
```

## Survival Analysis

```text
lifelines
scikit-survival
```

## Visualization

```text
Matplotlib
Plotly
```

## API

```text
FastAPI
Uvicorn
Pydantic
```

## Testing

```text
Pytest
```

## Deployment

```text
Docker
Docker Compose
GitHub Actions
```

---

# Dataset & Research References

## Primary Dataset

**NASA C-MAPSS Jet Engine Simulated Data**

Official NASA Open Data Portal:

[NASA C-MAPSS Dataset](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data?utm_source=chatgpt.com)

The NASA page identifies the dataset as public and provides the C-MAPSS ZIP resource.

## NASA Prognostics Data Repository

NASA's Prognostics Center of Excellence maintains a repository of prognostic datasets intended for development and benchmarking of prognostic algorithms.

[NASA Prognostics Data Repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/?utm_source=chatgpt.com)

---

# Original C-MAPSS Reference

A. Saxena, K. Goebel, D. Simon, and N. Eklund,

**"Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation"**

Proceedings of the 1st International Conference on Prognostics and Health Management (PHM08), Denver, Colorado, 2008.

---

# Project Development Strategy

The project follows a **baseline → advanced → production → generalized benchmark** strategy.

```text
                FD001
                  │
                  ▼
        RUL Baseline System
                  │
                  ▼
             XGBoost
                  │
                  ▼
          Feature Engineering
                  │
                  ▼
            LSTM / GRU
                  │
                  ▼
         Failure Prediction
                  │
                  ▼
         Survival Analysis
                  │
                  ▼
          SHAP Explainability
                  │
                  ▼
            FastAPI
                  │
                  ▼
             Docker
                  │
                  ▼
             CI/CD
                  │
                  ▼
              FD002
                  │
                  ▼
              FD003
                  │
                  ▼
              FD004
                  │
                  ▼
        Final PrognosX System
```

---

# What Makes PrognosX Different?

The project demonstrates more than model training.

```text
✓ Multivariate Time-Series Analysis
✓ Predictive Maintenance
✓ Remaining Useful Life Prediction
✓ Failure Risk Classification
✓ Survival Analysis
✓ Temporal Feature Engineering
✓ XGBoost / LightGBM
✓ LSTM / GRU / Transformer
✓ Explainable AI
✓ Model Evaluation
✓ Data Leakage Prevention
✓ Production ML Pipeline
✓ FastAPI
✓ Docker
✓ Automated Testing
✓ GitHub Actions
✓ CI/CD
✓ Model Versioning
✓ Data Drift Monitoring
✓ Multi-dataset Benchmarking
```

The strongest part is the progression from:

```text
FD001
Simple operating environment
        ↓
FD002
Multiple operating conditions
        ↓
FD003
Multiple degradation modes
        ↓
FD004
Multiple conditions + multiple degradation modes
```

This demonstrates **model generalization**, not just performance on a single dataset.

---

# Final Vision

PrognosX aims to transform raw sensor measurements into actionable maintenance intelligence.

```text
RAW SENSOR DATA
       ↓
"What is the machine doing?"
       ↓
DEGRADATION ANALYSIS
       ↓
"Is the machine getting worse?"
       ↓
FAILURE RISK
       ↓
"Is failure approaching?"
       ↓
RUL PREDICTION
       ↓
"How much useful life remains?"
       ↓
SURVIVAL ANALYSIS
       ↓
"How likely is failure at different horizons?"
       ↓
EXPLAINABILITY
       ↓
"Why does the model think this?"
       ↓
MAINTENANCE DECISION
```

### Final objective:

> **Build an industrial-grade predictive-maintenance intelligence platform that can detect degradation early, estimate Remaining Useful Life, quantify future failure risk, explain the underlying sensor signals, and expose predictions through a production-ready API.**

---

# 👨‍💻 Author

**Purushotham J V**

Computer Science — Data Science & Data Engineering

**Interests:** Machine Learning · Predictive Analytics · Data Engineering · AI · MLOps

---

# ⭐ **PrognosX**

### *Industrial Predictive Maintenance & Remaining Useful Life Intelligence*

> **Predict degradation. Estimate lifetime. Prevent failure.**

---

# 📌 Implementation Status

This repository currently implements **Phases 1–3** in full (see
`PHASE_PROGRESS.md` for the detailed checklist, and `SETUP.md` for how to
get it running with your dataset). Phases 4–17 are scaffolded as
placeholders so the structure matches the target architecture above, and
will be implemented incrementally.
