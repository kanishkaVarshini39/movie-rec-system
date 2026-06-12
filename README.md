# Netflix Prize Collaboration Filtering Engine

A modular, scalable recommendation system engineered for the 100-million-row Netflix Prize Dataset utilizing **Funk SVD Matrix Factorization**.

## 🛠️ System Architecture & Pipelines
* **Data Processing Pipeline (`src/data_pipeline.py`)**: Optimized chunk-loading data transformations using `pyarrow` storage structures downcast to compact data matrices (`np.int8`, `np.int32`).
* **Model Training Pipeline (`src/model_training.py`)**: Computes global model metrics over full dataset matrices via Stochastic Gradient Descent.
* **Evaluation Framework (`src/evaluation.py`)**: Baseline model verification utilizing statistical RMSE and MAP@10 parameters.
* **Recommender Module (`src/recommender.py`)**: Production class deployment for low-latency Top-K generation loops.

## 🚀 Reproduction Instructions

### 1. Setup Environment
Ensure your source directory contains the native dataset files (`combined_data_1.txt` to `combined_data_4.txt` and `movie_titles.csv`).
```bash
pip install -r requirements.txt

