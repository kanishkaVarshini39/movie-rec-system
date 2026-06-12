# Netflix Prize Collaboration Filtering Engine

A modular, scalable recommendation system engineered for the 100-million-row Netflix Prize Dataset utilizing **Funk SVD Matrix Factorization**.

## 🛠️ System Architecture & Pipelines
* **Data Processing Pipeline (`src/data_pipeline.py`)**: Optimized chunk-loading data transformations using `pyarrow` storage structures downcast to compact data matrices (`np.int8`, `np.int32`).
* **Model Training Pipeline (`src/model_training.py`)**: Computes global model metrics over full dataset matrices via Stochastic Gradient Descent.
* **Evaluation Framework (`src/evaluation.py`)**: Baseline model verification utilizing statistical RMSE and MAP@10 parameters.
* **Recommender Module (`src/recommender.py`)**: Production class deployment for low-latency Top-K generation loops.

## 🏗️ Repository Architecture

The repository is structured to separate offline pipeline operations (ingestion, training, evaluation) from live application delivery frameworks:

```text
netflix-prize-recommender/
│
├── Documents/                  # necessary documents
│   ├── Technical report.pdf             
│   └── Presentation.pdf         
│                      
│
├── src/                        # Core Data Science Pipeline Modules
│   ├── __init__.py
│   ├── data_pipeline.py        # Streamlined memory-optimized chunk loader
│   ├── model_training.py       # Global production model builder
│   ├── evaluation.py           # Verification scripts (RMSE & MAP@10 benchmarks)
│   └── recommender.py          # Class engine loaded by the deployment layer
│
├── app.py                      # Main production deployment orchestrator
├── backend.py                  # API service layer (FastAPI/Flask service backend)
├── frontend.py                 # User Interface client layer (Streamlit/UI)
├── .gitignore                  # Active binary and dataset filter
├── requirements.txt            # Project environment dependencies
└── Recommendation_system.ipynb # Prototyping and EDA sandbox notebook

## 🚀 Reproduction & Deployment Guide

Ensure your native source directory holds your downloaded dataset files (`combined_data_1.txt` to `combined_data_4.txt` and `movie_titles.csv`) before running the pipelines.

---

### 1. Initialize Project Environment
Install all core data structures, matrix optimization libraries, and application dependencies

### 2.Run the Data Preparation Ingestion
Process the raw source text files, apply the activity filters, and compile them into optimized, compressed Parquet storage formats

### 3.Generate Comparative Validation Reports
Verify algorithm precision, latency profiles, and ranking scores (Funk SVD vs. KNN) for your documentation metrics on unseen test slices

### 4.Compute and Save the Global Model
Compile the complete dataset weight parameters and freeze the production model binary to disk

### 5.Launch the Web Application
Start the backend API service
Launch the main production orchestration gateway app
