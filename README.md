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
