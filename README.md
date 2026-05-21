# Research_PhD_Codes

This repository contains all the codes and dependencies for the RAG, Regularization techniques & Data Drift modules.

## 1. Data Drift Detection (`Scripts/Data_drift.py`)

Detects and visualizes data drift and concept drift in ML pipelines.

- Monitors feature distribution shifts over time
- Generates drift visualizations saved to `Image/fig1_data_drift.png` and `Image/fig2_concept_drift.png`
- Useful for maintaining model performance in production environments

## 2. Retrieval-Augmented Generation (`Scripts/RAG.py`)

Implements a RAG pipeline for document-grounded question answering.

- Ingests documents (e.g., `data/AI.pdf`, `data/SQL.txt`)
- Retrieves relevant context and augments LLM prompts
- Enables accurate, citation-backed responses from custom knowledge bases

## 3. Regularization Model (`Scripts/regularization_model.py`)

Trains and compares neural network models on MNIST with regularization techniques (Dropout & Batch Normalization).

- Trained on the MNIST dataset (`data/MNIST/raw/`)
- Comparison results logged in `Logs/mnist.log`
- Model performance visualized in `Image/mnist_keras_comparison.png`

## Prerequisites

- Download Python 3.12
- Microsoft VS Code Editor (or any editor of your choice)

### Installation

```bash
pip install -r requirements.txt
```

## Running the Scripts

**Data Drift & Regularization Model:**
```bash
python file_name.py
```

**RAG Pipeline:**
```bash
streamlit run RAG.py
```

> **Note:** Add your `GROQ_API_KEY` to `Config/.env` to run the RAG pipeline.

## Tech Stack

- **Python** — Core language
- **TensorFlow / Keras** — Deep learning models
- **Scikit-learn** — ML utilities and drift detection
- **Pandas / NumPy** — Data manipulation
- **Matplotlib** — Visualization
- **LangChain / CHROMA DB** — RAG pipeline components
- **Streamlit** — UI
