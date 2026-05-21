# Research_PhD_Codes
This Repository contains all the codes and dependencies for the RAG, Regularization techniques &amp; Data Drift modules

1. Data Drift Detection (Scripts/Data_drift.py)
Detects and visualizes data drift and concept drift in ML pipelines.

    11.Monitors feature distribution shifts over time
    2.Generates drift visualizations saved to Image/fig1_data_drift.png and Image/fig2_concept_drift.png
    3.Useful for maintaining model performance in production environments

2. Retrieval-Augmented Generation (Scripts/RAG.py)
Implements a RAG pipeline for document-grounded question answering.

    1.Ingests documents (e.g., data/AI.pdf, data/SQL.txt)
    2.Retrieves relevant context and augments LLM prompts
    3.Enables accurate, citation-backed responses from custom knowledge bases

3. Regularization Model (Scripts/regularization_model.py)
Trains and compares neural network models on MNIST with regularization techniques - with Dropout & BN
    1.Trained on the MNIST dataset (data/MNIST/raw/)
    2.Comparison results logged in Logs/mnist.log
    3.Model performance visualized in Image/mnist_keras_comparison.png

Prerequisites
Download Python 3.12, Microsoft VS Code Editor (or any editor of your choice)

pip install -r requirements.txt

To run Data_drift.py & regularization_model.py : python file_name.py
To run RAG.py : streamlit run RAG.py
(Add your GROQ_API_KEY : to run this file in Config/.env)

TECH STACK
    Python — Core language
    TensorFlow / Keras — Deep learning models
    Scikit-learn — ML utilities and drift detection
    Pandas / NumPy — Data manipulation
    Matplotlib — Visualization
    LangChain / CHROMA DB — RAG pipeline components
    Streamlit - UI
