# 🧠 End-to-End Question Answering System with Transformers + BM25

This is a full-stack implementation of a Question Answering (QA) system that combines deep learning and traditional retrieval techniques. It fine-tunes a transformer model on the SQuAD v2 dataset and integrates a BM25 retriever to answer questions with high precision.

## 🚀 Features

- Fine-tunes `deepset/roberta-base-squad2` on the SQuAD v2 dataset
- Uses BM25 to retrieve the most relevant context
- Interactive Gradio interface for live question answering
- MLflow integration for experiment tracking

---

## 🛠 Tech Stack

- **HuggingFace Transformers**: for model fine-tuning and inference
- **BM25 (rank_bm25)**: for traditional keyword-based retrieval
- **Gradio**: for interactive UI
- **MLflow**: for logging experiments
- **SQuAD v2**: as the training dataset

---

## 📂 Project Structure

