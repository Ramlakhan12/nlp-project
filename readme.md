# 📄 Advanced NLP Document Analysis Tool

## 🌟 Overview

The **Advanced NLP Document Analysis Tool** is a comprehensive Streamlit application that allows users to perform in-depth analysis on a collection of text documents. It leverages advanced Natural Language Processing (NLP) techniques, including **Word2Vec** for semantic similarity, **TF-IDF** for traditional similarity, and **Latent Dirichlet Allocation (LDA)** for topic modeling.

The tool provides an interactive interface with visualizations for document statistics, similarity heatmaps, topic distributions, word associations, and word clouds.

## ✨ Features

* **Document Management:** Load multiple text files or use the built-in sample data for quick demonstration.
* **Preprocessing:** Robust text cleaning, tokenization, stopword removal, and lemmatization (with fallbacks if NLTK resources are unavailable).
* **Document Overview:** Metrics on total documents, word counts, and visual bar charts of document lengths.
* **Document Similarity:**
    * Calculates and visualizes document-to-document similarity using **Word2Vec** (semantic similarity) and **TF-IDF** (term-frequency similarity).
    * Displays similarity matrices as heatmaps using `seaborn`.
    * Lists the most similar document pairs based on a configurable threshold.
* **Topic Modeling (LDA):**
    * Performs **Latent Dirichlet Allocation** to uncover latent topics within the document collection.
    * Allows configuration of the number of topics.
    * Displays discovered topics with their keywords.
    * Visualizes Document-Topic distribution using a heatmap.
* **Word Associations (Word2Vec):**
    * Finds the most **similar words** to a selected term, demonstrating the power of word embeddings.
    * Supports **Word Arithmetic** (e.g., *king - man + woman = ?*) to explore vector space relationships.
* **Word Clouds:** Generates visually appealing word clouds for the entire corpus and individual documents, highlighting the most frequent terms after preprocessing.

## 🛠️ Technologies Used

This project is built using Python and leverages key libraries for NLP, Data Science, and interactive visualization:

| Category | Library | Purpose |
| :--- | :--- | :--- |
| **App Framework** | `streamlit` | Creating the interactive web application |
| **NLP** | `nltk`, `gensim` | Text processing, tokenization, lemmatization, Word2Vec, and LDA |
| **Data Handling** | `pandas`, `numpy` | Data manipulation and numerical operations |
| **Machine Learning** | `sklearn` | TF-IDF Vectorization and Cosine Similarity |
| **Visualization** | `matplotlib`, `seaborn`, `plotly.express`, `wordcloud` | Generating interactive and static plots (Heatmaps, Bar Charts, Word Clouds) |

## 🚀 Installation and Usage

### Prerequisites

You need **Python 3.8+** installed on your system.

![alt text](<Screenshot 2025-11-16 131231.png>)
![alt text](<Screenshot 2025-11-16 131331.png>) 
![alt text](<Screenshot 2025-11-16 131317.png>) 
![alt text](<Screenshot 2025-11-16 131252.png>)

### 1. Clone the repository

```bash
git clone [https://github.com/Ram/nlp-document-analyzer.git](https://github.com/Ramlakhan12/nlp-project/blob/ram/nlpp.py)
cd nlp-document-analyzer

![alt text](<Screenshot 2025-11-16 131231.png>)