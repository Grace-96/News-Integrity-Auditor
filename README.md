# 🛡️ TRUTH ENGINE v2.5: Neural Disinformation Auditor

A high-performance web application designed to audit the integrity of news articles using machine learning.

## 🚀 Live Demo
https://news-integrity-auditor-5ejfyjtgyfyg3ceqggwqua.streamlit.app/

## 🔬 How it Works
This engine utilizes a **Calibrated Support Vector Machine (SVM)** to analyze linguistic patterns. 
By processing text through a **TF-IDF Vectorizer**, it identifies specific markers associated with factual reporting vs. deceptive propaganda.

### Key Features:
* **Forensic Evidence Table**: See exactly which words influenced the verdict.
* **Uncertainty Audit Zone**: Flags articles between 35% and 65% for manual review.
* **Intelligence Reports**: Export your findings as timestamped text files.

## 🛠️ Tech Stack
* **Python** & **Streamlit** (Interface)
* **Scikit-Learn** (SVM Model)
* **NLTK** (Natural Language Processing)
* **Hugging Face** (Model Hosting)
