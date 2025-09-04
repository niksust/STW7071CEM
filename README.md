# STW7071CEM – Information Retrieval Coursework  

This repository contains the implementation of two tasks for the **STW7071CEM Information Retrieval** module at Coventry University.  
The coursework demonstrates both **information retrieval techniques** and **text classification** using Python.  

🔗 GitHub Repo: [STW7071CEM](https://github.com/niksust/STW7071CEM)  

---

## 📌 Tasks Overview  

### 1. Vertical Search Engine  
A domain-specific search engine focused on retrieving publications authored by members of Coventry University’s **School of Economics, Finance and Accounting**.  

**Key Features:**  
- Web crawler with polite crawling (robots.txt compliant).  
- Extracts metadata: title, authors, year, publication links, author profiles.  
- Inverted index built using TF–IDF.  
- Query processor with tokenisation, stopword removal, stemming, cosine similarity ranking.  
- Simple web interface for queries (Google Scholar style).  

👉 Detailed instructions are available in [`task1_searchengine/README.md`](task1_searchengine/README.md).  

---

### 2. Document Classification System  
A text classifier that automatically assigns documents into **Politics**, **Business**, or **Health**.  

**Key Features:**  
- Dataset collected from publicly available BBC articles.  
- Preprocessing: tokenisation, lowercasing, stop-word removal, lemmatisation.  
- Feature extraction using TF–IDF.  
- Classification with **Naïve Bayes** algorithm.  
- Evaluation with precision, recall, F1-score, and confusion matrix.  
- Web interface where users input text and receive predictions.  

👉 Detailed instructions are available in [`task2_classifier/README.md`](task2_classifier/README.md).  

---

## ⚙️ Environment Setup  

Both tasks share the respective Python dependencies.  
Install them with:  

```bash
pip install -r requirements.txt
