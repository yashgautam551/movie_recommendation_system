# 🎬 Movie Recommendation System (Content-Based)

## 📌 Overview
This project implements a **content-based movie recommendation system** using **machine learning and Natural Language Processing (NLP)**.
It recommends movies similar to a given movie by analyzing **textual similarity** between movie descriptions and genres.

---

## 🚀 Features
- Content-based recommendation system
- NLP-based text processing
- TF-IDF vectorization
- Cosine similarity for movie ranking
- Clean and beginner-friendly implementation

---

## 🛠️ Tech Stack
- Python
- Pandas
- Scikit-learn
- Natural Language Processing (TF-IDF)

---

## 🧠 How It Works
1. Combine movie overview and genre information
2. Convert text data into numerical vectors using TF-IDF
3. Compute cosine similarity between movies
4. Recommend top similar movies based on similarity scores

---

## 📂 Project Structure
movie-recommendation-system/
│
├── data/
│ └── movies.csv
├── recommender.py
├── requirements.txt
└── README.md

---

## ▶️ How to Run the Project
```bash
pip install -r requirements.txt
python recommender.py

🎬 Recommendations for 'Inception':

➡️ Interstellar
➡️ The Matrix
➡️ Avatar
➡️ The Dark Knight
➡️ The Avengers
