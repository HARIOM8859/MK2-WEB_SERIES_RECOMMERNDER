# 🎬 Web Series Recommendation System

An intelligent **Web Series Recommendation System** that suggests similar TV shows based on user selection. The system analyzes multiple attributes such as genres, cast, keywords, and storyline descriptions to generate accurate and personalized recommendations.

---

## 🚀 Features

* 🔎 Search any web series from the dataset
* 🎯 Get top 5 similar web series recommendations instantly
* 🖼️ Displays posters using TMDB API
* ⚡ Fast recommendation using preprocessed similarity matrix
* 💻 Interactive web interface built with Streamlit

---

## 🛠️ Technologies Used

* **Python**
* **Pandas & NumPy** – Data processing
* **Scikit‑learn** – Feature engineering & similarity computation
* **Streamlit** – Web app interface
* **TMDB API** – Fetching posters
* **Pickle** – Model storage

---

## 📂 Dataset

* Source: TMDB TV Shows Dataset
* Size: ~168,000 entries before cleaning
* Important columns used:

  * Title
  * Overview
  * Genres
  * Cast
  * Keywords
  * Vote Average

---

## ⚙️ How It Works

1. Data is cleaned and important features are selected.
2. Textual and categorical attributes are combined.
3. Feature vectors are created using machine learning techniques.
4. Cosine similarity is calculated between series.
5. When a user selects a show, the system returns the most similar series.

---

## 🖥️ Installation & Running Locally

### Step 1: Clone Repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Run the App

```bash
streamlit run app.py
```

---

## 📸 Output

* Displays recommended web series with posters
* Shows top similar results instantly

---

## 📈 Future Improvements

* Add user login & personalized recommendations
* Deploy on cloud (Streamlit / AWS)
* Hybrid recommendation using user ratings
* Add filtering by platform (Netflix, Prime, etc.)

---

## 👨‍💻 Author

**Hariom Yadav**
B.Tech CSE Student
Machine Learning & Web Developer

---

## ⭐ If you like this project

Give it a star on GitHub!
