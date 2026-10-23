# 🧠 Text Emotion Classifier

An interactive NLP app that predicts the emotion expressed in a sentence using **TF-IDF + Logistic Regression**. Explore predictions, emotion probabilities, dataset statistics, and model evaluation in a Streamlit dashboard.

## 🚀 Try the live app

### 👉 **[Open Text Emotion Classifier](https://abhi-nlp-emotion-classification.streamlit.app/)**

No installation is needed to try the hosted app. Enter a sentence in the Predict tab, or explore the Dashboard and Model Details tabs.

[![Open Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Open_App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://abhi-nlp-emotion-classification.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Built_with-Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)

## ✨ Features

- **Emotion prediction:** Enter a sentence and view the predicted emotion and model confidence.
- **Probability chart:** Compare the model's estimated probabilities across emotion classes.
- **Interactive dashboard:** Explore dataset size, emotion classes, vocabulary size, test accuracy, and emotion distribution.
- **Model evaluation:** Inspect a confusion matrix and classification report.
- **Text preprocessing:** Lowercase text and remove punctuation, numbers, non-ASCII characters, and English stopwords.
- **Dataset upload:** Upload a compatible text or CSV file from the sidebar.

## 🧪 Machine-learning pipeline

```text
Input text
   ↓
Text cleaning and stopword removal
   ↓
TF-IDF feature extraction
   ↓
Logistic Regression classifier
   ↓
Predicted emotion + class probabilities
```

The current app trains a TF-IDF vectorizer and `LogisticRegression(max_iter=1000)` on startup using an 80/20 train-test split with `random_state=42`. The evaluation metrics are calculated from that split. The notebook's original final model reported approximately **86.28% accuracy**; your deployed app displays the metric calculated from the dataset loaded at runtime.

> **Important:** This is an emotion-classification model, not a separate positive/negative/neutral sentiment model. Emotion predictions are estimates and may be wrong for ambiguous, sarcastic, or context-dependent text.

## 📁 Repository contents

```text
NLP-Emotion-Classification/
├── app.py             # Streamlit interface and ML pipeline
├── train.txt          # Emotion-labelled text dataset
├── requirements.txt   # Python dependencies
├── .streamlit/
│   └── config.toml    # App theme
└── README.md
```

The dataset uses semicolon-separated examples, with the text in the first column and the emotion label in the second. No header row is required.

Example:

```text
i didnt feel humiliated;sadness
i feel happy today;joy
i am very angry;anger
```

## 💻 Run locally

Requires Python and pip.

```powershell
# Clone the repository
git clone https://github.com/abhiii04-25/NLP-Emotion-Classification.git
cd NLP-Emotion-Classification

# Create and activate a virtual environment (Windows PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install the dependencies
python -m pip install -r requirements.txt

# Launch the app
python -m streamlit run app.py
```

Then open the local URL Streamlit prints in the terminal, usually `http://localhost:8501`.

## 🌐 Deployment

This project is hosted on **Streamlit Community Cloud**. The application entry point is `app.py`, the branch is `main`, and dependencies are listed in `requirements.txt`.

**Live link:** [https://abhi-nlp-emotion-classification.streamlit.app/](https://abhi-nlp-emotion-classification.streamlit.app/)

## 🛠️ Tech stack

- Python
- Pandas and NumPy
- NLTK text preprocessing
- Scikit-learn (TF-IDF, Logistic Regression, model metrics)
- Matplotlib
- Streamlit

---

**Project:** Text Emotion Classifier  
**Source code:** [GitHub repository](https://github.com/abhiii04-25/NLP-Emotion-Classification)  
**Live demo:** [Try the app](https://abhi-nlp-emotion-classification.streamlit.app/)
