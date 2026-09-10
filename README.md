# 💭 Sentiment & Emotion Analyzer — Streamlit

A complete Streamlit NLP application based on the uploaded `NLP_Project.ipynb`.

## What the notebook uses

The original notebook:
- reads `train.txt` using `sep=";"` with columns `text` and `emotions`
- converts text to lowercase
- removes punctuation
- removes numbers
- removes non-ASCII characters
- removes English stopwords
- uses an 80/20 train-test split with `random_state=42`
- compares Bag-of-Words/Naive Bayes and TF-IDF/Naive Bayes
- finishes with TF-IDF + `LogisticRegression(max_iter=1000)`

The notebook's final Logistic Regression result is approximately **86.28% accuracy** on its test split. The Streamlit app retrains that final model from the dataset when it starts, so the displayed score is calculated from the dataset currently loaded.

## Project structure

```text
sentiment_streamlit_app/
│
├── app.py
├── requirements.txt
├── README.md
└── train.txt          # put your dataset here
```

## Dataset format

Your `train.txt` should look like:

```text
i didnt feel humiliated;sadness
i feel happy today;joy
i am very angry;anger
```

The application expects exactly two logical columns:

```text
text;emotions
```

No header is required because the app assigns the column names automatically.

## Run in VS Code

### 1. Create a virtual environment

```powershell
python -m venv .venv
```

### 2. Activate it

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can run:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 3. Install packages

```powershell
python -m pip install -r requirements.txt
```

### 4. Put `train.txt` beside `app.py`

```text
sentiment_streamlit_app/
├── app.py
├── requirements.txt
└── train.txt
```

### 5. Start Streamlit

```powershell
python -m streamlit run app.py
```

This `python -m streamlit` command is useful when Windows says:

`streamlit is not recognized as the name of a cmdlet...`

## Features

### 🔮 Predict
Enter a sentence and get:
- predicted emotion
- confidence
- cleaned text
- probability chart for all classes

### 📊 Dashboard
Shows:
- number of dataset rows
- number of emotion classes
- TF-IDF vocabulary size
- test accuracy
- emotion distribution
- confusion matrix
- dataset preview

### 🧠 Model Details
Shows:
- preprocessing pipeline
- model configuration
- classification report
- emotion labels

## Important terminology

Your dataset contains an `emotions` column, so technically this project is **emotion classification** rather than binary positive/negative sentiment analysis. A good GitHub project title is:

**NLP Sentiment & Emotion Analyzer using TF-IDF and Logistic Regression**
