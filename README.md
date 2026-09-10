# NLP-Emotion-Classification
A Natural Language Processing application that classifies text into emotions using TF-IDF and Logistic Regression, with an interactive Streamlit interface for real-time predictions and model performance analysis
# 💭 NLP Emotion Classification with Streamlit

An end-to-end **Natural Language Processing (NLP)** project that classifies text into different emotions using **TF-IDF Vectorization** and **Logistic Regression**.

The project includes an interactive **Streamlit web application** where users can enter text and receive a predicted emotion along with the model's confidence and performance information.

---

## 🚀 Project Overview

This project demonstrates a complete NLP machine learning workflow:

**Text Dataset → Text Preprocessing → TF-IDF → Logistic Regression → Emotion Prediction → Streamlit Application**

The model learns patterns from text data and predicts the emotion associated with a given sentence.

> **Note:** Although the application is sometimes referred to as a sentiment analyzer, the target column in this project is `emotions`, so the task is technically **multi-class emotion classification**.

---

## ✨ Features

* 📝 Text preprocessing
* 🔤 Lowercase conversion
* ✂️ Punctuation removal
* 🔢 Number removal
* 🚫 Non-ASCII character removal
* 🛑 English stopword removal
* 📊 TF-IDF feature extraction
* 🤖 Logistic Regression classification
* 🎯 Emotion prediction
* 📈 Prediction confidence
* 📊 Emotion probability visualization
* 📉 Confusion matrix
* 📋 Classification report
* 📊 Dataset statistics
* 🌐 Interactive Streamlit interface

---

## 🛠️ Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **NLTK**
* **Scikit-learn**
* **Matplotlib**
* **Streamlit**

### Machine Learning

* TF-IDF Vectorizer
* Logistic Regression
* Train-Test Split
* Accuracy Score
* Confusion Matrix
* Classification Report

---

## 📂 Project Structure

```text
NLP-Emotion-Classification-Streamlit/
│
├── app.py
├── train.txt
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🧠 Machine Learning Workflow

### 1. Load Dataset

The project uses a text dataset containing two columns:

```text
text
emotions
```

Example:

```text
I am feeling really happy today;joy
I am extremely angry;anger
I feel lonely and sad;sadness
```

---

### 2. Text Preprocessing

The raw text is cleaned before training.

The preprocessing pipeline includes:

```text
Original Text
      ↓
Lowercase
      ↓
Remove Punctuation
      ↓
Remove Numbers
      ↓
Remove Non-ASCII Characters
      ↓
Remove Stopwords
      ↓
Clean Text
```

---

### 3. TF-IDF Vectorization

The cleaned text is converted into numerical features using:

```python
TfidfVectorizer()
```

TF-IDF helps the machine learning model identify words that are important for distinguishing between different emotions.

---

### 4. Train-Test Split

The dataset is divided into:

```text
80% → Training Data
20% → Testing Data
```

The project uses:

```python
random_state=42
```

---

### 5. Logistic Regression

The final classification model is:

```python
LogisticRegression(max_iter=1000)
```

The model learns the relationship between text features and emotion labels.

---

## 📊 Model Performance

The Logistic Regression model achieved approximately:

**86.28% test accuracy**

Performance may vary slightly depending on the dataset and preprocessing environment.

---

## 🌐 Streamlit Application

The project includes an interactive Streamlit application with three main sections.

### 🔮 Predict

Users can enter a sentence and receive:

* Predicted emotion
* Prediction confidence
* Cleaned text
* Emotion probabilities

Example:

```text
Input:
I am feeling really happy today!

Output:
Predicted emotion: joy
Confidence: XX.XX%
```

---

### 📊 Dashboard

The dashboard displays:

* Dataset size
* Number of emotion classes
* Vocabulary size
* Test accuracy
* Emotion distribution
* Confusion matrix
* Dataset preview

---

### 🧠 Model Details

This section provides information about:

* NLP preprocessing
* TF-IDF
* Logistic Regression
* Train-test split
* Classification report
* Emotion labels

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/NLP-Emotion-Classification-Streamlit.git
```

Move into the project directory:

```bash
cd NLP-Emotion-Classification-Streamlit
```

---

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

---

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

---

### 4. Run the Streamlit application

```bash
python -m streamlit run app.py
```

The application will open in your browser.

Usually:

```text
http://localhost:8501
```

---

## 📦 Requirements

The main dependencies are:

```text
streamlit
pandas
numpy
scikit-learn
nltk
matplotlib
```

They are included in:

```text
requirements.txt
```

---

## 📄 Dataset Format

The application expects the dataset in semicolon-separated format:

```text
text;emotion
```

For example:

```text
I am very happy today;joy
I am feeling terrible;sadness
I am angry about this;anger
```

Place the dataset in the project directory as:

```text
train.txt
```

You can also upload the dataset through the Streamlit sidebar.

---

## 🎯 Example Predictions

### Happy

```text
I am extremely happy with my results today.
```

### Sad

```text
I feel lonely and sad today.
```

### Angry

```text
I am very angry about what happened.
```

### Excited

```text
I am so excited about my new project!
```

### Fear

```text
I am scared about what might happen.
```

The exact predicted class depends on the emotion categories available in the training dataset.

---

## 🔬 Skills Demonstrated

This project demonstrates practical knowledge of:

* Natural Language Processing
* Text preprocessing
* Stopword removal
* Feature engineering
* TF-IDF
* Supervised Machine Learning
* Logistic Regression
* Model evaluation
* Classification metrics
* Data visualization
* Streamlit
* Python
* Git & GitHub

---

## 📌 Future Improvements

Possible improvements include:

* Add stemming and lemmatization
* Experiment with Naive Bayes and SVM
* Add Word2Vec embeddings
* Add deep learning models
* Add LSTM/GRU-based emotion classification
* Experiment with Transformer models
* Improve prediction accuracy
* Deploy the application online
* Add multilingual emotion detection

---

## 👨‍💻 Author

**Abhishek Sontakke**

MCA Student | AI/ML | Data Science | NLP

---

## ⭐ Project Highlights

```text
NLP
  ↓
Text Preprocessing
  ↓
TF-IDF
  ↓
Logistic Regression
  ↓
Emotion Classification
  ↓
Streamlit
  ↓
Interactive Prediction Dashboard
```

If you found this project useful, consider giving the repository a ⭐.
