import string
from pathlib import Path

import pandas as pd
import streamlit as st
import nltk
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

st.set_page_config(
    page_title="Sentiment & Emotion Analyzer",
    page_icon="💭",
    layout="wide",
)

# ---------- NLP resources ----------
@st.cache_resource
def load_stopwords():
    try:
        words = stopwords.words("english")
    except LookupError:
        nltk.download("stopwords", quiet=True)
        words = stopwords.words("english")
    return set(words)

stop_words = load_stopwords()

# ---------- Same preprocessing idea as the notebook ----------
def remove_punc(txt):
    return txt.translate(str.maketrans("", "", string.punctuation))

def remove_number(txt):
    return "".join(ch for ch in txt if not ch.isdigit())

def remove_emojis(txt):
    return "".join(ch for ch in txt if ch.isascii())

def clean_text(txt):
    txt = str(txt).lower()
    txt = remove_punc(txt)
    txt = remove_number(txt)
    txt = remove_emojis(txt)

    words = txt.split()
    cleaned = [word for word in words if word not in stop_words]
    return " ".join(cleaned)

# ---------- Model ----------
@st.cache_data(show_spinner=False)
def prepare_data(df):
    required = {"text", "emotions"}
    if not required.issubset(df.columns):
        raise ValueError("Dataset must contain columns: text and emotions")

    data = df[["text", "emotions"]].copy()
    data["text"] = data["text"].fillna("").astype(str)
    data["emotions"] = data["emotions"].astype(str)
    data = data[data["text"].str.strip() != ""].reset_index(drop=True)

    data["cleaned_text"] = data["text"].apply(clean_text)
    data = data[data["cleaned_text"].str.strip() != ""].reset_index(drop=True)
    return data

@st.cache_resource(show_spinner=True)
def train_model(df_for_training):
    # Notebook uses test_size=0.20 and random_state=42.
    X_train, X_test, y_train, y_test = train_test_split(
        df_for_training["cleaned_text"],
        df_for_training["emotions"],
        test_size=0.20,
        random_state=42,
        stratify=df_for_training["emotions"] if df_for_training["emotions"].value_counts().min() >= 2 else None,
    )

    vectorizer = TfidfVectorizer()
    x_train = vectorizer.fit_transform(X_train)
    x_test = vectorizer.transform(X_test)

    # Notebook's final model: LogisticRegression(max_iter=1000)
    model = LogisticRegression(max_iter=1000)
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    labels = list(model.classes_)
    cm = confusion_matrix(y_test, predictions, labels=labels)
    report = classification_report(
        y_test, predictions, labels=labels, output_dict=True, zero_division=0
    )

    return model, vectorizer, accuracy, cm, report, labels

# ---------- UI ----------
st.markdown(
    """
    <div style="text-align:center">
        <h1>💭 Sentiment & Emotion Analyzer</h1>
        <p style="font-size:18px;">
            NLP application using TF-IDF + Logistic Regression
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("⚙️ Settings")
    uploaded = st.file_uploader("Upload train.txt", type=["txt", "csv"])
    st.caption("Expected format: text;emotion")
    st.divider()
    st.info(
        "The model follows the preprocessing and final Logistic Regression "
        "approach used in your NLP notebook."
    )

# Dataset loading
default_path = Path("train.txt")

try:
    if uploaded is not None:
        raw = uploaded.getvalue().decode("utf-8", errors="ignore")
        from io import StringIO
        df = pd.read_csv(StringIO(raw), sep=";", header=None, names=["text", "emotions"])
        source_name = uploaded.name
    elif default_path.exists():
        df = pd.read_csv(default_path, sep=";", header=None, names=["text", "emotions"])
        source_name = "train.txt"
    else:
        st.warning("No train.txt found. Upload your dataset from the sidebar.")
        st.code(
            "your sentence;happy\n"
            "this is terrible;angry\n"
            "i feel great today;joy",
            language="text",
        )
        st.stop()

    data = prepare_data(df)
    model, vectorizer, accuracy, cm, report, labels = train_model(data)

except Exception as e:
    st.error(f"Could not load/train the model: {e}")
    st.stop()

# ---------- Dashboard ----------
tab1, tab2, tab3 = st.tabs(["🔮 Predict", "📊 Dashboard", "🧠 Model Details"])

with tab1:
    st.subheader("Analyze a sentence")

    examples = [
        "I am feeling really happy today!",
        "I feel terrible and hopeless.",
        "I am angry about what happened.",
        "I am excited about my new project.",
    ]

    selected = st.selectbox("Try an example", ["Custom text"] + examples)
    default_text = "" if selected == "Custom text" else selected

    text = st.text_area(
        "Enter your text",
        value=default_text,
        height=150,
        placeholder="Type a sentence here...",
    )

    if st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True):
        if not text.strip():
            st.warning("Please enter some text.")
        else:
            cleaned = clean_text(text)
            features = vectorizer.transform([cleaned])
            prediction = model.predict(features)[0]

            result = prediction
            probabilities = model.predict_proba(features)[0]
            best_index = probabilities.argmax()
            confidence = probabilities[best_index]

            st.success(f"Predicted emotion: **{result}**")

            c1, c2, c3 = st.columns(3)
            c1.metric("Prediction", str(result))
            c2.metric("Confidence", f"{confidence * 100:.2f}%")
            c3.metric("Cleaned words", len(cleaned.split()))

            st.write("**Cleaned text:**")
            st.code(cleaned if cleaned else "(empty after preprocessing)")

            st.write("**Emotion probabilities**")
            prob_df = pd.DataFrame(
                {"Emotion": model.classes_, "Probability": probabilities}
            ).sort_values("Probability", ascending=False)
            prob_df["Probability"] = prob_df["Probability"] * 100
            st.bar_chart(prob_df.set_index("Emotion")["Probability"])

with tab2:
    st.subheader("Dataset & model performance")

    a, b, c, d = st.columns(4)
    a.metric("Dataset rows", f"{len(data):,}")
    b.metric("Emotion classes", len(labels))
    c.metric("Vocabulary size", f"{len(vectorizer.vocabulary_):,}")
    d.metric("Test accuracy", f"{accuracy * 100:.2f}%")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Emotion distribution")
        counts = data["emotions"].value_counts()
        st.bar_chart(counts)

    with col2:
        st.markdown("### Confusion matrix")
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.imshow(cm)
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, cm[i, j], ha="center", va="center")
        fig.tight_layout()
        st.pyplot(fig)

    st.markdown("### Dataset preview")
    st.dataframe(data[["text", "emotions", "cleaned_text"]].head(20), use_container_width=True)

with tab3:
    st.subheader("How the application works")

    st.markdown(
        """
        **Pipeline**

        `train.txt` → lowercase → remove punctuation → remove numbers →
        remove non-ASCII characters → remove English stopwords →
        train/test split → TF-IDF → Logistic Regression → prediction

        **Model used**

        - TF-IDF Vectorizer
        - Logistic Regression
        - `max_iter=1000`
        - 80/20 train-test split
        - `random_state=42`
        """
    )

    st.metric("Notebook Logistic Regression accuracy", f"{accuracy * 100:.2f}%")

    report_df = pd.DataFrame(report).T
    st.markdown("### Classification report")
    st.dataframe(report_df.round(3), use_container_width=True)

    st.markdown("### Emotion labels")
    st.write(", ".join(labels))

st.divider()
st.caption(f"Dataset: {source_name} • NLP Project • Streamlit")
