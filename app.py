import string
from pathlib import Path

import pandas as pd
import streamlit as st
import nltk
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

st.set_page_config(
    page_title="Text Emotion Classifier",
    page_icon="💭",
    layout="wide",
)

st.markdown(
    """<style>

@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
:root { --app-bg:#080d1a; --line:rgba(159,180,224,.14); --muted:#9ba9c3; --text:#f4f7ff; --mint:#6ee7d8; --violet:#a895ff; --blue:#7cc8ff; }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
h1,h2,h3,[data-testid="stMetricValue"] { font-family:'Space Grotesk','DM Sans',sans-serif !important; }
[data-testid="stAppViewContainer"] { background:radial-gradient(ellipse at 78% -10%,rgba(104,84,210,.18),transparent 36%),radial-gradient(ellipse at 1% 28%,rgba(45,176,191,.10),transparent 30%),var(--app-bg); color:var(--text); }
[data-testid="stHeader"] { background:rgba(8,13,26,.25); }
main .block-container { max-width:1320px; padding-top:2rem; padding-bottom:3.2rem; }
[data-testid="stSidebar"] { background:linear-gradient(180deg,#111a2b 0%,#0b1221 100%); border-right:1px solid var(--line); }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color:#b5c1d8; }
.hero-shell { position:relative; overflow:hidden; padding:clamp(24px,4vw,44px); margin:4px 0 28px; border:1px solid rgba(132,164,236,.23); border-radius:26px; background:radial-gradient(circle at 86% 12%,rgba(168,149,255,.18),transparent 27%),radial-gradient(circle at 7% 105%,rgba(110,231,216,.13),transparent 36%),linear-gradient(135deg,rgba(20,31,54,.96),rgba(10,17,32,.96)); box-shadow:0 24px 70px rgba(0,0,0,.22); }
.hero-shell:after { content:"✳"; position:absolute; right:4%; bottom:-50px; font-size:clamp(150px,20vw,280px); line-height:1; color:rgba(170,158,255,.055); pointer-events:none; transform:rotate(-15deg); }
.hero-kicker { display:inline-flex; align-items:center; gap:9px; color:#a8f3e9; font-size:.72rem; font-weight:700; letter-spacing:.14em; text-transform:uppercase; margin-bottom:18px; }
.hero-dot { width:7px; height:7px; border-radius:100%; background:var(--mint); box-shadow:0 0 14px rgba(110,231,216,.8); }
.hero-shell h1 { position:relative; z-index:1; color:#f6f7ff; font-size:clamp(2.3rem,5vw,4.3rem); line-height:1.04; letter-spacing:-.055em; margin:0 0 14px; font-weight:700; }
.hero-shell h1 span { background:linear-gradient(90deg,var(--mint),#9db7ff 75%,#c8a6ff); -webkit-background-clip:text; background-clip:text; color:transparent; }
.hero-subtitle { position:relative; z-index:1; max-width:760px; color:#b9c5dc; font-size:1.05rem; line-height:1.7; margin-bottom:22px; }
.hero-badges { position:relative; z-index:1; display:flex; flex-wrap:wrap; gap:9px; }
.hero-badges span { padding:7px 11px; border:1px solid rgba(162,184,233,.19); border-radius:999px; background:rgba(255,255,255,.045); color:#dce5fa; font-size:.78rem; }
.section-eyebrow { color:var(--mint); text-transform:uppercase; letter-spacing:.16em; font-size:.73rem; font-weight:700; margin:4px 0 5px; }
.section-description { color:var(--muted); margin:0 0 18px; font-size:.94rem; }
[data-testid="stTabs"] [data-baseweb="tab-list"] { gap:8px; padding:7px; border:1px solid var(--line); border-radius:16px; background:rgba(15,24,42,.78); }
[data-testid="stTabs"] button[data-baseweb="tab"] { border-radius:11px; padding:11px 17px; color:#aebbd2; font-weight:600; }
[data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] { color:#091521; background:linear-gradient(135deg,var(--mint),#9debdc); }
[data-testid="stTabs"] [data-baseweb="tab-highlight"] { background:transparent; }
[data-testid="stVerticalBlockBorderWrapper"] { border-color:var(--line)!important; border-radius:20px!important; background:linear-gradient(145deg,rgba(19,29,49,.87),rgba(12,19,34,.91)); box-shadow:0 12px 32px rgba(0,0,0,.10); }
[data-testid="stMetric"] { padding:17px 18px; border:1px solid rgba(150,176,227,.15); border-radius:16px; background:linear-gradient(145deg,rgba(24,38,62,.95),rgba(15,24,41,.94)); }
[data-testid="stMetricLabel"] { color:#adbad2!important; font-size:.82rem; }
[data-testid="stMetricValue"] { color:#f3f6ff!important; font-weight:700; }
[data-testid="stMetricDelta"] { font-size:.78rem; }
[data-testid="stSelectbox"] [data-baseweb="select"] > div { border-radius:12px; border-color:rgba(153,177,229,.20); background:rgba(10,17,31,.70); }
[data-testid="stTextArea"] textarea { border-radius:14px; border-color:rgba(153,177,229,.20); background:rgba(8,14,27,.80); color:#f3f6ff; line-height:1.6; }
[data-testid="stTextArea"] textarea:focus { border-color:var(--mint); box-shadow:0 0 0 1px var(--mint); }
[data-testid="stFileUploader"] section { border-radius:14px; border-color:rgba(153,177,229,.27); background:rgba(8,14,27,.55); }
.stButton > button { border-radius:12px; font-weight:700; min-height:44px; transition:transform .15s ease,filter .15s ease,border-color .15s ease; }
.stButton > button:hover { transform:translateY(-1px); filter:brightness(1.06); border-color:var(--mint); }
.stButton > button[kind="primary"] { color:#071520; background:linear-gradient(100deg,#6ee7d8,#96f0d9); border:0; box-shadow:0 7px 22px rgba(72,212,190,.15); }
[data-testid="stAlert"] { border-radius:13px; }
[data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:12px; overflow:hidden; }
.result-label { font-size:.72rem; text-transform:uppercase; letter-spacing:.15em; font-weight:700; color:var(--mint); margin-bottom:10px; }
.empty-state { min-height:265px; display:flex; flex-direction:column; align-items:flex-start; justify-content:center; padding:25px; border:1px dashed rgba(145,173,230,.24); border-radius:16px; background:radial-gradient(circle at 92% 5%,rgba(155,137,255,.10),transparent 35%),rgba(6,12,25,.37); }
.empty-state .empty-icon { font-size:2rem; margin-bottom:12px; }
.empty-state h3 { color:#f1f5ff; font-size:1.3rem; margin:0 0 7px; }
.empty-state p { color:#9eacc6; line-height:1.6; margin:0; }
.pipeline { display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); gap:10px; margin:16px 0 20px; }
.pipeline-step { padding:17px 12px; border:1px solid rgba(151,179,231,.17); border-radius:14px; text-align:center; background:linear-gradient(150deg,rgba(25,39,64,.9),rgba(13,22,39,.88)); }
.pipeline-step .step-number { display:inline-grid; place-items:center; width:29px; height:29px; border-radius:9px; background:rgba(110,231,216,.13); color:var(--mint); font-weight:700; font-size:.8rem; margin-bottom:9px; }
.pipeline-step strong { display:block; color:#e8eeff; font-size:.86rem; }
.pipeline-step small { display:block; color:#92a2bf; font-size:.72rem; margin-top:5px; }
@media (max-width:900px) { main .block-container { padding:1.1rem 1rem 2rem; } .pipeline { grid-template-columns:repeat(2,minmax(0,1fr)); } }
@media (max-width:600px) { .hero-shell { border-radius:19px; padding:23px 19px; } .hero-shell h1 { font-size:2.35rem; } .hero-subtitle { font-size:.95rem; } [data-testid="stTabs"] [data-baseweb="tab-list"] { gap:3px; padding:5px; } [data-testid="stTabs"] button[data-baseweb="tab"] { padding:9px 10px; font-size:.82rem; } .pipeline { grid-template-columns:1fr 1fr; } }

</style>""",
    unsafe_allow_html=True,
)

# ---------- NLP resources ----------
@st.cache_resource
def load_stopwords():
    """Load NLTK stopwords, with an offline-safe fallback for hosted apps."""
    try:
        return set(stopwords.words("english"))
    except LookupError:
        # Community hosting environments may not have the NLTK corpus cached yet.
        # Try to fetch it, but keep the app usable if the download is unavailable.
        try:
            if nltk.download("stopwords", quiet=True):
                return set(stopwords.words("english"))
        except Exception:
            pass

        st.warning(
            "NLTK stopwords could not be downloaded; using scikit-learn's "
            "built-in English stopword list instead."
        )
        return set(ENGLISH_STOP_WORDS)

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
    <section class="hero-shell">
      <div class="hero-kicker"><span class="hero-dot"></span> NATURAL LANGUAGE PROCESSING · EMOTION INTELLIGENCE</div>
      <h1>Text Emotion <span>Classifier</span></h1>
      <div class="hero-subtitle">
        Turn everyday language into useful emotion insights. Explore a sentence, compare
        class probabilities, and inspect how the TF-IDF + Logistic Regression model performs.
      </div>
      <div class="hero-badges">
        <span>✦ Multi-class emotion prediction</span>
        <span>⌘ TF-IDF features</span>
        <span>◈ Interactive model dashboard</span>
      </div>
    </section>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## ⚙️ Workspace")
    st.markdown("Personalize the dataset used by the classifier.")
    uploaded = st.file_uploader("Upload a dataset", type=["txt", "csv"])
    st.caption("Expected format: one example per line, text;emotion")
    st.divider()
    st.markdown("### 🧠 Model stack")
    st.markdown(
        "- Text cleaning + stopword removal\n"
        "- TF-IDF vectorization\n"
        "- Logistic Regression classifier"
    )
    st.info(
        "Tip: for the built-in dataset, leave the uploader empty. "
        "Upload a compatible file to explore a different emotion dataset."
    )

# Dataset loading
default_path = Path(__file__).resolve().parent / "train.txt"

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

# ---------- Workspace tabs ----------
tab1, tab2, tab3 = st.tabs(["✦  Predict", "▥  Dashboard", "⌘  Model details"])

with tab1:
    st.markdown('<div class="section-eyebrow">01 / Live analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-description">Enter a sentence to see what emotion the model detects and how confident it is.</div>', unsafe_allow_html=True)

    input_col, result_col = st.columns([1.06, 0.94], gap="large")

    with input_col:
        with st.container(border=True):
            st.markdown("### ✍️ Your text")
            examples = [
                "I am feeling really happy today!",
                "I feel terrible and hopeless.",
                "I am angry about what happened.",
                "I am excited about my new project.",
            ]
            selected = st.selectbox("Start with an example", ["Custom text"] + examples)
            default_text = "" if selected == "Custom text" else selected
            user_text = st.text_area(
                "Sentence to analyze",
                value=default_text,
                height=175,
                placeholder="Type something like: I am proud of what I achieved today...",
            )
            st.caption("The model removes punctuation, numbers, non-ASCII characters and stopwords before prediction.")
            analyze_clicked = st.button("✦  Analyze emotion", type="primary", use_container_width=True)

        if analyze_clicked:
            if not user_text.strip():
                st.session_state.pop("emotion_prediction_result", None)
                st.warning("Enter a sentence first, or choose one of the examples.")
            else:
                cleaned = clean_text(user_text)
                if not cleaned.strip():
                    st.session_state.pop("emotion_prediction_result", None)
                    st.warning("After text cleaning, no words were left to analyze. Try a longer sentence.")
                else:
                    features = vectorizer.transform([cleaned])
                    probabilities = model.predict_proba(features)[0]
                    best_index = int(probabilities.argmax())
                    prediction = str(model.classes_[best_index])
                    st.session_state["emotion_prediction_result"] = {
                        "source_name": source_name,
                        "text": user_text,
                        "cleaned": cleaned,
                        "prediction": prediction,
                        "confidence": float(probabilities[best_index]),
                        "word_count": len(cleaned.split()),
                        "probabilities": {
                            str(label): float(probability)
                            for label, probability in zip(model.classes_, probabilities)
                        },
                    }

    saved_result = st.session_state.get("emotion_prediction_result")
    if saved_result and saved_result.get("source_name") != source_name:
        saved_result = None

    with result_col:
        with st.container(border=True):
            if saved_result:
                st.markdown('<div class="result-label">✦ Model result</div>', unsafe_allow_html=True)
                st.markdown("### Detected emotion")
                st.success(f"**{saved_result['prediction'].title()}**")
                stat1, stat2 = st.columns(2)
                stat1.metric("Model confidence", f"{saved_result['confidence'] * 100:.1f}%")
                stat2.metric("Words analyzed", saved_result["word_count"])
                st.markdown("**Cleaned text**")
                st.code(saved_result["cleaned"], language="text")
                st.caption("Confidence is the model's highest class probability, not a guarantee of correctness.")
            else:
                st.markdown(
                    """
                    <div class="empty-state">
                      <div class="empty-icon">🪄</div>
                      <h3>Your insight starts here</h3>
                      <p>Submit a sentence to reveal the predicted emotion, confidence score, cleaned text and class-by-class probability breakdown.</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    if saved_result:
        st.markdown("### Emotion probability breakdown")
        st.markdown('<div class="section-description">Compare the model score for each emotion. The highlighted bar is the top prediction.</div>', unsafe_allow_html=True)
        prob_df = pd.DataFrame(
            {
                "Emotion": list(saved_result["probabilities"].keys()),
                "Probability": [value * 100 for value in saved_result["probabilities"].values()],
            }
        ).sort_values("Probability", ascending=True)

        fig, ax = plt.subplots(figsize=(10, max(3.1, len(prob_df) * 0.48)), facecolor="#0e1728")
        ax.set_facecolor("#0e1728")
        chart_colors = [
            "#6ee7d8" if emotion == saved_result["prediction"] else "#7188b9"
            for emotion in prob_df["Emotion"]
        ]
        bars = ax.barh(prob_df["Emotion"], prob_df["Probability"], color=chart_colors, height=0.58)
        max_value = max(float(prob_df["Probability"].max()), 1.0)
        ax.set_xlim(0, min(125, max_value * 1.25 + 4))
        ax.set_xlabel("Model probability (%)", color="#9ba9c3", labelpad=10)
        ax.tick_params(axis="x", colors="#91a1bf", labelsize=9)
        ax.tick_params(axis="y", colors="#e1e9fb", labelsize=10, length=0)
        ax.grid(axis="x", color="#283650", alpha=0.75, linewidth=0.8)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_visible(False)
        for bar, value in zip(bars, prob_df["Probability"]):
            ax.text(
                min(value + 1.2, 119),
                bar.get_y() + bar.get_height() / 2,
                f"{value:.1f}%",
                va="center",
                ha="left",
                color="#f0f4ff",
                fontsize=9,
                fontweight="bold",
            )
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

with tab2:
    st.markdown('<div class="section-eyebrow">02 / Model observability</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-description">A snapshot of the data and how the classifier performs on its held-out test split.</div>', unsafe_allow_html=True)

    a, b, c, d = st.columns(4, gap="medium")
    a.metric("Dataset examples", f"{len(data):,}")
    b.metric("Emotion classes", len(labels))
    c.metric("TF-IDF vocabulary", f"{len(vectorizer.vocabulary_):,}")
    d.metric("Test accuracy", f"{accuracy * 100:.2f}%")

    st.divider()
    distribution_col, matrix_col = st.columns([0.95, 1.05], gap="large")

    with distribution_col:
        with st.container(border=True):
            st.markdown("### Emotion distribution")
            counts = data["emotions"].value_counts().sort_values(ascending=True)
            fig, ax = plt.subplots(figsize=(7.5, max(3.4, len(counts) * 0.44)), facecolor="#0e1728")
            ax.set_facecolor("#0e1728")
            bars = ax.barh(counts.index.astype(str), counts.values, color="#7cc8ff", height=0.62)
            ax.set_xlabel("Number of examples", color="#9ba9c3", labelpad=9)
            ax.tick_params(axis="x", colors="#91a1bf", labelsize=9)
            ax.tick_params(axis="y", colors="#e1e9fb", labelsize=9, length=0)
            ax.grid(axis="x", color="#283650", alpha=0.75, linewidth=0.8)
            ax.set_axisbelow(True)
            for spine in ax.spines.values():
                spine.set_visible(False)
            for bar in bars:
                width = bar.get_width()
                ax.text(width + max(counts.max() * 0.012, 0.2), bar.get_y() + bar.get_height()/2,
                        f"{width:,.0f}", va="center", color="#e8efff", fontsize=9)
            ax.set_xlim(0, counts.max() * 1.17)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    with matrix_col:
        with st.container(border=True):
            st.markdown("### Confusion matrix")
            fig, ax = plt.subplots(figsize=(7.5, 5.2), facecolor="#0e1728")
            ax.set_facecolor("#0e1728")
            image = ax.imshow(cm, cmap="viridis", aspect="auto")
            ax.set_xticks(range(len(labels)))
            ax.set_yticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=35, ha="right", color="#c9d5ed")
            ax.set_yticklabels(labels, color="#c9d5ed")
            ax.set_xlabel("Predicted label", color="#9ba9c3", labelpad=10)
            ax.set_ylabel("Actual label", color="#9ba9c3", labelpad=10)
            ax.tick_params(axis="both", length=0)
            threshold = float(cm.max()) / 2 if cm.size else 0
            for i in range(len(labels)):
                for j in range(len(labels)):
                    ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                            color="#f7f9ff" if cm[i, j] > threshold else "#dbe7ff",
                            fontsize=9, fontweight="bold")
            colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
            colorbar.ax.tick_params(colors="#9ba9c3", labelsize=8)
            colorbar.outline.set_visible(False)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    with st.container(border=True):
        st.markdown("### Dataset preview")
        st.caption("A small sample of the text and its cleaned version.")
        st.dataframe(data[["text", "emotions", "cleaned_text"]].head(20), use_container_width=True, hide_index=True)

with tab3:
    st.markdown('<div class="section-eyebrow">03 / Under the hood</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-description">A quick tour of the text-processing and machine-learning pipeline behind every prediction.</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="pipeline">
          <div class="pipeline-step"><span class="step-number">01</span><strong>Raw text</strong><small>Input sentence</small></div>
          <div class="pipeline-step"><span class="step-number">02</span><strong>Preprocess</strong><small>Clean + tokenize</small></div>
          <div class="pipeline-step"><span class="step-number">03</span><strong>TF-IDF</strong><small>Text to features</small></div>
          <div class="pipeline-step"><span class="step-number">04</span><strong>Classifier</strong><small>Logistic Regression</small></div>
          <div class="pipeline-step"><span class="step-number">05</span><strong>Prediction</strong><small>Emotion + probabilities</small></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    details_col, score_col = st.columns([1.25, 0.75], gap="large")
    with details_col:
        with st.container(border=True):
            st.markdown("### Preprocessing and training")
            st.markdown(
                "- Convert text to lowercase\n"
                "- Remove punctuation and digits\n"
                "- Remove non-ASCII characters\n"
                "- Remove English stopwords\n"
                "- Split examples into train and test sets (80/20)\n"
                "- Fit TF-IDF on the training text only\n"
                "- Train Logistic Regression with max_iter=1000"
            )
    with score_col:
        with st.container(border=True):
            st.markdown("### Current evaluation")
            st.metric("Held-out test accuracy", f"{accuracy * 100:.2f}%")
            st.metric("Unique text features", f"{len(vectorizer.vocabulary_):,}")
            st.metric("Labels supported", f"{len(labels)}")

    with st.container(border=True):
        st.markdown("### Classification report")
        st.caption("Precision, recall and F1-score for each emotion label on the test split.")
        report_df = pd.DataFrame(report).T
        st.dataframe(report_df.round(3), use_container_width=True)

    st.markdown("### Emotion labels")
    st.write(", ".join(labels))

st.divider()
st.caption(f"Dataset: {source_name} · Text Emotion Classifier · TF-IDF + Logistic Regression")
