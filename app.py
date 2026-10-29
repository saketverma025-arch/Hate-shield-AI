
import html
import re
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "twitter_data.csv"
MODEL_PATH = BASE_DIR / "hate_speech_model.joblib"

LABEL_NAMES = {
    0: "Hate Speech",
    1: "Offensive Language",
    2: "Neither",
}

LABEL_DESCRIPTIONS = {
    0: "The model detected language associated with hate speech.",
    1: "The model detected language associated with offensive content.",
    2: "The model did not classify this text as hate speech or offensive language.",
}

LABEL_ICONS = {0: "🛡️", 1: "⚠️", 2: "✓"}
RESULT_CLASSES = {0: "hate", 1: "offensive", 2: "safe"}


def clean_tweet(text):
    text = html.unescape(str(text))
    text = re.sub(r"https?://\S+|www\.\S+", " URL ", text)
    text = re.sub(r"@\w+", " USER ", text)
    return re.sub(r"\s+", " ", text).strip()


def build_model():
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            max_features=50000,
            sublinear_tf=True,
            min_df=2,
        )),
        ("classifier", LogisticRegression(
            max_iter=1500,
            class_weight="balanced",
            random_state=42,
        )),
    ])


@st.cache_resource(show_spinner=False)
def load_or_train_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            "twitter_data.csv is missing. Add it beside app.py."
        )

    df = pd.read_csv(DATASET_PATH)
    df.columns = df.columns.astype(str).str.strip()

    if not {"tweet", "class"}.issubset(df.columns):
        raise ValueError("CSV needs columns named tweet and class.")

    df = df[["tweet", "class"]].dropna().copy()
    df["tweet"] = df["tweet"].map(clean_tweet)
    df["class"] = pd.to_numeric(df["class"], errors="coerce")
    df = df.dropna(subset=["class"])
    df = df[df["class"] % 1 == 0].copy()
    df["class"] = df["class"].astype(int)
    df = df[df["class"].isin([0, 1, 2])]
    df = df[df["tweet"].str.len() > 0]
    df = df.drop_duplicates(subset=["tweet", "class"])

    if df.empty or df["class"].nunique() < 2:
        raise ValueError("Not enough valid classes to train the model.")

    model = build_model()
    model.fit(df["tweet"], df["class"])
    joblib.dump(model, MODEL_PATH)
    return model


def predict_text(model, text):
    cleaned = clean_tweet(text)
    prediction = int(model.predict([cleaned])[0])
    probabilities = model.predict_proba([cleaned])[0]
    classes = model.named_steps["classifier"].classes_
    scores = {
        int(label): float(score)
        for label, score in zip(classes, probabilities)
    }
    return prediction, scores


st.set_page_config(
    page_title="Hate-Shield AI",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600;700&display=swap');

:root {
    --bg: #020b0e;
    --panel: #06191d;
    --teal: #20f5c5;
    --muted: #9db9b8;
}

.stApp {
    background:
        radial-gradient(ellipse at 50% 5%, #07302e 0%, transparent 48%),
        linear-gradient(150deg, #02090d, #031517 65%, #02090d);
    color: #effffb;
    font-family: 'Inter', sans-serif;
}
[data-testid="stHeader"] { background: transparent; }
.block-container {
    max-width: 850px;
    padding: 1.4rem 1rem 3rem;
}
h1, h2, h3 {
    font-family: 'Space Grotesk', sans-serif !important;
    color: #f2fffc !important;
}
.brand {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 0 22px;
    border-bottom: 1px solid #16413e;
}
.brand-icon {
    border: 1px solid #20f5c5;
    border-radius: 12px;
    width: 43px;
    height: 43px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #20f5c5;
    font-size: 25px;
    background: #062522;
}
.brand-name {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 1.12rem;
    letter-spacing: .02em;
}
.brand-name span { color: #20f5c5; }
.brand-sub { font-size: .75rem; color: #8daaa8; }
.status-pill {
    margin-left: auto;
    border: 1px solid #176b5c;
    border-radius: 30px;
    padding: 7px 10px;
    color: #5df8d3;
    font-size: .65rem;
    white-space: nowrap;
}
.hero {
    text-align: center;
    padding: 30px 8px 20px;
}
.hero-shield {
    margin: 0 auto 13px;
    width: 104px;
    height: 104px;
    border-radius: 32px;
    border: 2px solid #20f5c5;
    background: radial-gradient(circle, #0b4b42, #041a1d 72%);
    box-shadow: 0 0 35px #00eac333, inset 0 0 22px #00eac322;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #20f5c5;
    font-size: 54px;
}
.hero h1 {
    font-size: clamp(2.2rem, 8vw, 3.5rem) !important;
    line-height: 1.08;
    letter-spacing: -.05em;
    margin: 8px 0 !important;
}
.hero h1 span { color: #20f5c5; }
.tagline {
    color: #20f5c5;
    font-weight: 700;
    font-size: 1rem;
}
.hero-copy {
    max-width: 510px;
    margin: 10px auto 0;
    color: #a9c4c0;
    font-size: .88rem;
    line-height: 1.65;
}
.section-heading {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 1.12rem;
    margin: 18px 0 12px;
}
.feature {
    min-height: 142px;
    background: linear-gradient(145deg, #082a2a, #061619);
    border: 1px solid #17635a;
    border-radius: 16px;
    padding: 16px 13px;
    margin: 4px 0;
}
.feature-icon {
    font-size: 1.6rem;
    margin-bottom: 8px;
}
.feature-title {
    font-weight: 700;
    color: #eafffa;
    font-size: .9rem;
    margin-bottom: 7px;
}
.feature-copy {
    color: #9dbbb5;
    font-size: .76rem;
    line-height: 1.5;
}
.panel {
    background: linear-gradient(145deg, #082124, #041316);
    border: 1px solid #17635a;
    border-radius: 18px;
    padding: 20px;
    margin-top: 12px;
}
.eyebrow {
    color: #20f5c5;
    font-size: .72rem;
    font-weight: 700;
    letter-spacing: .14em;
    text-transform: uppercase;
}
.stTextArea textarea {
    background: #020e11 !important;
    color: #eafffa !important;
    border: 1px solid #21665d !important;
    border-radius: 12px !important;
    font-size: 1rem !important;
}
.stTextArea textarea:focus {
    border-color: #20f5c5 !important;
    box-shadow: 0 0 0 1px #20f5c5 !important;
}
div.stButton > button {
    width: 100%;
    min-height: 3rem;
    border-radius: 12px;
    border: 1px solid #20f5c5;
    background: linear-gradient(90deg, #20f5c5, #10c9a4);
    color: #02211c;
    font-weight: 800;
    transition: .2s;
}
div.stButton > button:hover {
    border-color: #8bffe5;
    color: #001b17;
    filter: brightness(1.08);
}
.result {
    padding: 22px;
    margin-top: 15px;
    border-radius: 16px;
    background: #061b1d;
    border: 1px solid #27776a;
}
.result.hate { border-color: #ff6078; }
.result.offensive { border-color: #ffbd55; }
.result.safe { border-color: #20f5c5; }
.result-label {
    color: #9db9b8;
    text-transform: uppercase;
    letter-spacing: .13em;
    font-size: .7rem;
    font-weight: 700;
}
.result-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.7rem;
    font-weight: 700;
    margin: 8px 0;
}
.hate .result-title { color: #ff7187; }
.offensive .result-title { color: #ffca70; }
.safe .result-title { color: #20f5c5; }
.result-copy {
    color: #b3cbc6;
    font-size: .9rem;
    line-height: 1.6;
}
div[data-testid="stMetric"] {
    background: #061b1d;
    border: 1px solid #164b44;
    border-radius: 12px;
    padding: 12px;
}
div[data-testid="stMetricLabel"] { color: #9db9b8; }
div[data-testid="stMetricValue"] { color: #20f5c5; }
hr { border-color: #16413e; }
div[data-testid="stExpander"] {
    border-color: #164b44;
    border-radius: 12px;
}
.footer {
    text-align: center;
    color: #789794;
    font-size: .74rem;
    line-height: 1.7;
    padding: 20px 0 5px;
}
div[data-testid="stCaptionContainer"] { color: #9db9b8; }
@media (max-width: 600px) {
    .status-pill { font-size: .57rem; padding: 6px; }
    .panel { padding: 14px; }
    .feature { min-height: 130px; padding: 12px 9px; }
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="brand">
  <div class="brand-icon">⛨</div>
  <div>
    <div class="brand-name">HATE-SHIELD <span>AI</span></div>
    <div class="brand-sub">Responsible text analysis</div>
  </div>
  <div class="status-pill">◉ AI TEXT DETECTOR</div>
</div>

<div class="hero">
  <div class="hero-shield">⛨</div>
  <h1>Hate-Shield <span>AI</span></h1>
  <div class="tagline">Safer Internet. Kinder Communities.</div>
  <div class="hero-copy">
    Detect hate speech, offensive language and harmful content
    using machine learning. Help promote respectful online conversations.
  </div>
</div>
""", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""
    <div class="feature">
      <div class="feature-icon">🛡️</div>
      <div class="feature-title">Detect Hate Speech</div>
      <div class="feature-copy">Identify language associated with hate and discrimination.</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="feature">
      <div class="feature-icon">▣</div>
      <div class="feature-title">Identify Offensive Language</div>
      <div class="feature-copy">Flag language the model associates with offensive content.</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="feature">
      <div class="feature-icon">✓</div>
      <div class="feature-title">Keep the Internet Safer</div>
      <div class="feature-copy">Support respectful and positive communication.</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="section-heading">01 / Analyse a message</div>',
            unsafe_allow_html=True)

with st.container(border=True):
    st.markdown('<div class="eyebrow">Message input</div>',
                unsafe_allow_html=True)
    user_text = st.text_area(
        "Message to analyse",
        placeholder="Type or paste a message here...",
        height=140,
        max_chars=5000,
        label_visibility="collapsed",
        key="message_input",
    )
    st.caption("Enter a message to classify. Avoid personal information.")

    left, right = st.columns([3, 1])
    with left:
        detect_clicked = st.button(
            "🛡️  Analyse message", use_container_width=True
        )
    with right:
        clear_clicked = st.button("Clear", use_container_width=True)

if clear_clicked:
    st.session_state["last_result"] = None
    st.session_state["message_input"] = ""
    st.rerun()

if detect_clicked:
    if not user_text.strip():
        st.warning("Please enter a message before analysing it.")
    else:
        try:
            with st.spinner("Loading model and analysing message..."):
                model = load_or_train_model()
                prediction, scores = predict_text(model, user_text)

            st.session_state["last_result"] = {
                "text": user_text,
                "prediction": prediction,
                "scores": scores,
            }
        except Exception as exc:
            st.error(f"Could not run the detector: {exc}")

result = st.session_state.get("last_result")

if result:
    pred = result["prediction"]
    title = LABEL_NAMES.get(pred, "Unknown")
    icon = LABEL_ICONS.get(pred, "🔎")
    css_class = RESULT_CLASSES.get(pred, "safe")
    confidence = result["scores"].get(pred, 0.0)

    st.markdown('<div class="section-heading">02 / Detection result</div>',
                unsafe_allow_html=True)

    st.markdown(
        f'<div class="result {css_class}">'
        f'<div class="result-label">Classification result</div>'
        f'<div class="result-title">{icon} {title}</div>'
        f'<div class="result-copy">{LABEL_DESCRIPTIONS.get(pred, "")}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    a, b = st.columns(2)
    with a:
        st.metric("Model score", f"{confidence * 100:.1f}%")
    with b:
        st.metric("Characters analysed", f"{len(result['text']):,}")

    with st.expander("View all class scores"):
        for label_id in [0, 1, 2]:
            score = result["scores"].get(label_id, 0.0)
            st.write(f"**{LABEL_NAMES[label_id]}** · {score * 100:.1f}%")
            st.progress(min(max(score, 0.0), 1.0))

    st.caption(
        "The score is the model's estimated class probability, not a "
        "guarantee of correctness. Context, sarcasm, slang and dataset "
        "bias can affect predictions. Human review is important."
    )

with st.expander("About Hate-Shield AI"):
    st.write(
        "Hate-Shield AI is a Class XII Artificial Intelligence capstone "
        "project. It uses TF-IDF text features and Logistic Regression "
        "to classify text into three dataset categories: Hate Speech, "
        "Offensive Language, and Neither."
    )

st.markdown("""
<div class="footer">
  🛡️ HATE-SHIELD AI · CLASS XII AI CAPSTONE<br>
  Educational prototype · Predictions can be incorrect.
  Do not use this model alone to penalise people.
</div>
""", unsafe_allow_html=True)
