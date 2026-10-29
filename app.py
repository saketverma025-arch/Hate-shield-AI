import html
import re
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# -------------------------------------------------
# HATE-SHIELD AI | Streamlit web application
# Dataset columns: tweet, class
# Labels: 0 = Hate Speech, 1 = Offensive Language, 2 = Neither
# -------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "twitter_data.csv"
MODEL_PATH = BASE_DIR / "hate_speech_model.joblib"

LABEL_NAMES = {
    0: "Hate Speech",
    1: "Offensive Language",
    2: "Neither",
}
LABEL_DESCRIPTIONS = {
    0: "The model detected language it associates with hate speech.",
    1: "The model detected language it associates with offensive content.",
    2: "The model did not classify this text as hate speech or offensive language.",
}
LABEL_ICONS = {0: "🛡️", 1: "⚠️", 2: "✅"}


def clean_tweet(text):
    text = html.unescape(str(text))
    text = re.sub(r"https?://\S+|www\.\S+", " URL ", text)
    text = re.sub(r"@\w+", " USER ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


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
    """Load the saved pipeline, or train and save one if it is missing."""
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH), "loaded"

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            "twitter_data.csv was not found. Put it in the same folder as app.py."
        )

    df = pd.read_csv(DATASET_PATH)
    df.columns = df.columns.astype(str).str.strip()

    missing = {"tweet", "class"} - set(df.columns)
    if missing:
        raise ValueError(
            "The CSV must contain columns named 'tweet' and 'class'. "
            f"Missing: {', '.join(sorted(missing))}"
        )

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
        raise ValueError("The dataset does not contain enough valid classes to train.")

    model = build_model()
    model.fit(df["tweet"], df["class"])
    joblib.dump(model, MODEL_PATH)
    return model, "trained"


def predict_text(model, text):
    cleaned = clean_tweet(text)
    prediction = int(model.predict([cleaned])[0])
    probabilities = model.predict_proba([cleaned])[0]
    model_classes = model.named_steps["classifier"].classes_
    scores = {int(label): float(score) for label, score in zip(model_classes, probabilities)}
    confidence = scores.get(prediction, 0.0)
    return prediction, confidence, scores


# Page setup
st.set_page_config(
    page_title="Hate-Shield AI",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
    .stApp {
        background:
            radial-gradient(ellipse at 10% 0%, rgba(44, 100, 180, .17), transparent 35%),
            radial-gradient(ellipse at 100% 25%, rgba(42, 177, 157, .10), transparent 30%),
            #0b1020;
        color: #eef3ff;
        font-family: 'DM Sans', sans-serif;
    }
    [data-testid="stHeader"] { background: rgba(11,16,32,0); }
    .block-container { max-width: 850px; padding-top: 2rem; padding-bottom: 3rem; }
    h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; color: #f5f7ff !important; }
    .brand-row { display:flex; align-items:center; gap:12px; margin-bottom: 1rem; }
    .brand-mark {
        width:52px; height:52px; border-radius:16px; display:flex; align-items:center;
        justify-content:center; font-size:26px;
        background:linear-gradient(135deg,#173d72,#147d83);
        border:1px solid rgba(140,202,255,.35);
        box-shadow:0 8px 30px rgba(20,125,131,.2);
    }
    .brand-name { font-family:'Space Grotesk',sans-serif; font-size:1.45rem; font-weight:700; letter-spacing:.04em; }
    .brand-sub { color:#9aabc9; font-size:.88rem; margin-top:2px; }
    .hero {
        padding: 1.5rem 1.55rem; border-radius:24px;
        border:1px solid rgba(151,177,224,.18);
        background:linear-gradient(135deg,rgba(23,39,72,.94),rgba(16,32,53,.88));
        margin: 1rem 0 1.2rem;
    }
    .eyebrow { color:#71d9d0; text-transform:uppercase; letter-spacing:.16em; font-size:.72rem; font-weight:700; }
    .hero-title { font-family:'Space Grotesk',sans-serif; font-size:2rem; line-height:1.15; font-weight:700; margin:.55rem 0; }
    .hero-copy { color:#b9c7df; line-height:1.6; margin-bottom:0; }
    .section-label { color:#dce6fb; font-weight:700; font-size:1.05rem; margin:.8rem 0 .5rem; }
    .hint { color:#93a4c4; font-size:.84rem; }
    .result-card {
        padding:1.3rem 1.4rem; border-radius:20px; margin-top:1rem;
        background:rgba(18,31,54,.92); border:1px solid rgba(151,177,224,.2);
    }
    .result-kicker { color:#9aabc9; text-transform:uppercase; letter-spacing:.12em; font-size:.72rem; font-weight:700; }
    .result-title { font-family:'Space Grotesk',sans-serif; font-size:1.7rem; font-weight:700; margin:.45rem 0; }
    .result-copy { color:#bdcbe1; line-height:1.55; }
    .small-note { color:#91a1bd; font-size:.78rem; line-height:1.5; }
    div.stButton > button {
        border:0; border-radius:12px; min-height:3rem; font-weight:700;
        background:linear-gradient(90deg,#4e8cff,#20b8ad); color:white;
        width:100%;
    }
    div.stButton > button:hover { border:0; color:white; filter:brightness(1.08); }
    .stTextArea textarea {
        background:#101a2d !important; color:#edf3ff !important;
        border:1px solid #30415f !important; border-radius:14px !important;
    }
    [data-testid="stMetric"] {
        background:rgba(18,31,54,.72); padding:12px 15px; border-radius:14px;
        border:1px solid rgba(151,177,224,.14);
    }
    hr { border-color:rgba(151,177,224,.16); }
    footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="brand-row">
  <div class="brand-mark">🛡️</div>
  <div>
    <div class="brand-name">HATE-SHIELD AI</div>
    <div class="brand-sub">Responsible text analysis · Class XII AI Capstone</div>
  </div>
</div>
<div class="hero">
  <div class="eyebrow">AI-powered text classification</div>
  <div class="hero-title">Make online conversations safer.</div>
  <p class="hero-copy">Enter a message below to see how the machine-learning model classifies its language. This is a learning prototype, not a final judge of a person's intent.</p>
</div>
""", unsafe_allow_html=True)

with st.expander("How it works"):
    st.write(
        "Hate-Shield AI uses a TF-IDF text representation and a Logistic Regression "
        "classifier trained on the supplied CSV dataset. It returns one of three "
        "dataset labels: Hate Speech, Offensive Language, or Neither."
    )

st.markdown('<div class="section-label">01 / Enter a message</div>', unsafe_allow_html=True)
user_text = st.text_area(
    "Message to analyse",
    placeholder="Type or paste a message here…",
    height=145,
    max_chars=5000,
    label_visibility="collapsed",
)
st.markdown('<div class="hint">Tip: use a complete sentence for a more useful test. Avoid entering personal information.</div>', unsafe_allow_html=True)

col1, col2 = st.columns([3, 1])
with col1:
    detect_clicked = st.button("Analyse message  →", use_container_width=True)
with col2:
    clear_clicked = st.button("Clear", use_container_width=True)

if clear_clicked:
    st.session_state["last_text"] = ""
    st.session_state["last_result"] = None
    st.rerun()

if detect_clicked:
    if not user_text.strip():
        st.warning("Please enter a message before analysing it.")
    else:
        try:
            with st.spinner("Preparing the model and analysing text…"):
                model, model_status = load_or_train_model()
                prediction, confidence, scores = predict_text(model, user_text)
            st.session_state["last_text"] = user_text
            st.session_state["last_result"] = {
                "prediction": prediction,
                "confidence": confidence,
                "scores": scores,
                "model_status": model_status,
            }
        except Exception as exc:
            st.error(f"Could not run the detector: {exc}")

result = st.session_state.get("last_result")
if result:
    pred = result["prediction"]
    title = LABEL_NAMES.get(pred, "Unknown")
    icon = LABEL_ICONS.get(pred, "🔎")
    st.markdown("---")
    st.markdown('<div class="section-label">02 / Model result</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="result-card"><div class="result-kicker">Classification result</div>'
        f'<div class="result-title">{icon} {title}</div>'
        f'<div class="result-copy">{LABEL_DESCRIPTIONS.get(pred, "")}</div></div>',
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Model score", f"{result['confidence'] * 100:.1f}%")
    with c2:
        st.metric("Characters analysed", f"{len(st.session_state.get('last_text', '')):,}")

    with st.expander("View all class scores"):
        for label_id in [0, 1, 2]:
            score = result["scores"].get(label_id, 0.0)
            st.write(f"**{LABEL_NAMES[label_id]}** · {score * 100:.1f}%")
            st.progress(min(max(score, 0.0), 1.0))

    st.caption(
        "Important: the score is the model's estimated class probability, not a guarantee "
        "of correctness. Slang, sarcasm, context, dialect, and dataset bias can affect results. "
        "Use human review for real decisions."
    )

st.markdown("---")
st.markdown(
    '<div class="small-note">HATE-SHIELD AI · Educational prototype · '
    'Predictions may be wrong and should not be used alone to moderate or penalise people.</div>',
    unsafe_allow_html=True,
)
