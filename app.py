
import html
import re
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


# ================ FILE PATHS ================

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "twitter_data.csv"
MODEL_PATH = BASE_DIR / "hate_speech_model.joblib"
LOGO_PATH = BASE_DIR / "hate_shield_logo_12.png"


# ================ PAGE CONFIG ================

st.set_page_config(
    page_title="Hate-Shield AI",
    page_icon="🛡️",
    layout="wide",
)


# ================ DESIGN ================

st.markdown("""
<style>
.stApp {
    background: radial-gradient(ellipse at top, #08283c 0%, #020b17 65%);
    color: #eaf5ff;
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] {
    background: #03111f;
    border-right: 1px solid #08708b;
}
h1, h2, h3 { color: #f1f8ff !important; }
p, label { color: #c7dff6; }

.hero {
    background: linear-gradient(120deg, #031323, #073149, #031323);
    border: 1px solid #00d9ff;
    border-radius: 18px;
    padding: 24px;
    margin-bottom: 20px;
    box-shadow: 0 0 24px #00d9ff18;
}
.hero-title {
    font-size: clamp(28px, 4vw, 44px);
    font-weight: 900;
    letter-spacing: 1px;
}
.cyan { color: #20e5ff; }
.tagline {
    color: #67dfff;
    letter-spacing: 2px;
    font-size: 12px;
    font-weight: 800;
}
.panel {
    background: linear-gradient(135deg, #041829, #031321);
    border: 1px solid #087b99;
    border-radius: 15px;
    padding: 20px;
    margin-bottom: 18px;
}
.section-title {
    font-size: 21px;
    font-weight: 850;
    margin-bottom: 6px;
}
.muted { color: #91bce0; font-size: 14px; }

.stTextArea textarea {
    background: #061626 !important;
    color: #f0f8ff !important;
    border: 1px solid #00c8ee !important;
    border-radius: 12px !important;
    font-size: 16px !important;
}
.stButton > button {
    border: 1px solid #16dff5;
    border-radius: 10px;
    min-height: 44px;
    font-weight: 800;
    background: linear-gradient(90deg, #35e6ef, #00bfe8);
    color: #001521;
}
.stButton > button:hover {
    box-shadow: 0 0 18px #00d9ff50;
    border-color: #a4f9ff;
    color: #001521;
}

.guide-card {
    background: #061a2b;
    border: 1px solid #12536b;
    border-radius: 11px;
    padding: 13px;
    min-height: 120px;
}
.result {
    border: 1px solid;
    border-radius: 14px;
    padding: 22px;
    margin: 12px 0;
}
.hate {
    background: #35141c;
    border-color: #ff5364;
}
.offensive {
    background: #352a12;
    border-color: #ffc247;
}
.neither {
    background: #103328;
    border-color: #39d99a;
}
.hate .result-title { color: #ff6979; }
.offensive .result-title { color: #ffd16b; }
.neither .result-title { color: #55e6ad; }
.result-title { font-size: 27px; font-weight: 900; }
.score-card {
    background: #061626;
    border: 1px solid #16445b;
    border-radius: 9px;
    padding: 12px;
    margin: 8px 0;
}
.footer {
    text-align: center;
    color: #739cbd;
    padding: 20px 0;
    font-size: 12px;
}
</style>
""", unsafe_allow_html=True)


# ================ LOGO DISPLAY ================

def show_logo(width):
    if LOGO_PATH.is_file():
        st.image(str(LOGO_PATH), width=width)
    else:
        st.warning(
            "Logo file not found. Upload "
            "'hate_shield_logo_12.png' beside app.py."
        )


# ================ MACHINE LEARNING ================

LABELS = {
    0: "Hate Speech",
    1: "Offensive Language",
    2: "Neither",
}

RESULT_STYLES = {
    0: ("hate", "🔴", "#ff5364"),
    1: ("offensive", "🟡", "#ffc247"),
    2: ("neither", "🟢", "#39d99a"),
}


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


@st.cache_resource
def load_or_train_model():
    if MODEL_PATH.is_file():
        try:
            return joblib.load(MODEL_PATH)
        except Exception:
            pass

    if not DATASET_PATH.is_file():
        raise FileNotFoundError(
            "twitter_data.csv is missing. Upload it beside app.py."
        )

    data = pd.read_csv(DATASET_PATH)

    if not {"tweet", "class"}.issubset(data.columns):
        raise ValueError(
            "Dataset columns must include 'tweet' and 'class'."
        )

    data = data[["tweet", "class"]].dropna()
    data["class"] = pd.to_numeric(data["class"], errors="coerce")
    data = data.dropna(subset=["class"])
    data["class"] = data["class"].astype(int)
    data = data[data["class"].isin([0, 1, 2])]

    data["tweet"] = data["tweet"].apply(clean_tweet)
    data = data[data["tweet"].str.len() > 0]

    if data.empty or data["class"].nunique() < 2:
        raise ValueError(
            "The dataset needs text examples from at least "
            "two supported classes."
        )

    model = build_model()
    model.fit(data["tweet"], data["class"])

    try:
        joblib.dump(model, MODEL_PATH)
    except Exception:
        pass

    return model


def predict_text(model, text):
    cleaned = clean_tweet(text)
    if not cleaned:
        raise ValueError("Please enter some text.")

    prediction = int(model.predict([cleaned])[0])
    probabilities = model.predict_proba([cleaned])[0]
    classes = model.named_steps["classifier"].classes_

    scores = {
        int(label): float(score)
        for label, score in zip(classes, probabilities)
    }
    return prediction, scores


# ================ SIDEBAR ================

with st.sidebar:
    show_logo(210)

    st.markdown(
        "<h2 style='text-align:center'>HATE-SHIELD "
        "<span style='color:#20e5ff'>AI</span></h2>",
        unsafe_allow_html=True,
    )
    st.caption("SAFER INTERNET. KINDER COMMUNITIES.")
    st.markdown("---")
    st.markdown("### 🧭 NAVIGATION")
    st.markdown("🏠 Home")
    st.markdown("💬 Detect Message")
    st.markdown("ℹ️ About")
    st.markdown("---")
    st.markdown(
        "Better words.<br>A kinder internet.<br>A safer tomorrow.",
        unsafe_allow_html=True,
    )


# ================ HERO ================

left, right = st.columns([4, 1])

with left:
    st.markdown("""
    <div class="hero">
        <div class="hero-title">
            HATE-SHIELD <span class="cyan">AI</span>
        </div>
        <p class="tagline">
            DETECT · ANALYSE · MAKE A DIFFERENCE
        </p>
        <p>
            An AI-powered educational tool that classifies text into
            hate speech, offensive language, or neither, helping
            encourage safer and more respectful online communication.
        </p>
        <p style="color:#55e8ff;font-style:italic">
            Kindness is a form of strength.
        </p>
    </div>
    """, unsafe_allow_html=True)

with right:
    show_logo(180)


# ================ HOW TO USE ================

st.markdown('<div class="panel">', unsafe_allow_html=True)
st.markdown(
    '<div class="section-title">📘 HOW TO USE HATE-SHIELD AI</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="muted">Four simple steps to analyse a message.</p>',
    unsafe_allow_html=True,
)

guide_columns = st.columns(4)
guide_items = [
    ("01", "Enter text", "Type or paste a message."),
    ("02", "Detect", "Press the Detect button."),
    ("03", "Review", "Read the result and class scores."),
    ("04", "Use judgement", "Check the context; AI can be wrong."),
]

for col, (num, title, description) in zip(guide_columns, guide_items):
    with col:
        st.markdown(
            f'<div class="guide-card">'
            f'<span style="color:#20e5ff;font-weight:900">{num}</span>'
            f'<br><b>{title}</b>'
            f'<p class="muted">{description}</p></div>',
            unsafe_allow_html=True,
        )

st.markdown('</div>', unsafe_allow_html=True)


# ================ MESSAGE INPUT ================

st.markdown('<div class="panel">', unsafe_allow_html=True)
st.markdown(
    '<div class="section-title">💬 ANALYSE A MESSAGE</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="muted">Enter a message below. Sample buttons have '
    'been removed.</p>',
    unsafe_allow_html=True,
)

message = st.text_area(
    "Message",
    key="message_input",
    placeholder="Type or paste your message here...",
    height=150,
    max_chars=5000,
    label_visibility="collapsed",
)

detect_col, clear_col, blank_col = st.columns([2, 1, 2])

with detect_col:
    detect_clicked = st.button(
        "🛡️ DETECT MESSAGE →",
        type="primary",
        use_container_width=True,
    )

with clear_col:
    if st.button("↻ Clear", use_container_width=True):
        st.session_state["message_input"] = ""
        st.rerun()

st.markdown('</div>', unsafe_allow_html=True)


# ================ DETECTION RESULTS ================

st.markdown('<div class="panel">', unsafe_allow_html=True)
st.markdown(
    '<div class="section-title">📊 DETECTION RESULT</div>',
    unsafe_allow_html=True,
)

if detect_clicked:
    if not message.strip():
        st.warning("Please enter a message first.")
    else:
        try:
            with st.spinner("Analysing your message..."):
                model = load_or_train_model()
                prediction, scores = predict_text(model, message)

            style, icon, colour = RESULT_STYLES[prediction]
            confidence = scores.get(prediction, 0.0) * 100

            descriptions = {
                0: "The model classified this text as hate speech. "
                   "Review the wording and context carefully.",
                1: "The model classified this text as offensive language. "
                   "Offensive wording is not always hate speech.",
                2: "The model classified this text as neither hate "
                   "speech nor offensive language.",
            }

            st.markdown(
                f'<div class="result {style}">'
                f'<div style="font-size:30px">{icon}</div>'
                f'<div class="result-title">{LABELS[prediction]}</div>'
                f'<p>{descriptions[prediction]}</p>'
                f'<b>Model score: {confidence:.1f}%</b>'
                f'</div>',
                unsafe_allow_html=True,
            )

            st.markdown("#### Class scores")

            for label_id in [0, 1, 2]:
                _, _, score_colour = RESULT_STYLES[label_id]
                score = scores.get(label_id, 0.0)

                st.markdown(
                    f'<div class="score-card">'
                    f'<span style="color:{score_colour};font-weight:800">'
                    f'{LABELS[label_id]}</span>'
                    f'<span style="float:right">{score * 100:.1f}%</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                st.progress(score)

            st.caption(
                "Scores are model estimates, not guarantees of correctness."
            )

        except Exception as error:
            st.error(f"Detection failed: {error}")

else:
    st.markdown("""
    <div style="text-align:center;padding:25px;
                background:#041626;border:1px solid #08708a;
                border-radius:12px">
        <div style="font-size:38px">🛡️</div>
        <h3 style="color:#20e5ff !important">
            Your result will appear here
        </h3>
        <p class="muted">
            Enter a message and click Detect Message.
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)


# ================ ABOUT ================

st.markdown('<div class="panel">', unsafe_allow_html=True)
st.markdown(
    '<div class="section-title">ℹ️ ABOUT HATE-SHIELD AI</div>',
    unsafe_allow_html=True,
)

about_left, about_right = st.columns(2)

with about_left:
    st.markdown("""
    **Hate-Shield AI** is a Class XII Artificial Intelligence
    capstone project.

    It uses TF-IDF text vectorisation and Logistic Regression to
    classify messages into three categories using labelled tweet data.

    **Dataset labels**
    - `0`: Hate Speech
    - `1`: Offensive Language
    - `2`: Neither
    """)

with about_right:
    st.markdown("""
    ### ✨ KEY FEATURES
    - Three-category text classification
    - Colour-coded detection results
    - Estimated class scores
    - Simple, accessible interface
    - Guide and project information
    """)

st.markdown('</div>', unsafe_allow_html=True)


# ================ RESPONSIBLE USE ================

st.info(
    "Educational project only: the model can misread sarcasm, slang, "
    "context, or unfamiliar wording. Do not treat a prediction as proof "
    "that a person is hateful. Human review is important."
)

st.markdown("""
<div class="footer">
    HATE-SHIELD AI · CLASS XII AI CAPSTONE PROJECT<br>
    BETTER WORDS. A KINDER INTERNET. A SAFER TOMORROW.
</div>
""", unsafe_allow_html=True)
