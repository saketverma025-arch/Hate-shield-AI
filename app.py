
import html
import re
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


# -------------------- FILE PATHS --------------------

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "twitter_data.csv"
MODEL_PATH = BASE_DIR / "hate_speech_model.joblib"
LOGO_PATH = BASE_DIR / "hate_shield_logo.png"


# -------------------- PAGE SETTINGS --------------------

st.set_page_config(
    page_title="Hate-Shield AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -------------------- THEME --------------------

st.markdown("""
<style>
.stApp {
    background:
        radial-gradient(ellipse at 70% 0%, #06263b 0%, transparent 45%),
        linear-gradient(140deg, #020d1b, #031729 65%, #00101e);
    color: #eaf5ff;
}

[data-testid="stHeader"] {
    background: transparent;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #03101f, #04192a);
    border-right: 1px solid #07506a;
}

[data-testid="stSidebar"] * {
    color: #d8eaff;
}

h1, h2, h3 {
    color: #f2f8ff !important;
}

p, li, label {
    color: #d1e4ff;
}

.hero {
    background: linear-gradient(115deg, #041426, #062b40, #031323);
    border: 1px solid #00d9ff;
    border-radius: 20px;
    padding: 26px;
    margin: 6px 0 22px 0;
    box-shadow: 0 0 22px #00cfff16;
}

.hero-title {
    font-size: clamp(25px, 3vw, 42px);
    font-weight: 900;
    letter-spacing: 2px;
    line-height: 1.15;
}

.cyan {
    color: #21e6ff;
}

.tagline {
    color: #67dfff;
    font-size: 13px;
    letter-spacing: 3px;
    margin-top: 10px;
    font-weight: 700;
}

.panel {
    background: linear-gradient(135deg, #041629, #031323);
    border: 1px solid #007f9e;
    border-radius: 16px;
    padding: 20px;
    margin-bottom: 18px;
}

.section-title {
    font-size: 20px;
    font-weight: 800;
    margin-bottom: 5px;
}

.muted {
    color: #91bce5;
    font-size: 14px;
}

.stTextArea textarea {
    background: #061626 !important;
    color: #eaf7ff !important;
    border: 1px solid #00bfe9 !important;
    border-radius: 12px !important;
    font-size: 16px !important;
}

.stTextArea textarea:focus {
    border-color: #35efff !important;
    box-shadow: 0 0 0 1px #35efff !important;
}

.stButton > button {
    border-radius: 10px;
    min-height: 44px;
    font-weight: 750;
    border: 1px solid #00cce9;
    background: linear-gradient(90deg, #35e5ef, #00bde8);
    color: #001522;
    transition: 0.2s;
}

.stButton > button:hover {
    border-color: #8af6ff;
    box-shadow: 0 0 16px #00d9ff45;
    color: #001522;
}

.result-box {
    padding: 22px;
    border-radius: 14px;
    border: 1px solid;
    margin: 10px 0;
}

.result-hate {
    background: #32131b;
    border-color: #ff5364;
    box-shadow: 0 0 18px #ff536415;
}

.result-offensive {
    background: #332811;
    border-color: #ffc247;
    box-shadow: 0 0 18px #ffc24715;
}

.result-neither {
    background: #103126;
    border-color: #39d99a;
    box-shadow: 0 0 18px #39d99a15;
}

.result-title {
    font-size: 25px;
    font-weight: 900;
    margin-bottom: 6px;
}

.result-hate .result-title {
    color: #ff6978;
}

.result-offensive .result-title {
    color: #ffd16b;
}

.result-neither .result-title {
    color: #55e6ad;
}

.score-row {
    background: #061626;
    border: 1px solid #16445b;
    padding: 12px;
    border-radius: 10px;
    margin: 8px 0;
}

.guide-step {
    background: #061a2b;
    border: 1px solid #11516a;
    border-radius: 11px;
    padding: 13px;
    margin: 7px 0;
}

.feature-card {
    background: #061a2b;
    border: 1px solid #10516b;
    border-radius: 12px;
    padding: 15px;
    min-height: 100px;
}

.footer {
    text-align: center;
    color: #739cbd;
    font-size: 12px;
    padding: 20px 0 5px 0;
}
</style>
""", unsafe_allow_html=True)


# -------------------- MODEL --------------------

LABELS = {
    0: "Hate Speech",
    1: "Offensive Language",
    2: "Neither",
}

LABEL_INFO = {
    0: {
        "class_name": "result-hate",
        "icon": "🔴",
        "description": (
            "The model classified this text as hate speech. "
            "Review the context carefully before taking action."
        ),
    },
    1: {
        "class_name": "result-offensive",
        "icon": "🟡",
        "description": (
            "The model classified this text as offensive language. "
            "Offensive wording is not automatically hate speech."
        ),
    },
    2: {
        "class_name": "result-neither",
        "icon": "🟢",
        "description": (
            "The model classified this text as neither hate speech "
            "nor offensive language."
        ),
    },
}


def clean_tweet(text):
    text = html.unescape(str(text))
    text = re.sub(r"https?://\S+|www\.\S+", " URL ", text)
    text = re.sub(r"@\w+", " USER ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def build_model():
    return Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                max_features=50000,
                sublinear_tf=True,
                min_df=2,
            ),
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1500,
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ])


@st.cache_resource
def load_or_train_model():
    if MODEL_PATH.exists():
        try:
            return joblib.load(MODEL_PATH)
        except Exception:
            # If the saved model cannot be loaded, train it again.
            pass

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            "twitter_data.csv was not found. "
            "Upload it to the same folder as app.py."
        )

    data = pd.read_csv(DATASET_PATH)

    if not {"tweet", "class"}.issubset(data.columns):
        raise ValueError(
            "The dataset must contain columns named 'tweet' and 'class'."
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
            "The dataset needs usable text examples from at least "
            "two of the supported classes."
        )

    model = build_model()
    model.fit(data["tweet"], data["class"])

    try:
        joblib.dump(model, MODEL_PATH)
    except Exception:
        # The app can still work if saving the model is not possible.
        pass

    return model


def predict_text(model, text):
    cleaned = clean_tweet(text)

    if not cleaned:
        raise ValueError("Please enter some text to analyse.")

    prediction = int(model.predict([cleaned])[0])
    probabilities = model.predict_proba([cleaned])[0]

    scores = {
        int(label): float(probability)
        for label, probability in zip(
            model.named_steps["classifier"].classes_,
            probabilities,
        )
    }

    return prediction, scores


# -------------------- SIDEBAR --------------------

with st.sidebar:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=175)
    else:
        st.markdown(
            "<h1 style='text-align:center;color:#21e6ff'>🛡️</h1>",
            unsafe_allow_html=True,
        )

    st.markdown(
        "<h2 style='text-align:center;font-size:22px'>"
        "HATE-SHIELD <span style='color:#21e6ff'>AI</span></h2>",
        unsafe_allow_html=True,
    )
    st.caption("SAFER INTERNET. KINDER COMMUNITIES.")

    st.markdown("---")
    st.markdown("### 🧭 Navigation")
    st.markdown("🏠 Home")
    st.markdown("💬 Detect Message")
    st.markdown("ℹ️ About")

    st.markdown("---")
    st.markdown(
        "<p style='color:#91bce5'>Better words.<br>"
        "A kinder internet.<br>A safer tomorrow.</p>",
        unsafe_allow_html=True,
    )
    st.markdown("<span style='color:#21e6ff'>━━━━━━</span>",
                unsafe_allow_html=True)


# -------------------- HERO HEADER --------------------

hero_left, hero_right = st.columns([5, 1])

with hero_left:
    st.markdown("""
    <div class="hero">
        <div class="hero-title">
            HATE-SHIELD <span class="cyan">AI</span>
        </div>
        <div class="tagline">
            DETECT &nbsp; • &nbsp; ANALYSE &nbsp; • &nbsp;
            MAKE A DIFFERENCE
        </div>
        <p style="margin-top:15px;max-width:720px">
            An AI-powered tool to detect hate speech, offensive language,
            and other text. Helping build a safer and more respectful
            online space.
        </p>
        <p style="color:#62dfff;font-style:italic;margin-bottom:0">
            Kindness is a form of strength.
        </p>
    </div>
    """, unsafe_allow_html=True)

with hero_right:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=140)


# -------------------- GUIDE --------------------

st.markdown('<div class="panel">', unsafe_allow_html=True)
st.markdown(
    '<div class="section-title">📘 HOW TO USE HATE-SHIELD AI</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="muted">Four simple steps to analyse a message.</p>',
    unsafe_allow_html=True,
)

guide_cols = st.columns(4)
guide = [
    ("01", "Enter text", "Type or paste a message in the box below."),
    ("02", "Run detector", "Press Detect to analyse your message."),
    ("03", "Read the result", "Check the category and model scores."),
    ("04", "Use your judgement", "Review context. AI can make mistakes."),
]

for col, (number, title, description) in zip(guide_cols, guide):
    with col:
        st.markdown(
            f'<div class="guide-step">'
            f'<span style="color:#21e6ff;font-weight:900">{number}</span>'
            f'<br><b>{title}</b>'
            f'<p class="muted">{description}</p>'
            f'</div>',
            unsafe_allow_html=True,
        )

st.markdown('</div>', unsafe_allow_html=True)


# -------------------- MESSAGE DETECTOR --------------------

st.markdown('<div class="panel">', unsafe_allow_html=True)
st.markdown(
    '<div class="section-title">💬 ANALYSE A MESSAGE</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="muted">Enter a message to check how the model classifies it. '
    'No sample-message buttons are used.</p>',
    unsafe_allow_html=True,
)


def clear_message():
    st.session_state["message_input"] = ""


message = st.text_area(
    "Message to analyse",
    key="message_input",
    height=150,
    max_chars=5000,
    placeholder="Type or paste your message here...",
    label_visibility="collapsed",
)

button_col, clear_col, empty_col = st.columns([2, 1, 2])

with button_col:
    detect_clicked = st.button(
        "🛡️  DETECT MESSAGE  →",
        use_container_width=True,
        type="primary",
    )

with clear_col:
    st.button(
        "↻  Clear",
        use_container_width=True,
        on_click=clear_message,
    )

st.markdown('</div>', unsafe_allow_html=True)


# -------------------- DETECTION RESULT --------------------

st.markdown('<div class="panel">', unsafe_allow_html=True)
st.markdown(
    '<div class="section-title">📊 DETECTION RESULT</div>',
    unsafe_allow_html=True,
)

if detect_clicked:
    if not message.strip():
        st.warning("Please enter a message before pressing Detect.")
    else:
        with st.spinner("Analysing your message..."):
            try:
                model = load_or_train_model()
                prediction, scores = predict_text(model, message)

                result = LABEL_INFO[prediction]
                confidence = scores.get(prediction, 0.0) * 100

                st.markdown(
                    f'<div class="result-box {result["class_name"]}">'
                    f'<div style="font-size:30px">{result["icon"]}</div>'
                    f'<div class="result-title">{LABELS[prediction]}</div>'
                    f'<p>{result["description"]}</p>'
                    f'<b>Model score: {confidence:.1f}%</b>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                st.markdown("#### Class scores")

                for label_id in [0, 1, 2]:
                    score = scores.get(label_id, 0.0)
                    colour = {
                        0: "#ff5364",
                        1: "#ffc247",
                        2: "#39d99a",
                    }[label_id]

                    st.markdown(
                        f'<div class="score-row">'
                        f'<span style="color:{colour};font-weight:800">'
                        f'{LABELS[label_id]}</span>'
                        f'<span style="float:right">{score * 100:.1f}%</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    st.progress(score)

                st.caption(
                    "These scores are the model's estimated class "
                    "probabilities, not a guarantee of correctness."
                )

            except Exception as error:
                st.error(f"Could not run detection: {error}")
                st.info(
                    "Check that twitter_data.csv and the required "
                    "Python libraries are available in your project."
                )
else:
    st.markdown("""
    <div style="text-align:center;padding:28px 10px;
                background:#041626;border:1px solid #08708a;
                border-radius:12px">
        <div style="font-size:38px">🛡️</div>
        <h3 style="color:#21e6ff !important">
            Your result will appear here
        </h3>
        <p class="muted">
            Enter a message above and click Detect Message.
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)


# -------------------- ABOUT --------------------

st.markdown('<div class="panel">', unsafe_allow_html=True)
st.markdown(
    '<div class="section-title">ℹ️ ABOUT HATE-SHIELD AI</div>',
    unsafe_allow_html=True,
)

about_left, about_right = st.columns(2)

with about_left:
    st.markdown("""
    **Hate-Shield AI** is a Class XII Artificial Intelligence capstone
    project designed to help identify potentially harmful language in text.

    The project uses a machine-learning pipeline with **TF-IDF**
    text vectorisation and **Logistic Regression** to classify a message
    into one of three dataset categories.

    The model learns from labelled examples in the supplied dataset.
    """)

with about_right:
    st.markdown("### ✨ KEY FEATURES")
    st.markdown("""
    - Detects three text categories
    - Displays estimated class scores
    - Uses TF-IDF and Logistic Regression
    - Includes a simple, user-friendly interface
    - Supports review of potentially harmful text
    """)

st.markdown('</div>', unsafe_allow_html=True)


# -------------------- IMPORTANT NOTE --------------------

st.markdown("""
<div class="panel">
    <b style="color:#21e6ff">⚠️ Important note</b>
    <p class="muted">
        This is an educational AI project, not a perfect judge of language.
        It may misclassify sarcasm, context, slang, or unfamiliar wording.
        A prediction should not be treated as proof that a person or message
        is hateful. Human review is important.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="footer">
    HATE-SHIELD AI · CLASS XII AI CAPSTONE PROJECT<br>
    BETTER WORDS. A KINDER INTERNET. A SAFER TOMORROW.
</div>
""", unsafe_allow_html=True)
