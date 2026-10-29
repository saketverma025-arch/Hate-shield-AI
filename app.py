
import html
import re
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# ==========================================
# HATE-SHIELD AI
# ==========================================

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

    missing = {"tweet", "class"} - set(df.columns)
    if missing:
        raise ValueError(
            "CSV needs columns named 'tweet' and 'class'. "
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


# ==========================================
# PAGE SETTINGS
# ==========================================

st.set_page_config(
    page_title="Hate-Shield AI",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ==========================================
# DESIGN
# ==========================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');

:root {
    --cyan: #19e6f2;
    --mint: #20f5c5;
    --muted: #a5c2d0;
}

.stApp {
    background:
        radial-gradient(ellipse at 50% 0%, #07313e 0%, transparent 48%),
        linear-gradient(145deg, #020910, #041720 60%, #020910);
    color: #effbff;
    font-family: 'Inter', sans-serif;
}

[data-testid="stHeader"] { background: transparent; }

.block-container {
    max-width: 900px;
    padding: 1.2rem 1rem 3rem;
}

h1, h2, h3 {
    font-family: 'Space Grotesk', sans-serif !important;
    color: #effbff !important;
}

.brand {
    display: flex;
    align-items: center;
    gap: 13px;
    padding: 10px 0 18px;
    border-bottom: 1px solid #164653;
}

.brand-logo {
    display: block;
    width: 52px;
    height: 60px;
    flex-shrink: 0;
    filter: drop-shadow(0 0 8px #00eaff66);
}

.brand-name {
    font: 700 1.15rem 'Space Grotesk', sans-serif;
    letter-spacing: .025em;
}

.brand-name span { color: var(--cyan); }

.brand-sub {
    color: #91b4c4;
    font-size: .7rem;
    letter-spacing: .07em;
    margin-top: 4px;
}

.status {
    margin-left: auto;
    color: var(--mint);
    border: 1px solid #147b70;
    border-radius: 25px;
    padding: 7px 9px;
    font-size: .62rem;
    white-space: nowrap;
}

.hero {
    text-align: center;
    padding: 25px 12px 22px;
    margin: 15px 0;
    background: linear-gradient(145deg, #071d2a99, #062a2b66);
    border: 1px solid #126479;
    border-radius: 20px;
}

.hero-logo {
    display: block;
    width: 105px;
    height: 118px;
    margin: 0 auto 8px;
    filter: drop-shadow(0 0 17px #00dfff55);
}

.hero-title {
    font: 700 clamp(2rem, 7vw, 3.1rem) 'Space Grotesk', sans-serif;
    line-height: 1.1;
    letter-spacing: -.04em;
}

.hero-title span { color: var(--cyan); }

.hero-tag {
    color: var(--mint);
    font-size: .76rem;
    font-weight: 800;
    letter-spacing: .13em;
    margin: 12px 0;
}

.hero-copy {
    color: #b2cbd7;
    max-width: 560px;
    margin: auto;
    font-size: .87rem;
    line-height: 1.75;
}

.section-title {
    font: 700 1.1rem 'Space Grotesk', sans-serif;
    margin: 24px 0 12px;
}

.section-title span { color: var(--cyan); }

.feature {
    background: linear-gradient(145deg, #08252e, #06151e);
    border: 1px solid #14556a;
    border-radius: 13px;
    padding: 15px 12px;
    min-height: 138px;
    margin-bottom: 8px;
}

.feature-icon { font-size: 1.6rem; margin-bottom: 8px; }

.feature-name {
    color: #f0fbff;
    font-size: .86rem;
    font-weight: 800;
    margin-bottom: 6px;
}

.feature-copy {
    color: #a5c2d0;
    font-size: .75rem;
    line-height: 1.6;
}

.guide {
    background: linear-gradient(140deg, #071e2b, #062326);
    border: 1px solid var(--cyan);
    border-radius: 16px;
    padding: 19px;
    margin: 10px 0 18px;
    box-shadow: 0 0 20px #00dfff0c;
}

.guide-heading {
    color: var(--cyan);
    font: 700 1.05rem 'Space Grotesk', sans-serif;
    margin-bottom: 15px;
}

.guide-step {
    display: flex;
    gap: 12px;
    margin: 13px 0;
    align-items: flex-start;
}

.guide-number {
    min-width: 30px;
    height: 30px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #00151c;
    background: var(--cyan);
    border-radius: 9px;
    font-weight: 800;
    font-size: .82rem;
}

.guide-text { color: #c0d7e0; font-size: .83rem; line-height: 1.65; }
.guide-text b { color: #f1fcff; }

[data-testid="stVerticalBlockBorderWrapper"] {
    border-color: #126479 !important;
    border-radius: 15px !important;
    background: #061721 !important;
}

.stTextArea textarea {
    background: #031019 !important;
    color: #f0fbff !important;
    border: 1px solid #176579 !important;
    border-radius: 12px !important;
}

.stTextArea textarea:focus {
    border-color: var(--cyan) !important;
    box-shadow: 0 0 0 1px var(--cyan) !important;
}

div.stButton > button {
    width: 100%;
    min-height: 2.9rem;
    border: 1px solid var(--cyan);
    border-radius: 10px;
    background: linear-gradient(100deg, #22e6f1, #18cbbb);
    color: #00171d;
    font-weight: 800;
}

div.stButton > button:hover {
    color: #00171d;
    border-color: #aaffff;
    filter: brightness(1.08);
}

.result {
    border: 1px solid var(--mint);
    background: linear-gradient(130deg, #06272b, #04151d);
    border-radius: 15px;
    padding: 20px;
    margin: 12px 0;
}

.result.hate { border-color: #ff6078; }
.result.offensive { border-color: #ffbd55; }
.result.safe { border-color: var(--mint); }

.result-label {
    color: #9dbbc9;
    font-size: .7rem;
    font-weight: 800;
    letter-spacing: .13em;
}

.result-title {
    font: 700 1.55rem 'Space Grotesk', sans-serif;
    margin: 10px 0;
}

.hate .result-title { color: #ff7189; }
.offensive .result-title { color: #ffca70; }
.safe .result-title { color: var(--mint); }

.result-copy {
    color: #bad1dc;
    font-size: .87rem;
    line-height: 1.7;
}

.about {
    background: linear-gradient(135deg, #071d29, #062528);
    border: 1px solid #126479;
    border-radius: 15px;
    padding: 20px;
    margin-top: 12px;
}

.about-heading {
    color: var(--cyan);
    font: 700 1.08rem 'Space Grotesk', sans-serif;
    margin-bottom: 12px;
}

.about-copy {
    color: #b5ced9;
    font-size: .84rem;
    line-height: 1.85;
}

.about-list {
    color: #c1d8e1;
    font-size: .82rem;
    line-height: 2;
}

.about-list span { color: var(--mint); }

div[data-testid="stMetric"] {
    background: #061e29;
    border: 1px solid #12546a;
    border-radius: 12px;
    padding: 12px;
}

div[data-testid="stMetricLabel"] { color: #a5c2d0; }
div[data-testid="stMetricValue"] { color: var(--cyan); }

div[data-testid="stExpander"] {
    border: 1px solid #14556a;
    border-radius: 12px;
}

hr { border-color: #164653; }

.footer {
    text-align: center;
    color: #81a3b2;
    font-size: .73rem;
    line-height: 1.8;
    padding: 22px 0 5px;
}

@media (max-width: 600px) {
    .block-container { padding: 1rem .7rem 2rem; }
    .brand-logo { width: 43px; height: 50px; }
    .brand-name { font-size: .95rem; }
    .brand-sub { font-size: .59rem; }
    .status { font-size: .53rem; padding: 6px; }
    .hero { padding: 20px 10px; }
    .guide, .about { padding: 15px; }
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# ONE CONSISTENT CUSTOM SHIELD LOGO
# ==========================================

logo_svg = """
<svg viewBox="0 0 100 112" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="hs-gradient" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#a3ffff"/>
      <stop offset="48%" stop-color="#19e6f2"/>
      <stop offset="100%" stop-color="#0875ff"/>
    </linearGradient>
  </defs>
  <path d="M50 3 L94 21 L89 65 Q80 91 50 109
           Q20 91 11 65 L6 21 Z"
        fill="#041722" stroke="url(#hs-gradient)" stroke-width="4"/>
  <path d="M50 12 L83 26 L79 62 Q71 82 50 96
           Q29 82 21 62 L17 26 Z"
        fill="#06232e" stroke="#087dba" stroke-width="1.5"/>
  <path d="M64 27 H39 L27 40 V67 L42 80 V58 H57 V77
           L73 63 V41 H60 V49 H43 V38 H64 Z"
        fill="url(#hs-gradient)"/>
  <path d="M50 3 L50 12" stroke="#d4ffff" stroke-width="2"/>
</svg>
"""

st.markdown(f"""
<div class="brand">
  <div class="brand-logo">{logo_svg}</div>
  <div>
    <div class="brand-name">HATE-SHIELD <span>AI</span></div>
    <div class="brand-sub">AI-POWERED TEXT ANALYSIS</div>
  </div>
  <div class="status">● TEXT DETECTOR</div>
</div>

<div class="hero">
  <div class="hero-logo">{logo_svg}</div>
  <div class="hero-title">HATE-SHIELD <span>AI</span></div>
  <div class="hero-tag">DETECT · ANALYSE · PROMOTE KINDNESS</div>
  <div class="hero-copy">
    Explore how artificial intelligence classifies online language.
    Enter a message to see whether the model categorises it as hate
    speech, offensive language or neither.
  </div>
</div>
""", unsafe_allow_html=True)

# ==========================================
# FEATURE CARDS
# ==========================================

st.markdown(
    '<div class="section-title">✦ &nbsp; <span>WHAT THIS APP DOES</span></div>',
    unsafe_allow_html=True,
)

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""
    <div class="feature">
      <div class="feature-icon">🛡️</div>
      <div class="feature-name">Hate Speech</div>
      <div class="feature-copy">
        Classifies language patterns associated with hate speech.
      </div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="feature">
      <div class="feature-icon">⚠️</div>
      <div class="feature-name">Offensive Language</div>
      <div class="feature-copy">
        Identifies patterns associated with offensive content.
      </div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="feature">
      <div class="feature-icon">✓</div>
      <div class="feature-name">Neither</div>
      <div class="feature-copy">
        Shows when neither of the first two labels is predicted.
      </div>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# USER GUIDE BOX
# ==========================================

st.markdown("""
<div class="guide">
  <div class="guide-heading">▣ &nbsp; HOW TO USE HATE-SHIELD AI</div>

  <div class="guide-step">
    <div class="guide-number">01</div>
    <div class="guide-text">
      <b>Enter your message</b><br>
      Type or paste the text you want the model to analyse into the
      message input box below.
    </div>
  </div>

  <div class="guide-step">
    <div class="guide-number">02</div>
    <div class="guide-text">
      <b>Run the detector</b><br>
      Tap the Analyse Message button. The app loads the trained
      machine-learning model and predicts a category.
    </div>
  </div>

  <div class="guide-step">
    <div class="guide-number">03</div>
    <div class="guide-text">
      <b>Read the result</b><br>
      Check the result panel to see Hate Speech, Offensive Language
      or Neither, along with the model's estimated class score.
    </div>
  </div>

  <div class="guide-step">
    <div class="guide-number">04</div>
    <div class="guide-text">
      <b>Understand the prediction</b><br>
      The model can make mistakes. Read the limitations and do not
      treat a prediction as a definitive judgement of a person.
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ==========================================
# MESSAGE ANALYSER
# ==========================================

st.markdown(
    '<div class="section-title">01 / <span>MESSAGE ANALYSER</span></div>',
    unsafe_allow_html=True,
)

with st.container(border=True):
    st.markdown("**Enter text for analysis**")
    st.caption("Avoid entering private or personally identifying information.")

    user_text = st.text_area(
        "Message",
        placeholder="Type or paste a message here...",
        height=140,
        max_chars=5000,
        label_visibility="collapsed",
        key="message_input",
    )

    col1, col2 = st.columns([2, 1])

    with col1:
        detect_clicked = st.button(
            "🛡️  ANALYSE MESSAGE",
            use_container_width=True,
        )

    with col2:
        clear_clicked = st.button("CLEAR", use_container_width=True)

    with st.expander("Try a sample message"):
        st.caption("These are test examples, not guaranteed model results.")

        if st.button("Sample 1: Neutral message", use_container_width=True):
            st.session_state["message_input"] = (
                "I really enjoyed watching the cricket game today."
            )
            st.rerun()

        if st.button("Sample 2: Offensive wording", use_container_width=True):
            st.session_state["message_input"] = "You are an idiot."
            st.rerun()

        if st.button("Sample 3: Positive message", use_container_width=True):
            st.session_state["message_input"] = (
                "Thank you for helping me with my work."
            )
            st.rerun()

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

# ==========================================
# RESULT PANEL
# ==========================================

st.markdown(
    '<div class="section-title">02 / <span>ANALYSIS RESULT</span></div>',
    unsafe_allow_html=True,
)

result = st.session_state.get("last_result")

if result:
    pred = result["prediction"]
    title = LABEL_NAMES.get(pred, "Unknown")
    icon = LABEL_ICONS.get(pred, "🔎")
    css_class = RESULT_CLASSES.get(pred, "safe")
    confidence = result["scores"].get(pred, 0.0)

    st.markdown(
        f'<div class="result {css_class}">'
        f'<div class="result-label">MODEL CLASSIFICATION</div>'
        f'<div class="result-title">{icon} &nbsp; {title}</div>'
        f'<div class="result-copy">'
        f'{LABEL_DESCRIPTIONS.get(pred, "")}'
        f'</div></div>',
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
        "Scores are estimated class probabilities, not guarantees of "
        "correctness. Context, sarcasm, slang and dataset bias can affect "
        "results. Human review is important."
    )

else:
    st.markdown("""
    <div class="result safe" style="text-align:center;">
      <div class="result-label">AWAITING INPUT</div>
      <div class="result-title" style="color:#19e6f2;">
        Your result will appear here
      </div>
      <div class="result-copy">
        Enter a message above and tap Analyse Message to begin.
      </div>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# ABOUT BOX
# ==========================================

st.markdown(
    '<div class="section-title">03 / <span>ABOUT THE PROJECT</span></div>',
    unsafe_allow_html=True,
)

st.markdown("""
<div class="about">
  <div class="about-heading">ⓘ &nbsp; ABOUT HATE-SHIELD AI</div>

  <div class="about-copy">
    Hate-Shield AI is a Class XII Artificial Intelligence capstone project
    exploring how machine learning can classify language in online text.
    It uses TF-IDF (Term Frequency-Inverse Document Frequency) to convert
    text into numerical features and Logistic Regression to predict one
    of three categories: Hate Speech, Offensive Language or Neither.
    The model learns from labelled examples in the supplied dataset.
  </div>

  <br>

  <div class="about-heading">✦ &nbsp; KEY FEATURES</div>

  <div class="about-list">
    <span>✓</span> &nbsp; Three-category text classification<br>
    <span>✓</span> &nbsp; TF-IDF feature extraction<br>
    <span>✓</span> &nbsp; Logistic Regression model<br>
    <span>✓</span> &nbsp; Interactive text analysis<br>
    <span>✓</span> &nbsp; Class-score display and responsible-use guidance
  </div>

  <br>

  <div class="about-copy">
    <b style="color:#19e6f2;">Limitations:</b>
    Predictions may be wrong because language depends on context and
    the training data may contain bias. This educational prototype
    should not be used alone to moderate, punish or judge people.
  </div>
</div>
""", unsafe_allow_html=True)

# ==========================================
# FOOTER
# ==========================================

st.markdown("""
<div class="footer">
  🛡️ HATE-SHIELD AI · CLASS XII AI CAPSTONE<br>
  Safer Internet. Kinder Communities.<br>
  Educational prototype · Predictions require human judgement.
</div>
""", unsafe_allow_html=True)

