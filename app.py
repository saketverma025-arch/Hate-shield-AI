
import html
import re
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# ==========================================
# HATE-SHIELD AI | CLASS XII AI CAPSTONE
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
            "twitter_data.csv was not found beside app.py."
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
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ==========================================
# FUTURISTIC CYBERSECURITY THEME
# ==========================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');

:root {
    --bg: #020b11;
    --panel: #061923;
    --teal: #19e6f2;
    --green: #20f5c5;
    --muted: #a4c7d8;
}

.stApp {
    background:
        radial-gradient(ellipse at 50% 0%, #07313e 0%, transparent 45%),
        linear-gradient(140deg, #020910, #03141d 60%, #020910);
    color: #effbff;
    font-family: 'Inter', sans-serif;
}

[data-testid="stHeader"] { background: transparent; }
[data-testid="stDecoration"] { display: none; }

.block-container {
    max-width: 1120px;
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

h1, h2, h3 {
    font-family: 'Space Grotesk', sans-serif !important;
    color: #f0fbff !important;
}

.brand {
    display: flex;
    align-items: center;
    gap: 13px;
    border-bottom: 1px solid #164353;
    padding: 8px 0 20px;
}

.brand-logo {
    width: 58px;
    height: 66px;
    flex-shrink: 0;
    filter: drop-shadow(0 0 10px #00eaff77);
}

.brand-name {
    font: 700 1.3rem 'Space Grotesk', sans-serif;
    letter-spacing: .025em;
}
.brand-name span { color: var(--teal); }
.brand-sub {
    color: #92b7c9;
    font-size: .76rem;
    letter-spacing: .09em;
}

.status-pill {
    margin-left: auto;
    color: var(--green);
    border: 1px solid #147d77;
    background: #062b2b;
    border-radius: 30px;
    padding: 8px 12px;
    font-size: .68rem;
    white-space: nowrap;
}

.hero {
    display: flex;
    align-items: center;
    gap: 28px;
    padding: 26px 18px;
    margin: 20px 0;
    border: 1px solid #087d94;
    border-radius: 20px;
    background: linear-gradient(115deg, #061c2b, #06282d, #04131e);
    box-shadow: 0 0 25px #00cfe90c, inset 0 0 22px #00d9f30a;
}

.hero-logo {
    width: 145px;
    max-width: 30%;
    flex-shrink: 0;
    filter: drop-shadow(0 0 18px #00dfff66);
}

.hero h1 {
    font-size: clamp(2rem, 5vw, 3.3rem) !important;
    letter-spacing: -.04em;
    margin: 0 0 5px !important;
    line-height: 1.1;
}
.hero h1 span { color: var(--teal); }

.hero-tag {
    color: var(--teal);
    letter-spacing: .16em;
    font-size: .73rem;
    font-weight: 800;
    margin: 10px 0;
}
.hero-copy {
    color: #b5d1df;
    font-size: .88rem;
    line-height: 1.75;
    max-width: 620px;
}

.section-title {
    font: 700 1.15rem 'Space Grotesk', sans-serif;
    margin: 24px 0 12px;
    color: #effbff;
}
.section-title span { color: var(--teal); }

.feature {
    border: 1px solid #146578;
    background: linear-gradient(145deg, #08242e, #06151f);
    border-radius: 14px;
    padding: 16px 14px;
    min-height: 145px;
    margin-bottom: 8px;
}
.feature-icon { font-size: 1.65rem; margin-bottom: 9px; }
.feature-name {
    font-size: .9rem;
    font-weight: 800;
    color: #effbff;
    margin-bottom: 7px;
}
.feature-copy {
    font-size: .78rem;
    color: #9ebfce;
    line-height: 1.6;
}

[data-testid="stVerticalBlockBorderWrapper"] {
    border-color: #087d94 !important;
    border-radius: 16px !important;
    background: linear-gradient(140deg, #061923, #04121b) !important;
}

.input-label {
    font-weight: 800;
    font-size: 1rem;
    margin-bottom: 6px;
}
.input-help {
    color: #a2c6d7;
    font-size: .8rem;
    line-height: 1.6;
    margin-bottom: 12px;
}

.stTextArea textarea {
    background: #03111c !important;
    color: #effbff !important;
    border: 1px solid #087d94 !important;
    border-radius: 12px !important;
    font-size: .96rem !important;
}
.stTextArea textarea:focus {
    border-color: var(--teal) !important;
    box-shadow: 0 0 0 1px var(--teal) !important;
}

div.stButton > button {
    width: 100%;
    min-height: 3rem;
    border-radius: 10px;
    border: 1px solid var(--teal);
    background: linear-gradient(100deg, #22e4f1, #18cbbd);
    color: #00171d;
    font-weight: 800;
    transition: .2s;
}
div.stButton > button:hover {
    color: #00171d;
    border-color: #a3ffff;
    filter: brightness(1.08);
}
div.stButton > button:focus {
    box-shadow: 0 0 0 2px #19e6f255;
}

.example-label {
    color: var(--teal);
    font-weight: 800;
    font-size: .77rem;
    letter-spacing: .08em;
    margin: 8px 0;
}

div[data-testid="stMetric"] {
    background: #061e29;
    border: 1px solid #12546a;
    border-radius: 12px;
    padding: 14px;
}
div[data-testid="stMetricLabel"] { color: #a4c7d8; }
div[data-testid="stMetricValue"] { color: var(--teal); }

.result {
    border: 1px solid #18d6d2;
    border-radius: 14px;
    padding: 22px;
    margin: 12px 0;
    background: linear-gradient(120deg, #06252d, #04131d);
}
.result.hate { border-color: #ff5874; }
.result.offensive { border-color: #ffb94f; }
.result.safe { border-color: #20f5c5; }

.result-kicker {
    font-size: .72rem;
    color: #9dbdca;
    letter-spacing: .13em;
    font-weight: 800;
}
.result-title {
    font: 700 1.7rem 'Space Grotesk', sans-serif;
    margin: 9px 0;
}
.hate .result-title { color: #ff7189; }
.offensive .result-title { color: #ffca70; }
.safe .result-title { color: #20f5c5; }
.result-copy {
    color: #b7d1dc;
    font-size: .88rem;
    line-height: 1.7;
}

.about-panel {
    border: 1px solid #087d94;
    background: linear-gradient(120deg, #061b27, #06252b);
    border-radius: 16px;
    padding: 22px;
    margin-top: 20px;
    box-shadow: 0 0 20px #00d9e80b;
}
.about-heading {
    color: #f1fbff;
    font: 700 1.08rem 'Space Grotesk', sans-serif;
    margin-bottom: 12px;
}
.about-heading span { color: var(--teal); }
.about-copy {
    color: #b5d0dc;
    font-size: .86rem;
    line-height: 1.9;
}
.about-feature {
    color: #b5d0dc;
    font-size: .82rem;
    line-height: 2.1;
}
.about-feature span { color: var(--teal); font-weight: 800; }

div[data-testid="stExpander"] {
    border: 1px solid #12546a;
    border-radius: 12px;
}
div[data-testid="stExpander"] summary { color: #eafaff; }
hr { border-color: #164353; }

.footer {
    text-align: center;
    padding: 22px 0 5px;
    color: #7fa3b4;
    font-size: .75rem;
    line-height: 1.8;
}
@media (max-width: 600px) {
    .block-container { padding: 1rem .7rem 2rem; }
    .brand-logo { width: 44px; height: 52px; }
    .brand-name { font-size: 1rem; }
    .brand-sub { font-size: .63rem; }
    .status-pill { font-size: .55rem; padding: 6px; }
    .hero { gap: 12px; padding: 17px 12px; }
    .hero-logo { width: 85px; }
    .hero-copy { font-size: .78rem; }
    .about-panel { padding: 15px; }
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# CUSTOM SHIELD LOGO
# ==========================================

logo = """
<svg class="brand-logo" viewBox="0 0 100 112"
     xmlns="http://www.w3.org/2000/svg" role="img"
     aria-label="Hate-Shield AI logo">
  <defs>
    <linearGradient id="hsBlue" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#8bffff"/>
      <stop offset="45%" stop-color="#19e6f2"/>
      <stop offset="100%" stop-color="#0875ff"/>
    </linearGradient>
  </defs>
  <path d="M50 3 L94 21 L89 65 Q80 91 50 109
           Q20 91 11 65 L6 21 Z"
        fill="#041722" stroke="url(#hsBlue)" stroke-width="4"/>
  <path d="M50 12 L83 26 L79 62 Q71 82 50 96
           Q29 82 21 62 L17 26 Z"
        fill="#06232e" stroke="#087dba" stroke-width="1.5"/>
  <path d="M63 27 L39 27 L27 40 L27 67 L42 80
           L42 58 L57 58 L57 77 L73 63 L73 41
           L60 41 L60 49 L43 49 L43 38 L63 38 Z"
        fill="url(#hsBlue)"/>
  <path d="M50 3 L50 12 M11 65 L21 62 M89 65 L79 62"
        stroke="#c2ffff" stroke-width="1.5" opacity=".8"/>
</svg>
"""

hero_logo = logo.replace("brand-logo", "hero-logo")

st.markdown(f"""
<div class="brand">
  {logo}
  <div>
    <div class="brand-name">HATE-SHIELD <span>AI</span></div>
    <div class="brand-sub">DETECT · ANALYSE · MAKE A DIFFERENCE</div>
  </div>
  <div class="status-pill">◉ AI TEXT DETECTOR</div>
</div>

<div class="hero">
  {hero_logo}
  <div>
    <h1>HATE-SHIELD <span>AI</span></h1>
    <div class="hero-tag">DETECT · ANALYSE · PROTECT</div>
    <div class="hero-copy">
      An AI-powered educational tool that classifies text as hate speech,
      offensive language or neither, helping explore safer and more
      respectful online conversations.
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ==========================================
# FEATURE CARDS
# ==========================================

st.markdown('<div class="section-title">✦ &nbsp; WHAT IT CAN DO</div>',
            unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""
    <div class="feature">
      <div class="feature-icon">🛡️</div>
      <div class="feature-name">Hate Speech Detection</div>
      <div class="feature-copy">
        Classify text patterns associated with hate speech.
      </div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="feature">
      <div class="feature-icon">⚠️</div>
      <div class="feature-name">Offensive Language</div>
      <div class="feature-copy">
        Identify text patterns associated with offensive language.
      </div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="feature">
      <div class="feature-icon">✓</div>
      <div class="feature-name">Three Classifications</div>
      <div class="feature-copy">
        Display the model's predicted category and class scores.
      </div>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# MESSAGE INPUT PANEL
# ==========================================

st.markdown(
    '<div class="section-title">01 / <span>ANALYSE A MESSAGE</span></div>',
    unsafe_allow_html=True,
)

with st.container(border=True):
    st.markdown(
        '<div class="input-label">▣ &nbsp; Enter text for analysis</div>'
        '<div class="input-help">Type or paste a message below to see '
        'the classification predicted by the model.</div>',
        unsafe_allow_html=True,
    )

    user_text = st.text_area(
        "Message to analyse",
        placeholder="Type or paste your message here...",
        height=145,
        max_chars=5000,
        label_visibility="collapsed",
        key="message_input",
    )

    st.caption("Maximum 5,000 characters. Avoid entering personal information.")

    left, right = st.columns([2, 1])

    with left:
        detect_clicked = st.button(
            "🛡️  DETECT MESSAGE  →",
            use_container_width=True,
        )

    with right:
        clear_clicked = st.button(
            "↻  CLEAR",
            use_container_width=True,
        )

    with st.expander("⚡ Try an example message"):
        st.caption("Choose an example to place it in the message box.")
        if st.button("Example 1 · Neutral message", use_container_width=True):
            st.session_state["message_input"] = (
                "I really enjoyed watching the cricket game today."
            )
            st.rerun()

        if st.button("Example 2 · Offensive wording", use_container_width=True):
            st.session_state["message_input"] = "You are an idiot."
            st.rerun()

        if st.button("Example 3 · Another neutral message", use_container_width=True):
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
            with st.spinner("Loading model and analysing text..."):
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
# DETECTION RESULT
# ==========================================

st.markdown(
    '<div class="section-title">02 / <span>DETECTION RESULT</span></div>',
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
        f'<div class="result-kicker">MODEL CLASSIFICATION</div>'
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
        "Model scores are estimated class probabilities, not guarantees "
        "of correctness. Context, sarcasm, slang and dataset bias can "
        "affect predictions. Human review is important."
    )
else:
    st.markdown(f"""
    <div class="result safe" style="text-align:center;padding:25px 12px;">
      <div style="font-size:2.4rem;color:#19e6f2;">{logo}</div>
      <div style="font-weight:800;color:#19e6f2;margin:8px 0;">
        Your result will appear here
      </div>
      <div class="result-copy">
        Enter a message and tap Detect Message to see its classification.
      </div>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# PERMANENT ABOUT BOX
# ==========================================

st.markdown(
    '<div class="section-title">03 / <span>ABOUT THE PROJECT</span></div>',
    unsafe_allow_html=True,
)

st.markdown("""
<div class="about-panel">
  <div class="about-heading">
    ⓘ &nbsp; ABOUT HATE-SHIELD AI
  </div>
  <div class="about-copy">
    Hate-Shield AI is a Class XII Artificial Intelligence capstone project
    that explores the classification of potentially harmful language in text.
    It uses a machine-learning pipeline combining TF-IDF text vectorization
    and Logistic Regression to classify messages into three categories:
    Hate Speech, Offensive Language, and Neither, based on labels in the
    supplied dataset.
  </div>
  <br>
  <div class="about-heading">
    <span style="color:#19e6f2;">✦</span> &nbsp; KEY FEATURES
  </div>
  <div class="about-feature">
    <span>✓</span> &nbsp; Three-category text classification<br>
    <span>✓</span> &nbsp; TF-IDF text feature extraction<br>
    <span>✓</span> &nbsp; Logistic Regression classifier<br>
    <span>✓</span> &nbsp; Interactive text analysis and result display<br>
    <span>✓</span> &nbsp; Class-score display and responsible-use notes
  </div>
  <br>
  <div class="about-copy">
    <b style="color:#19e6f2;">Important:</b>
    This is an educational prototype. Predictions may be incorrect because
    language depends on context and the training dataset may contain bias.
    The model should not be used as the sole basis for penalising people.
  </div>
</div>
""", unsafe_allow_html=True)

# ==========================================
# FOOTER
# ==========================================

st.markdown("""
<div class="footer">
  🛡️ HATE-SHIELD AI · CLASS XII ARTIFICIAL INTELLIGENCE CAPSTONE<br>
  Safer Internet. Kinder Communities.<br>
  Educational prototype · AI predictions require human judgement.
</div>
""", unsafe_allow_html=True)

