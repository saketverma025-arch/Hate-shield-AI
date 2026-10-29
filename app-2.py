import html
import re
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# ---------------- PAGE + PATHS ----------------
st.set_page_config(page_title="Hate-Shield AI", page_icon="🛡️", layout="wide")
BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "twitter_data.csv"
MODEL_PATH = BASE_DIR / "hate_speech_model.joblib"

# Self-contained vector shield logo. No separate image upload needed.
SHIELD_LOGO = """
<svg viewBox="0 0 100 112" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Hate-Shield AI logo">
 <defs>
  <linearGradient id="shieldGlow" x1="0" y1="0" x2="1" y2="1">
   <stop offset="0%" stop-color="#6ffff0"/><stop offset="48%" stop-color="#00e6d1"/><stop offset="100%" stop-color="#00a8ff"/>
  </linearGradient>
  <filter id="glow"><feGaussianBlur stdDeviation="2.5" result="blur"/><feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
 </defs>
 <path d="M50 5 L87 22 L83 68 Q76 91 50 105 Q24 91 17 68 L13 22 Z" fill="#021824" stroke="url(#shieldGlow)" stroke-width="4" filter="url(#glow)"/>
 <path d="M50 14 L78 27 L75 65 Q69 83 50 94 Q31 83 25 65 L22 27 Z" fill="#052c3a" stroke="#0ba9b8" stroke-width="1.5"/>
 <path d="M32 31 L43 31 L43 45 L57 45 L57 31 L68 31 L68 77 L57 77 L57 57 L43 57 L43 77 L32 77 Z" fill="url(#shieldGlow)"/>
 <path d="M50 15 L50 94" stroke="#8ffff4" stroke-opacity=".35" stroke-width="1"/>
</svg>
"""

st.markdown("""
<style>
:root { color-scheme: dark; }
.stApp {
 background: radial-gradient(ellipse at 50% 0%, #07313b 0%, #031720 44%, #020b12 100%);
 color: #eafcff;
}
[data-testid="stHeader"] { background: transparent; }
.block-container { max-width: 1250px; padding-top: 1.2rem; padding-bottom: 2rem; }
h1,h2,h3,h4,p,label { color: #eafcff; }
div[data-testid="stTextArea"] textarea {
 background:#031a25 !important; color:#eaffff !important;
 border:1px solid #00e5d2 !important; border-radius:12px !important;
 font-size:16px !important;
}
div[data-testid="stTextArea"] textarea:focus { box-shadow:0 0 0 1px #00e5d2,0 0 18px #00e5d222; }
.stButton > button {
 border-radius:10px; border:1px solid #00e5c7; min-height:44px;
 font-weight:800; color:#001b22;
 background:linear-gradient(90deg,#4dffd0,#00d9c8);
}
.stButton > button:hover { border-color:#9bfff0; box-shadow:0 0 18px #00e5d244; color:#001b22; }
.hero {
 background:linear-gradient(120deg,#03151e,#06303a 60%,#03141e);
 border:1px solid #00e5cf; border-radius:17px; padding:22px 24px;
 box-shadow:0 0 24px #00e5cf12; margin-bottom:18px;
}
.brand { display:flex; align-items:center; gap:12px; }
.brand svg { width:42px; height:48px; flex-shrink:0; }
.brand-name { font-size:22px; font-weight:900; letter-spacing:.3px; }
.ai { color:#25f5d1; }
.pill { display:inline-block; border:1px solid #00e5cf; color:#5ffff0;
 border-radius:24px; padding:7px 12px; font-size:10px; font-weight:800; letter-spacing:1px; }
.hero-title { font-size:clamp(30px,4vw,48px); font-weight:950; line-height:1.08; margin:12px 0 4px; }
.tagline { color:#38f5db; font-size:13px; font-weight:800; letter-spacing:1.4px; }
.hero-desc { color:#b8dbe2; max-width:700px; font-size:15px; }
.feature { background:#031923; border:1px solid #00a995; border-radius:12px; padding:14px; min-height:120px; }
.feature-title { color:#f0ffff; font-weight:850; margin-bottom:6px; }
.panel { background:linear-gradient(135deg,#041a24,#03131d); border:1px solid #00cbb8;
 border-radius:14px; padding:19px; margin:14px 0; }
.panel-title { font-size:20px; font-weight:900; margin-bottom:4px; }
.sub { color:#9cc9d2; font-size:13px; }
.guide { background:#041c26; border:1px solid #087d7d; border-radius:11px; padding:13px; min-height:112px; }
.dot { display:inline-block; width:11px; height:11px; border-radius:50%; margin-right:7px; }
.hate-dot { background:#ff647d; box-shadow:0 0 8px #ff647d66; }
.off-dot { background:#ffd45e; box-shadow:0 0 8px #ffd45e66; }
.safe-dot { background:#39f0b2; box-shadow:0 0 8px #39f0b266; }
.result { border:1px solid; border-radius:14px; padding:20px; background:#031821; }
.result.hate { border-color:#ff4d87; box-shadow:0 0 20px #ff4d8712; }
.result.offensive { border-color:#f7c94c; box-shadow:0 0 20px #f7c94c12; }
.result.neither { border-color:#26e9ae; box-shadow:0 0 20px #26e9ae12; }
.result.hate .result-title { color:#ff6eaa; }
.result.offensive .result-title { color:#ffda72; }
.result.neither .result-title { color:#4ff2bd; }
.result-title { font-size:27px; font-weight:900; margin:8px 0; }
.result-meta { color:#a8c9d0; font-size:11px; letter-spacing:.6px; font-weight:800; }
.score-box { background:#031720; border:1px solid #12515a; border-radius:9px; padding:10px 12px; margin:8px 0; }
.footer { text-align:center; color:#719ba4; font-size:11px; letter-spacing:1px; padding:20px 0 4px; }
div[data-testid="stAlert"] { border-radius:10px; }
</style>
""", unsafe_allow_html=True)

# ---------------- MODEL ----------------
LABELS = {0: "Hate Speech", 1: "Offensive Language", 2: "Neither"}
RESULTS = {
    0: {"style": "hate", "icon": "✹", "colour": "#ff4d87",
        "desc": "The text contains language the model associates with hate speech or targeting a group."},
    1: {"style": "offensive", "icon": "⚠", "colour": "#f7c94c",
        "desc": "The text may contain insulting, vulgar, or otherwise offensive wording."},
    2: {"style": "neither", "icon": "✓", "colour": "#26e9ae",
        "desc": "The model did not identify hate speech or offensive language in this text."},
}

def clean_tweet(text):
    text = html.unescape(str(text))
    text = re.sub(r"https?://\S+|www\.\S+", " URL ", text)
    text = re.sub(r"@\w+", " USER ", text)
    return re.sub(r"\s+", " ", text).strip()

def build_model():
    return Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2),
                                  max_features=50000, sublinear_tf=True, min_df=2)),
        ("classifier", LogisticRegression(max_iter=1500, class_weight="balanced",
                                          random_state=42)),
    ])

@st.cache_resource
def load_or_train_model():
    if MODEL_PATH.is_file():
        try:
            return joblib.load(MODEL_PATH)
        except Exception:
            pass
    if not DATASET_PATH.is_file():
        raise FileNotFoundError("twitter_data.csv is missing. Upload it beside app.py.")
    df = pd.read_csv(DATASET_PATH)
    if not {"tweet", "class"}.issubset(df.columns):
        raise ValueError("Dataset must contain columns named 'tweet' and 'class'.")
    df = df[["tweet", "class"]].dropna()
    df["class"] = pd.to_numeric(df["class"], errors="coerce")
    df = df.dropna(subset=["class"])
    df["class"] = df["class"].astype(int)
    df = df[df["class"].isin([0, 1, 2])]
    df["tweet"] = df["tweet"].apply(clean_tweet)
    df = df[df["tweet"].str.len() > 0]
    if df.empty or df["class"].nunique() < 2:
        raise ValueError("Dataset must contain examples from at least two classes.")
    model = build_model()
    model.fit(df["tweet"], df["class"])
    try:
        joblib.dump(model, MODEL_PATH)
    except Exception:
        pass
    return model

def predict_text(model, text):
    cleaned = clean_tweet(text)
    prediction = int(model.predict([cleaned])[0])
    probs = model.predict_proba([cleaned])[0]
    classes = model.named_steps["classifier"].classes_
    return prediction, {int(c): float(p) for c, p in zip(classes, probs)}

# ---------------- HEADER / FRONT PAGE ----------------
st.markdown(f"""
<div class="hero">
  <div style="display:flex;justify-content:space-between;align-items:center;gap:16px;flex-wrap:wrap;border-bottom:1px solid #12535b;padding-bottom:12px">
    <div class="brand">{SHIELD_LOGO}<div class="brand-name">Hate-Shield <span class="ai">AI</span></div></div>
    <div class="pill">◉ &nbsp; AI HATE SPEECH DETECTOR</div>
  </div>
  <div style="text-align:center;padding:18px 0 12px">
    <div style="width:100px;margin:auto">{SHIELD_LOGO}</div>
    <div class="hero-title">Hate-Shield <span class="ai">AI</span></div>
    <div class="tagline">Safer Internet. Kinder Communities.</div>
    <p class="hero-desc" style="margin:10px auto 0">
      Detect hate speech, offensive language and harmful content using machine learning.
    </p>
  </div>
</div>
""", unsafe_allow_html=True)

feature_cols = st.columns(3)
features = [
    ("🛡️", "Detect Hate Speech", "Find content the model associates with hateful or discriminatory language."),
    ("▣", "Identify Offensive Language", "Recognise insults, profanity and potentially offensive wording."),
    ("✓", "Keep the Internet Safer", "Encourage respectful, thoughtful online communication."),
]
for col, (icon, title, desc) in zip(feature_cols, features):
    with col:
        st.markdown(f'<div class="feature"><div class="feature-title">{icon} &nbsp; {title}</div><div class="sub">{desc}</div></div>', unsafe_allow_html=True)

# ---------------- INPUT PANEL ----------------
st.markdown('<div class="panel">', unsafe_allow_html=True)
st.markdown('<div class="panel-title">Analyze a Message</div><div class="sub">Enter a tweet or social media message to classify it.</div>', unsafe_allow_html=True)
message = st.text_area("Message", placeholder="Type your message here...", max_chars=5000,
                       height=110, key="message", label_visibility="collapsed")
count_col, _ = st.columns([1, 8])
with count_col:
    st.caption(f"{len(message)} / 5000")
analyze_col, clear_col, _ = st.columns([2, 1, 5])
with analyze_col:
    analyze = st.button("➤  Analyze Text", use_container_width=True, type="primary")
with clear_col:
    if st.button("↻  Clear", use_container_width=True):
        st.session_state["message"] = ""
        st.session_state.pop("last_result", None)
        st.rerun()
st.markdown('</div>', unsafe_allow_html=True)

# ---------------- GUIDE + ABOUT ----------------
guide_col, about_col = st.columns([1, 1])
with guide_col:
    st.markdown("""
    <div class="panel">
      <div class="panel-title">Classification Guide</div>
      <div class="sub" style="margin-bottom:12px">Three possible outcomes</div>
      <p><span class="dot hate-dot"></span><b>Hate Speech</b><br><span class="sub">Content classified as hateful or targeting a group.</span></p>
      <p><span class="dot off-dot"></span><b>Offensive Language</b><br><span class="sub">Insults, profanity, or other potentially offensive wording.</span></p>
      <p><span class="dot safe-dot"></span><b>Neither</b><br><span class="sub">Content not classified into either of the other categories.</span></p>
    </div>
    """, unsafe_allow_html=True)
with about_col:
    st.markdown("""
    <div class="panel">
      <div class="panel-title">About This Model</div>
      <div class="sub">Text is cleaned, converted into TF-IDF features, then classified using Logistic Regression.</div>
      <div style="border:1px solid #087d7d;border-radius:10px;padding:12px;margin-top:14px">
        <b>Responsible use</b>
        <p class="sub" style="margin:6px 0 0">This is a screening aid, not a final moderation decision. Review ambiguous cases with human judgement and avoid using a model score as the sole basis for action.</p>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ---------------- RESULTS ----------------
if analyze:
    if not message.strip():
        st.warning("Please type a message before analysing it.")
        st.session_state.pop("last_result", None)
    else:
        try:
            with st.spinner("Analysing text..."):
                model = load_or_train_model()
                prediction, scores = predict_text(model, message)
            st.session_state["last_result"] = {
                "prediction": prediction, "scores": scores, "message": message
            }
        except Exception as e:
            st.error(f"Could not run detection: {e}")

result = st.session_state.get("last_result")
if result:
    pred = result["prediction"]
    item = RESULTS.get(pred, RESULTS[2])
    score = result["scores"].get(pred, 0.0) * 100
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    result_left, result_right = st.columns([1.2, 1])
    with result_left:
        st.markdown(f"""
        <div class="result {item['style']}">
          <div class="result-meta">02 / MODEL RESULT &nbsp; · &nbsp; ✓ ANALYSIS COMPLETE</div>
          <div class="result-title">{item['icon']} &nbsp; {LABELS[pred]}</div>
          <p class="sub">{item['desc']}</p>
          <div style="margin-top:20px;font-weight:750">Model confidence score <span style="float:right">{score:.1f}%</span></div>
          <div style="background:#102c37;border-radius:20px;height:12px;margin-top:8px;overflow:hidden">
            <div style="width:{score:.1f}%;height:12px;background:{item['colour']};border-radius:20px"></div>
          </div>
          <p class="sub" style="margin-top:18px">This score is the model's estimated class probability, not a guarantee that the label is correct. Context, sarcasm, dialect and quoted language can affect results.</p>
        </div>
        """, unsafe_allow_html=True)
    with result_right:
        st.markdown('<div class="panel-title">Classification Guide</div>', unsafe_allow_html=True)
        for class_id in (0, 1, 2):
            dot_class = {0: "hate-dot", 1: "off-dot", 2: "safe-dot"}[class_id]
            st.markdown(f'<div class="score-box"><span class="dot {dot_class}"></span><b>{LABELS[class_id]}</b><br><span class="sub">{RESULTS[class_id]["desc"]}</span></div>', unsafe_allow_html=True)
        st.markdown('<div class="panel-title" style="margin-top:14px">About This Model</div><div class="sub">TF-IDF converts text into numeric features; Logistic Regression predicts one of the dataset labels.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="footer">HATE-SHIELD AI &nbsp; · &nbsp; CLASS XII ARTIFICIAL INTELLIGENCE CAPSTONE<br>DETECT · ANALYSE · MAKE A DIFFERENCE</div>', unsafe_allow_html=True)
