"""Streamlit UI for the IMDb sentiment model.
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC = str(ROOT / "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
import common  # noqa: F401  (needed so joblib can unpickle common.SentimentModel)

MODEL_PATH = ROOT / "models" / "final_model.joblib"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def predict(model, text):
    """Return (label, confidence or None). label: 1 = positive, 0 = negative."""
    label = int(model.predict([text])[0])
    conf = None
    try:
        if hasattr(model, "predict_proba"):
            p = np.asarray(model.predict_proba([text]))[0]
            conf = float(p[label]) if len(p) > 1 else float(p)
        elif hasattr(model, "decision_function"):
            d = float(np.ravel(model.decision_function([text]))[0])
            p_pos = 1 / (1 + np.exp(-d))
            conf = p_pos if label == 1 else 1 - p_pos
    except Exception:
        conf = None  # confidence is optional; never break the UI over it
    return label, conf


st.set_page_config(page_title="IMDb Sentiment", page_icon="🎬", layout="centered")
st.title("🎬 IMDb Review Sentiment")
st.caption("Logistic regression on TF-IDF 1-2 grams with negation handling (~90.1% test accuracy)")

if not MODEL_PATH.exists():
    st.error(f"Model not found at {MODEL_PATH}. Run `python run_all.py` first.")
    st.stop()

model = load_model()

if "review" not in st.session_state:
    st.session_state.review = ""



review = st.text_area("Paste or type a movie review", key="review", height=180)

if st.button("Analyse", type="primary"):
    if not review.strip():
        st.warning("Please enter a review first.")
    else:
        label, conf = predict(model, review)
        if label == 1:
            st.success("😊 **POSITIVE** review")
        else:
            st.error("😞 **NEGATIVE** review")
        if conf is not None:
            st.progress(min(max(conf, 0.0), 1.0), text=f"Confidence: {conf:.1%}")
