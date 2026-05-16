from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
MODEL_CANDIDATES = [APP_DIR / "xgb_model.pkl", APP_DIR / "models" / "xgb_model.pkl"]

FEATURES = ["BUN", "CREA", "HCT", "MCHC", "MCV", "Na", "NLR", "MLR"]

FEATURE_INFO = {
    "BUN": ("Blood urea nitrogen (BUN)", "mmol/L", 1.86, 25.96, 6.09, 0.01, "%.2f"),
    "CREA": ("Creatinine (CREA)", "umol/L", 31.0, 442.2, 76.1, 0.1, "%.1f"),
    "HCT": ("Hematocrit (HCT)", "%", 18.0, 59.9, 38.4, 0.1, "%.1f"),
    "MCHC": ("Mean corpuscular hemoglobin concentration (MCHC)", "g/L", 252.0, 368.0, 320.0, 1.0, "%.0f"),
    "MCV": ("Mean corpuscular volume (MCV)", "fL", 56.3, 108.5, 93.3, 0.1, "%.1f"),
    "Na": ("Sodium (Na)", "mmol/L", 117.9, 148.9, 137.8, 0.1, "%.1f"),
    "NLR": ("Neutrophil-to-lymphocyte ratio (NLR)", "ratio", 0.296, 86.23, 7.391, 0.001, "%.3f"),
    "MLR": ("Monocyte-to-lymphocyte ratio (MLR)", "ratio", 0.011, 5.312, 0.747, 0.001, "%.3f"),
}


@st.cache_resource
def load_model():
    model_path = next((path for path in MODEL_CANDIDATES if path.exists()), None)
    if model_path is None:
        raise FileNotFoundError("Cannot find xgb_model.pkl.")
    payload = joblib.load(model_path)
    return payload["model"], float(payload.get("threshold", 0.5))


st.set_page_config(page_title="T2RF Risk Calculator", layout="centered")
st.title("T2RF Risk Calculator")
st.write(
    "Prediction of type 2 respiratory failure risk in elderly patients with acute exacerbation of COPD and CAD."
)

values = {}
col1, col2 = st.columns(2)
for i, feature in enumerate(FEATURES):
    label, unit, min_value, max_value, default, step, fmt = FEATURE_INFO[feature]
    with col1 if i % 2 == 0 else col2:
        values[feature] = st.number_input(
            f"{label} ({unit})",
            min_value=float(min_value),
            max_value=float(max_value),
            value=float(default),
            step=float(step),
            format=fmt,
        )

model, threshold = load_model()
x = pd.DataFrame([[values[feature] for feature in FEATURES]], columns=FEATURES)
probability = float(np.asarray(model.predict_proba(x)[:, 1], dtype=float)[0])
risk_group = "High risk" if probability >= threshold else "Low risk"

st.divider()
st.metric("Predicted probability of T2RF", f"{probability * 100:.1f}%")

if risk_group == "High risk":
    st.error(f"Risk group: {risk_group}")
else:
    st.success(f"Risk group: {risk_group}")

st.caption(f"Model threshold: {threshold:.4f}")
st.info(
    "This calculator is intended for research reporting and clinical discussion only. "
    "It should not be used as the sole basis for diagnosis, treatment, or triage decisions."
)
