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
    "BUN": ("Blood urea nitrogen", "BUN", "mmol/L", 1.86, 25.96, 6.09, 0.01, "%.2f"),
    "CREA": ("Creatinine", "CREA", "umol/L", 31.0, 442.2, 76.1, 0.1, "%.1f"),
    "HCT": ("Hematocrit", "HCT", "%", 18.0, 59.9, 38.4, 0.1, "%.1f"),
    "MCHC": ("Mean corpuscular hemoglobin concentration", "MCHC", "g/L", 252.0, 368.0, 320.0, 1.0, "%.0f"),
    "MCV": ("Mean corpuscular volume", "MCV", "fL", 56.3, 108.5, 93.3, 0.1, "%.1f"),
    "Na": ("Sodium", "Na", "mmol/L", 117.9, 148.9, 137.8, 0.1, "%.1f"),
    "NLR": ("Neutrophil-to-lymphocyte ratio", "NLR", "ratio", 0.296, 86.23, 7.391, 0.001, "%.3f"),
    "MLR": ("Monocyte-to-lymphocyte ratio", "MLR", "ratio", 0.011, 5.312, 0.747, 0.001, "%.3f"),
}


st.set_page_config(page_title="T2RF Risk Calculator", page_icon="🫁", layout="wide")

st.markdown(
    """
    <style>
    .main .block-container {
        max-width: 1080px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .hero {
        border: 1px solid #d6e0ee;
        border-radius: 8px;
        padding: 24px 28px;
        background: linear-gradient(135deg, #f8fbff 0%, #eef5ff 100%);
        margin-bottom: 20px;
    }
    .hero h1 {
        margin: 0 0 8px 0;
        color: #172033;
        font-size: 2.35rem;
        letter-spacing: 0;
    }
    .hero p {
        margin: 0;
        color: #475569;
        font-size: 1.02rem;
    }
    .panel {
        border: 1px solid #d6e0ee;
        border-radius: 8px;
        padding: 20px 22px;
        background: #ffffff;
    }
    .result-card {
        border: 1px solid #cbd5e1;
        border-radius: 8px;
        padding: 20px 22px;
        background: #f8fafc;
    }
    .result-number {
        font-size: 2.6rem;
        font-weight: 800;
        color: #0f172a;
        line-height: 1.1;
    }
    .risk-high {
        color: #b91c1c;
        font-weight: 800;
        font-size: 1.25rem;
    }
    .risk-low {
        color: #047857;
        font-weight: 800;
        font-size: 1.25rem;
    }
    .small-note {
        color: #64748b;
        font-size: 0.92rem;
    }
    div.stButton > button:first-child {
        width: 100%;
        border-radius: 6px;
        font-weight: 700;
        min-height: 3rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    model_path = next((path for path in MODEL_CANDIDATES if path.exists()), None)
    if model_path is None:
        raise FileNotFoundError("Cannot find xgb_model.pkl.")

    payload = joblib.load(model_path)
    model = payload["model"]
    threshold = float(payload.get("threshold", 0.5))
    return model, threshold


def make_input_frame(values: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame([[values[feature] for feature in FEATURES]], columns=FEATURES)


def robust_predict_probability(model, x: pd.DataFrame) -> float:
    try:
        return float(np.asarray(model.predict_proba(x)[:, 1], dtype=float)[0])
    except Exception:
        if hasattr(model, "named_steps") and "model" in model.named_steps:
            inner_model = model.named_steps["model"]
            values = x.astype(float).to_numpy()
            return float(np.asarray(inner_model.predict_proba(values)[:, 1], dtype=float)[0])
        raise


def render_input(feature: str) -> float:
    full_name, abbreviation, unit, min_value, max_value, default, step, number_format = FEATURE_INFO[feature]
    return float(
        st.number_input(
            f"{abbreviation} ({unit})",
            min_value=float(min_value),
            max_value=float(max_value),
            value=float(default),
            step=float(step),
            format=number_format,
            help=full_name,
        )
    )


st.markdown(
    """
    <div class="hero">
      <h1>T2RF Risk Calculator</h1>
      <p>XGBoost-based prediction of type 2 respiratory failure risk in elderly patients with acute exacerbation of COPD and CAD.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

left, right = st.columns([1.05, 0.95], gap="large")

with left:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Patient Laboratory Values")
    st.caption("Enter laboratory values and click Calculate Risk.")

    with st.form("risk_calculator_form"):
        values = {}
        col_a, col_b = st.columns(2)
        for index, feature in enumerate(FEATURES):
            with col_a if index % 2 == 0 else col_b:
                values[feature] = render_input(feature)

        submitted = st.form_submit_button("Calculate Risk", type="primary")

    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    st.subheader("Prediction Result")

    if submitted:
        model, threshold = load_model()
        x_input = make_input_frame(values)
        probability = robust_predict_probability(model, x_input)
        risk_group = "High risk" if probability >= threshold else "Low risk"

        st.markdown(f'<div class="result-number">{probability * 100:.1f}%</div>', unsafe_allow_html=True)
        st.progress(min(max(probability, 0.0), 1.0))

        if risk_group == "High risk":
            st.markdown('<div class="risk-high">High risk</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="risk-low">Low risk</div>', unsafe_allow_html=True)

        st.write(f"Decision threshold: `{threshold:.4f}`")
        with st.expander("Entered values"):
            st.dataframe(x_input, hide_index=True, use_container_width=True)
    else:
        st.markdown('<div class="result-number">--</div>', unsafe_allow_html=True)
        st.write("Click **Calculate Risk** after entering the laboratory values.")

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    """
    <p class="small-note">
    This calculator is intended for research reporting and clinical discussion only.
    It should not be used as the sole basis for diagnosis, treatment, or triage decisions.
    </p>
    """,
    unsafe_allow_html=True,
)
