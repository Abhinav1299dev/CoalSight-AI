
import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

from src.preprocess import load_data, preprocess_data, split_data


# =========================================================
# CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "model.pkl"

st.set_page_config(
    page_title="Coal Analytics | Production Predictor",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# PROFESSIONAL STYLING
# =========================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

    :root {
        --ink: #14251f;
        --muted: #687a73;
        --line: #e1e9e4;
        --surface: #ffffff;
        --canvas: #f4f7f5;
        --green: #13795b;
        --green-dark: #0e5e46;
        --green-soft: #e8f5ee;
    }

    html, body, [class*="css"], .stApp {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at 88% 0%, rgba(19,121,91,.07), transparent 26%),
            var(--canvas);
        color: var(--ink);
    }

    .block-container {
        max-width: 1440px;
        padding-top: 1.6rem;
        padding-bottom: 2.5rem;
    }

    [data-testid="stSidebar"] {
        background: #13251f;
        border-right: 1px solid rgba(255,255,255,.08);
    }

    [data-testid="stSidebar"] * {
        color: #e8f1ec;
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #b6c9bf;
        line-height: 1.65;
    }

    [data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,.14);
    }

    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #ffffff;
        font-family: 'Manrope', sans-serif;
        letter-spacing: -.02em;
    }

    .hero {
        position: relative;
        overflow: hidden;
        background: linear-gradient(115deg, #13251f 0%, #1b4938 58%, #17694e 100%);
        padding: 34px 38px;
        border-radius: 22px;
        margin: 4px 0 28px 0;
        color: #ffffff;
        box-shadow: 0 16px 36px rgba(19, 54, 40, .12);
    }

    .hero:after {
        content: "";
        position: absolute;
        width: 250px;
        height: 250px;
        border: 1px solid rgba(255,255,255,.12);
        border-radius: 50%;
        right: -65px;
        top: -105px;
        box-shadow: 0 0 0 28px rgba(255,255,255,.035),
                    0 0 0 58px rgba(255,255,255,.025);
        pointer-events: none;
    }

    .eyebrow {
        display: inline-block;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: .13em;
        text-transform: uppercase;
        color: #bcebd4;
        background: rgba(255,255,255,.09);
        border: 1px solid rgba(255,255,255,.14);
        border-radius: 999px;
        padding: 6px 10px;
        margin-bottom: 15px;
    }

    .hero h1 {
        font-family: 'Manrope', sans-serif;
        font-size: clamp(27px, 3vw, 38px);
        font-weight: 800;
        letter-spacing: -.045em;
        margin: 0 0 10px 0;
        color: #ffffff;
        line-height: 1.15;
    }

    .hero p {
        color: #d3e4da;
        font-size: 15px;
        max-width: 720px;
        margin: 0;
        line-height: 1.75;
    }

    .section-heading {
        font-family: 'Manrope', sans-serif;
        font-size: 22px;
        font-weight: 800;
        letter-spacing: -.035em;
        color: var(--ink);
        margin: 10px 0 5px 0;
    }

    .section-description {
        color: var(--muted);
        font-size: 14px;
        margin-bottom: 18px;
        line-height: 1.6;
    }

    .result-card {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 24px 25px;
        min-height: 142px;
        box-shadow: 0 5px 18px rgba(20, 37, 31, .035);
    }

    .result-label {
        font-size: 11px;
        font-weight: 700;
        letter-spacing: .1em;
        text-transform: uppercase;
        color: var(--muted);
        margin-bottom: 13px;
    }

    .result-value {
        font-family: 'Manrope', sans-serif;
        font-size: clamp(25px, 3vw, 34px);
        line-height: 1.2;
        font-weight: 800;
        letter-spacing: -.04em;
        color: var(--ink);
        overflow-wrap: anywhere;
    }

    .result-note {
        color: var(--muted);
        font-size: 12px;
        margin-top: 11px;
        line-height: 1.5;
    }

    div[data-testid="stForm"] {
        background: rgba(255,255,255,.92);
        border: 1px solid var(--line);
        border-radius: 20px;
        padding: 26px 28px 22px 28px;
        box-shadow: 0 8px 26px rgba(20, 37, 31, .04);
    }

    [data-testid="stTextInput"] label,
    [data-testid="stSelectbox"] label,
    [data-testid="stNumberInput"] label {
        font-weight: 600;
        color: #263b32;
        font-size: 13px;
    }

    [data-testid="stTextInput"] input,
    [data-testid="stNumberInput"] input {
        border-radius: 10px;
        border-color: #d6e2da;
        min-height: 44px;
        background: #fbfdfb;
    }

    [data-testid="stSelectbox"] div[data-baseweb="select"] > div {
        border-radius: 10px;
        border-color: #d6e2da;
        background: #fbfdfb;
        min-height: 44px;
    }

    div.stButton > button[kind="primary"],
    div[data-testid="stFormSubmitButton"] button {
        background: var(--green);
        border: 1px solid var(--green);
        color: white;
        font-weight: 700;
        min-height: 48px;
        border-radius: 11px;
        transition: all .18s ease;
        box-shadow: 0 5px 12px rgba(19,121,91,.13);
    }

    div.stButton > button[kind="primary"]:hover,
    div[data-testid="stFormSubmitButton"] button:hover {
        background: var(--green-dark);
        border-color: var(--green-dark);
        color: white;
        transform: translateY(-1px);
    }

    [data-testid="stDownloadButton"] button {
        border-radius: 10px;
        min-height: 44px;
        font-weight: 700;
        border-color: #cbdcd1;
    }

    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 17px 18px;
        box-shadow: 0 4px 14px rgba(20, 37, 31, .025);
    }

    [data-testid="stMetricLabel"] {
        color: var(--muted);
        font-weight: 600;
    }

    [data-testid="stMetricValue"] {
        color: var(--ink);
        font-family: 'Manrope', sans-serif;
        font-weight: 800;
        overflow-wrap: anywhere;
    }

    [data-testid="stAlert"] {
        border-radius: 12px;
    }

    h2, h3 {
        font-family: 'Manrope', sans-serif;
        letter-spacing: -.025em;
        color: var(--ink);
    }

    .footer {
        text-align: center;
        color: #7b8b83;
        font-size: 12px;
        padding: 16px 0 4px 0;
    }

    #MainMenu, footer {
        visibility: hidden;
    }

    @media (max-width: 768px) {
        .block-container { padding: 1rem 1rem 2rem 1rem; }
        .hero { padding: 26px 22px; border-radius: 16px; }
        div[data-testid="stForm"] { padding: 20px 17px; }
        .result-card { padding: 20px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# MODEL AND DATA LOADING
# =========================================================

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    with MODEL_PATH.open("rb") as file:
        return pickle.load(file)


@st.cache_data
def load_training_columns():
    df = load_data()
    df = preprocess_data(df)
    X, _ = split_data(df)

    return X.columns.tolist()


try:
    model = load_model()
    training_columns = load_training_columns()

except Exception:
    st.error(
        "The application could not load its model or training data. "
        "Check the model path and preprocessing configuration."
    )
    st.stop()


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.markdown("## ⛏️ Coal Analytics")
    st.caption("MACHINE LEARNING WORKSPACE")

    st.divider()

    st.markdown("### Platform Overview")
    st.write(
        "A machine-learning application designed to estimate "
        "coal production from mine ownership, resource type, "
        "state and geographical coordinates."
    )

    st.markdown("### Input Features")

    st.markdown(
        """
        - State / Union Territory
        - Coal or lignite classification
        - Ownership category
        - Latitude and longitude
        """
    )

    st.divider()

    st.markdown("### Model Disclaimer")
    st.caption(
        "Predictions depend on the training data and model "
        "quality. They are estimates, not verified production "
        "records or official forecasts."
    )

    st.markdown("---")
    st.caption("Coal Analytics  •  ML Research Project")


# =========================================================
# HERO HEADER
# =========================================================

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">Mining intelligence · Predictive analytics</div>
        <h1>Coal Production Predictor</h1>
        <p>
            Estimate mine production using a machine learning model and key
            location, resource classification, and ownership features.
            Enter the mine details below to generate an estimate.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# INPUT FORM
# =========================================================

st.markdown(
    '<div class="section-heading">Mine Information</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-description">'
    "Enter the mine's core characteristics. Fields are grouped to make data entry clear and consistent."
    '</div>',
    unsafe_allow_html=True,
)

with st.form("prediction_form"):
    col1, col2 = st.columns(2, gap="large")

    with col1:
        state = st.text_input(
            "State / Union Territory",
            placeholder="e.g. Maharashtra",
            help="Enter a state name consistent with the training dataset.",
        )

        coal_type = st.selectbox(
            "Resource Classification",
            ["Coal", "Lignite"],
        )

        ownership = st.selectbox(
            "Ownership Type",
            ["Govt Owned", "Private"],
        )

    with col2:
        latitude = st.number_input(
            "Latitude",
            min_value=-90.0,
            max_value=90.0,
            value=20.0,
            step=0.0001,
            format="%.4f",
        )

        longitude = st.number_input(
            "Longitude",
            min_value=-180.0,
            max_value=180.0,
            value=78.0,
            step=0.0001,
            format="%.4f",
        )

        st.caption(
            "Coordinates should correspond to the mine's location."
        )

    submitted = st.form_submit_button(
        "Run Production Prediction  →",
        use_container_width=True,
    )


# =========================================================
# PREDICTION
# =========================================================

if submitted:
    if not state.strip():
        st.warning(
            "Please enter a State / Union Territory before continuing."
        )

    else:
        try:
            # Keep the column names consistent with the training data.
            input_data = pd.DataFrame(
                [
                    {
                        "State/UT Name": state.strip(),
                        "Coal/Lignite": coal_type,
                        "Govt Owned/Private": ownership,
                        "Latitude ": latitude,
                        "Longitude ": longitude,
                    }
                ]
            )

            # One-hot encode categorical features.
            input_encoded = pd.get_dummies(input_data)

            # Match the exact training feature structure.
            input_encoded = input_encoded.reindex(
                columns=training_columns,
                fill_value=0,
            )

            # Predict using the loaded model.
            prediction = float(model.predict(input_encoded)[0])

            if not pd.notna(prediction):
                raise ValueError("The model returned an invalid prediction.")

            # IMPORTANT:
            # Do not multiply by 45 unless this is the documented
            # inverse transformation used during model training.
            production = prediction

            # These thresholds are examples. Adjust them to match
            # the actual target unit and project requirements.
            if production < 10:
                production_level = "Low"
                interpretation = (
                    "The predicted value falls within the low "
                    "production band configured for this application."
                )
            elif production < 25:
                production_level = "Medium"
                interpretation = (
                    "The predicted value falls within the medium "
                    "production band configured for this application."
                )
            else:
                production_level = "High"
                interpretation = (
                    "The predicted value falls within the high "
                    "production band configured for this application."
                )

            st.divider()

            st.markdown(
                '<div class="section-heading">Prediction Results</div>',
                unsafe_allow_html=True,
            )

            st.markdown(
                '<div class="section-description">'
                'Results generated from the supplied mine information.'
                '</div>',
                unsafe_allow_html=True,
            )

            # -------------------------------------------------
            # RESULT CARDS
            # -------------------------------------------------

            result_col1, result_col2 = st.columns(2, gap="large")

            with result_col1:
                st.markdown(
                    f"""
                    <div class="result-card">
                        <div class="result-label">
                            PREDICTED PRODUCTION
                        </div>
                        <div class="result-value">
                            {production:,.2f}
                        </div>
                        <div class="result-note">
                            Model output in the target's training unit
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with result_col2:
                st.markdown(
                    f"""
                    <div class="result-card">
                        <div class="result-label">
                            PRODUCTION CATEGORY
                        </div>
                        <div class="result-value">
                            {production_level}
                        </div>
                        <div class="result-note">
                            Based on the configured thresholds
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # -------------------------------------------------
            # INPUT SUMMARY
            # -------------------------------------------------

            st.markdown("### Prediction Context")

            summary_col1, summary_col2, summary_col3 = st.columns(3)

            with summary_col1:
                st.metric("State / UT", state.strip())

            with summary_col2:
                st.metric("Resource", coal_type)

            with summary_col3:
                st.metric("Ownership", ownership)

            location_col1, location_col2 = st.columns(2)

            with location_col1:
                st.metric("Latitude", f"{latitude:.4f}")

            with location_col2:
                st.metric("Longitude", f"{longitude:.4f}")

            # -------------------------------------------------
            # INTERPRETATION
            # -------------------------------------------------

            st.markdown("### Model Interpretation")
            st.info(interpretation)

            st.caption(
                "This result is a model-generated estimate. "
                "Its reliability depends on the quality and coverage "
                "of the training dataset. It should not be treated "
                "as an official production figure."
            )

            # -------------------------------------------------
            # EXPORT RESULT
            # -------------------------------------------------

            report = pd.DataFrame(
                [
                    {
                        "State/UT Name": state.strip(),
                        "Coal/Lignite": coal_type,
                        "Ownership": ownership,
                        "Latitude": latitude,
                        "Longitude": longitude,
                        "Predicted Production": production,
                        "Production Category": production_level,
                    }
                ]
            )

            st.download_button(
                label="Download Prediction Report · CSV",
                data=report.to_csv(index=False).encode("utf-8"),
                file_name="coal_production_prediction.csv",
                mime="text/csv",
                use_container_width=True,
            )

        except Exception:
            st.error(
                "Prediction failed. Verify the input features, "
                "model file, preprocessing steps and target transformation."
            )
            st.caption(
                "For debugging, check the application logs and terminal "
                "for the underlying exception."
            )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.markdown(
    """
    <div class="footer">
        COAL ANALYTICS &nbsp;•&nbsp;
        Machine Learning for Production Estimation
    </div>
    """,
    unsafe_allow_html=True,
)
