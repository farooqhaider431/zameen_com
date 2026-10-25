import json
import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

BASE = Path(__file__).parent
MODEL_FILE = BASE / "zameen_price_model.pkl"
META_FILE = BASE / "model_meta.json"

st.set_page_config(page_title="Zameen Property Price Predictor", page_icon="🏠", layout="centered")


@st.cache_resource
def load_model():
    with open(MODEL_FILE, "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_meta():
    with open(META_FILE) as f:
        return json.load(f)


def pretty_price(pkr: float) -> str:
    """Format a price in PKR with Crore / Lakh wording."""
    if pkr >= 1e7:
        return f"{pkr / 1e7:,.2f} Crore"
    if pkr >= 1e5:
        return f"{pkr / 1e5:,.2f} Lakh"
    return f"{pkr:,.0f}"


model = load_model()
meta = load_meta()

st.title("🏠 Zameen.com Property Price Predictor")
st.caption(
    f"Model: **{meta['algorithm']}** trained on {meta['n_listings']:,} Zameen.com sale listings "
    f"(10-fold cross-validation RMSE ≈ PKR {meta['cv_rmse'] / 1e6:,.1f} million). "
    "Predicts the **sale price** only (not rent)."
)

col1, col2 = st.columns(2)
with col1:
    city = st.selectbox("City", sorted(meta["locations_by_city"].keys()), index=0)
    location = st.selectbox("Location / Society", meta["locations_by_city"][city])
    property_type = st.selectbox(
        "Property type",
        meta["property_types"],
        index=meta["property_types"].index("House") if "House" in meta["property_types"] else 0,
    )
with col2:
    unit = st.radio("Area unit", ["Marla", "Kanal"], horizontal=True)
    default_area = 10.0 if unit == "Marla" else 1.0
    max_area = meta["area_marla_range"][1] / (20 if unit == "Kanal" else 1)
    area_value = st.number_input(f"Area ({unit})", min_value=0.5, max_value=float(max_area),
                                 value=default_area, step=0.5)
    bedrooms = st.number_input("Bedrooms", min_value=0, max_value=10, value=4, step=1)
    baths = st.number_input("Bathrooms", min_value=0, max_value=10, value=4, step=1)

area_marla = area_value * (20 if unit == "Kanal" else 1)

if st.button("Predict price", type="primary"):
    row = pd.DataFrame(
        [{
            "area_marla": area_marla,
            "bedrooms": bedrooms,
            "baths": baths,
            "property_type": property_type,
            "city": city,
            "location": location,
        }]
    )
    price = float(model.predict(row)[0])
    st.success(f"Estimated sale price: **PKR {price:,.0f}**  ({pretty_price(price)})")
    st.info(
        f"{area_marla:g} Marla {property_type.lower()} in {location}, {city}. "
        f"Typical model error is roughly PKR {meta['cv_rmse'] / 1e6:,.1f} million (RMSE); "
        "it is larger for very expensive properties and smaller for ordinary ones."
    )

with st.expander("About this app"):
    st.write(
        "Course project for Machine Learning Fundamentals (AIC354), COMSATS University Islamabad, Lahore Campus. "
        "Five regression algorithms (Ridge, ElasticNet, Extra Trees, HistGradientBoosting, XGBoost) were compared "
        "with 10-fold cross-validation and the best one is used here. 1 Kanal = 20 Marla."
    )
