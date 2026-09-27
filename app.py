import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Road Accident Severity Predictor", page_icon="🚧", layout="centered")

@st.cache_resource
def load_artifacts():
    model = joblib.load("accident_severity_model.pkl")
    encoders = joblib.load("encoders.pkl")
    feature_order = joblib.load("feature_order.pkl")
    return model, encoders, feature_order

model, encoders, feature_order = load_artifacts()
target_le = encoders["__target__"]

st.title("🚧 Road Accident Severity Predictor")
st.write(
    "This app predicts the likely **severity** of a road accident "
    "(Slight / Serious / Fatal) based on road, weather, vehicle and time conditions. "
    "Built with a Random Forest model trained on UK road accident data."
)

st.header("Enter accident conditions")

col1, col2 = st.columns(2)

with col1:
    month = st.selectbox("Month", list(encoders["Month"].classes_))
    day_of_week = st.selectbox("Day of Week", list(encoders["Day_of_Week"].classes_))
    hour = st.slider("Hour of Day", 0, 23, 12)
    year = st.number_input("Year", min_value=2015, max_value=2030, value=2021, step=1)
    speed_limit = st.selectbox("Speed Limit (mph)", [20, 30, 40, 50, 60, 70])
    num_vehicles = st.number_input("Number of Vehicles Involved", min_value=1, max_value=20, value=2)
    num_casualties = st.number_input("Number of Casualties", min_value=1, max_value=50, value=1)
    urban_rural = st.selectbox("Area Type", list(encoders["Urban_or_Rural_Area"].classes_))

with col2:
    light = st.selectbox("Light Conditions", list(encoders["Light_Conditions"].classes_))
    weather = st.selectbox("Weather Conditions", list(encoders["Weather_Conditions"].classes_))
    road_surface = st.selectbox("Road Surface Conditions", list(encoders["Road_Surface_Conditions"].classes_))
    road_type = st.selectbox("Road Type", list(encoders["Road_Type"].classes_))
    junction_control = st.selectbox("Junction Control", list(encoders["Junction_Control"].classes_))
    junction_detail = st.selectbox("Junction Detail", list(encoders["Junction_Detail"].classes_))
    vehicle_type = st.selectbox("Vehicle Type", list(encoders["Vehicle_Type"].classes_))
    hazard = st.selectbox("Carriageway Hazards", list(encoders["Carriageway_Hazards"].classes_))

if st.button("Predict Severity", type="primary"):
    raw = {
        "Month": month,
        "Day_of_Week": day_of_week,
        "Junction_Control": junction_control,
        "Junction_Detail": junction_detail,
        "Light_Conditions": light,
        "Road_Surface_Conditions": road_surface,
        "Road_Type": road_type,
        "Urban_or_Rural_Area": urban_rural,
        "Weather_Conditions": weather,
        "Vehicle_Type": vehicle_type,
        "Carriageway_Hazards": hazard,
        "Year": year,
        "Hour": hour,
        "Number_of_Vehicles": num_vehicles,
        "Number_of_Casualties": num_casualties,
        "Speed_limit": speed_limit,
    }

    row = {}
    cat_cols = ["Month", "Day_of_Week", "Junction_Control", "Junction_Detail",
                "Light_Conditions", "Road_Surface_Conditions", "Road_Type",
                "Urban_or_Rural_Area", "Weather_Conditions", "Vehicle_Type",
                "Carriageway_Hazards"]
    for c in cat_cols:
        row[c] = encoders[c].transform([raw[c]])[0]
    for c in ["Year", "Hour", "Number_of_Vehicles", "Number_of_Casualties", "Speed_limit"]:
        row[c] = raw[c]

    X = pd.DataFrame([row])[feature_order]

    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    label = target_le.inverse_transform([pred])[0]

    st.subheader("Prediction Result")
    color = {"Slight": "green", "Serious": "orange", "Fatal": "red"}.get(label, "blue")
    st.markdown(f"### Predicted Severity: :{color}[{label}]")

    proba_df = pd.DataFrame({
        "Severity": target_le.classes_,
        "Probability": proba
    }).sort_values("Probability", ascending=False)
    st.bar_chart(proba_df.set_index("Severity"))
    st.dataframe(proba_df, hide_index=True, use_container_width=True)

st.divider()
st.caption(
    "Model: Random Forest Classifier | Trained on Road_Accident_Data.csv | "
    "Test accuracy ≈ 69%. Note: the 'Slight' class dominates the dataset (~85% of records), "
    "so the model favors that class; class-weighting was used to improve detection of Serious/Fatal cases."
)
