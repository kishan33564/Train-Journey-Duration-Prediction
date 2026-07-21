import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ==========================
# Page setup
# ==========================
st.set_page_config(page_title="Train Journey Duration Predictor", page_icon="🚆", layout="wide")
st.title("🚆 Train Journey Duration Prediction")
st.write(
    "Enter journey details in the sidebar and get an instant prediction of total "
    "journey duration, plus a look at how the model performs against real data."
)

# ==========================
# Load model + encoder (cached so they load once per session)
# ==========================
@st.cache_resource
def load_artifacts():
    model = joblib.load("journey_duration_model.pkl")
    encoder = joblib.load("encoder.pkl")
    return model, encoder


@st.cache_data
def load_test_results():
    # Actual vs Predicted values on the held-out test set, produced at training time
    return pd.read_csv("test_results.csv")


model, encoder = load_artifacts()
test_results = load_test_results()

station_names = sorted(encoder.classes_)

# ==========================
# Sidebar: user inputs
# ==========================
st.sidebar.header("Journey Details")

distance = st.sidebar.number_input("Total Distance (KM)", min_value=1, max_value=5000, value=500)
stops = st.sidebar.number_input("Total Number of Stops", min_value=1, max_value=100, value=10)
start_station = st.sidebar.selectbox("Starting Station", station_names, index=0)
end_station = st.sidebar.selectbox("Ending Station", station_names, index=1)

predict_clicked = st.sidebar.button("Predict Journey Duration", type="primary")

# ==========================
# Prediction
# ==========================
predicted_duration = None

if predict_clicked:
    if start_station == end_station:
        st.error("Starting Station and Ending Station must be different.")
    else:
        start_enc = encoder.transform([start_station])[0]
        end_enc = encoder.transform([end_station])[0]

        user_data = pd.DataFrame({
            "Total_Distance": [distance],
            "Total_Stops": [stops],
            "Starting_Station": [start_enc],
            "Ending_Station": [end_enc],
        })

        predicted_duration = model.predict(user_data)[0]
        predicted_duration = max(predicted_duration, 0)

        hours = int(predicted_duration)
        minutes = int(round((predicted_duration - hours) * 60))

        st.success(f"### Predicted Journey Duration: **{predicted_duration:.2f} hours** (~{hours}h {minutes}m)")
        st.caption(f"{start_station} → {end_station} · {distance} km · {stops} stops")

# ==========================
# Visualization: Predicted vs Actual
# ==========================
st.divider()
st.subheader("Model Performance: Predicted vs Actual Journey Duration")

col1, col2 = st.columns([2, 1])

with col1:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(
        test_results["Actual_Duration"],
        test_results["Predicted_Duration"],
        alpha=0.4,
        color="#1f77b4",
        label="Test set journeys",
    )

    lims = [
        min(test_results["Actual_Duration"].min(), test_results["Predicted_Duration"].min()),
        max(test_results["Actual_Duration"].max(), test_results["Predicted_Duration"].max()),
    ]
    ax.plot(lims, lims, color="gray", linestyle="--", linewidth=1, label="Perfect prediction")

    if predicted_duration is not None:
        # Plot the user's new prediction against its own input distance context
        ax.scatter(
            [predicted_duration], [predicted_duration],
            color="red", s=140, marker="*", zorder=5,
            label="Your prediction",
        )

    ax.set_xlabel("Actual Journey Duration (hours)")
    ax.set_ylabel("Predicted Journey Duration (hours)")
    ax.set_title("Actual vs Predicted (Test Set)")
    ax.legend()
    st.pyplot(fig)

with col2:
    errors = test_results["Actual_Duration"] - test_results["Predicted_Duration"]
    mae = errors.abs().mean()
    rmse = np.sqrt((errors ** 2).mean())
    ss_res = (errors ** 2).sum()
    ss_tot = ((test_results["Actual_Duration"] - test_results["Actual_Duration"].mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot

    st.metric("MAE (hours)", f"{mae:.2f}")
    st.metric("RMSE (hours)", f"{rmse:.2f}")
    st.metric("R² Score", f"{r2:.3f}")
    st.caption(f"Evaluated on {len(test_results)} held-out journeys.")

with st.expander("View sample of test predictions"):
    st.dataframe(
        test_results[
            ["Starting_Station", "Ending_Station", "Total_Distance", "Total_Stops",
             "Actual_Duration", "Predicted_Duration"]
        ].head(20),
        use_container_width=True,
    )
