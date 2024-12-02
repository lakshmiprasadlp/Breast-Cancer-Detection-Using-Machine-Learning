import streamlit as st
import pickle
import numpy as np

# Load the saved model and scaler
with open("cancer_model.pkl", "rb") as f:
    model, scaler = pickle.load(f)

# App title
st.title("Breast Cancer Diagnosis Predictor")

# Input fields for user to provide feature values
st.header("Enter the following details:")
features = []

feature_names = [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
    "smoothness_mean", "compactness_mean", "concavity_mean", "concave points_mean",
    "symmetry_mean", "fractal_dimension_mean", "radius_se", "texture_se",
    "perimeter_se", "area_se", "smoothness_se", "compactness_se",
    "concavity_se", "concave points_se", "symmetry_se", "fractal_dimension_se",
    "radius_worst", "texture_worst", "perimeter_worst", "area_worst",
    "smoothness_worst", "compactness_worst", "concavity_worst", "concave points_worst",
    "symmetry_worst", "fractal_dimension_worst"
]

for name in feature_names:
    value = st.number_input(f"{name.replace('_', ' ').capitalize()}", min_value=0.0, step=0.01)
    features.append(value)

# Prediction
if st.button("Predict"):
    # Combine inputs into a single array
    features = np.array([features])

    # Scale features
    features_scaled = scaler.transform(features)

    # Predict using the model
    prediction = model.predict(features_scaled)[0]
    prediction_prob = model.predict_proba(features_scaled)[0]

    # Output prediction
    if prediction == 1:
        st.error(f"The tumor is predicted to be **Malignant** with a probability of {prediction_prob[1]*100:.2f}%.")
    else:
        st.success(f"The tumor is predicted to be **Benign** with a probability of {prediction_prob[0]*100:.2f}%.")

# Footer
st.caption("This prediction is for informational purposes only and not a substitute for professional medical advice.")
