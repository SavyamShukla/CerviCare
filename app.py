import streamlit as st
import joblib
import numpy as np
import pandas as pd

# --- LOAD THE TRAINED MODELS ---
# Use st.cache_resource to load the model and scaler only once
@st.cache_resource
def load_model_and_scaler():
    """Loads the saved model and scaler from disk."""
    model = joblib.load("cervical_cancer_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_model_and_scaler()

# --- IMPORTANT: DEFINE YOUR FEATURE NAMES ---
# The order of these names MUST EXACTLY MATCH the order of features
# your model was trained on.
# I'm using a plausible list based on common datasets.
# Please REPLACE this with your actual feature names.
FEATURE_NAMES = [
    'Age', 'Number of sexual partners', 'First sexual intercourse',
    'Num of pregnancies', 'Smokes', 'Smokes (years)', 'Smokes (packs/year)',
    'Hormonal Contraceptives', 'Hormonal Contraceptives (years)', 'IUD',
    'IUD (years)', 'STDs', 'STDs (number)', 'STDs:condylomatosis',
    'STDs:cervical condylomatosis', 'STDs:vaginal condylomatosis',
    'STDs:vulvo-perineal condylomatosis', 'STDs:syphilis',
    'STDs:pelvic inflammatory disease', 'STDs:genital herpes',
    'STDs:molluscum contagiosum', 'STDs:AIDS', 'STDs:HIV',
    'STDs:Hepatitis B', 'STDs:HPV', 'STDs: Number of diagnosis',
    'Dx:CIN', 'Dx:HPV'
]


# --- USER INTERFACE (UI) ---
st.set_page_config(page_title="Cervical Cancer Prediction", layout="wide")

# App Title
st.title("👩‍⚕ Cervical Cancer Prediction App")

# App Description
st.markdown("""
This app uses a machine learning model to predict the likelihood of cervical cancer based on patient data. 
Please enter the patient's information in the sidebar to get a prediction.

*Disclaimer:* This is a tool for educational purposes only and is *not a substitute for professional medical advice*.
""")

# --- SIDEBAR FOR USER INPUT ---
st.sidebar.header("Patient Input Features")
st.sidebar.markdown("Please provide the following information:")

# Create a dictionary to hold user inputs
input_data = {}

# Use a loop to create number inputs for each feature
for feature in FEATURE_NAMES:
    # Replace underscores with spaces for better readability
    label = feature.replace('_', ' ').title()
    input_data[feature] = st.sidebar.number_input(
        label,
        min_value=0.0,
        max_value=100.0, # Adjust max_value if necessary
        value=0.0,       # Default value
        step=0.1
    )

# --- PREDICTION LOGIC ---
if st.sidebar.button("Predict"):
    # 1. Convert the input dictionary to a pandas DataFrame
    input_df = pd.DataFrame([input_data])

    # 2. Ensure the column order is the same as the training data
    input_df = input_df[FEATURE_NAMES]

    # 3. Scale the user's input using the loaded scaler
    scaled_input = scaler.transform(input_df)

    # 4. Make a prediction using the loaded model
    prediction = model.predict(scaled_input)
    prediction_proba = model.predict_proba(scaled_input)

    # --- DISPLAY THE RESULT ---
    st.subheader("Prediction Result")
    
    # Get the probability of the positive class (Cancer)
    probability_cancer = prediction_proba[0][1] * 100

    if prediction[0] == 1:
        st.error(f"*Result: Positive for Cancer*")
        st.warning(f"*Probability:* {probability_cancer:.2f}%")
        st.markdown("Immediate consultation with a healthcare professional is strongly recommended.")
    else:
        st.success(f"*Result: Negative for Cancer*")
        st.info(f"*Probability of Cancer:* {probability_cancer:.2f}%")
        st.markdown("Regular check-ups are still advised for maintaining good health.")