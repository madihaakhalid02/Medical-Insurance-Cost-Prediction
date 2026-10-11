# app.py
# Medical Insurance Cost Prediction - Streamlit App (Phase 7)

import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression

# ------------------------------------------------------------------
# Page configuration
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Medical Insurance Cost Predictor",
    page_icon="🏥",
    layout="centered",
)


# ------------------------------------------------------------------
# Load data, preprocess, and train the model (cached so it only runs once)
# ------------------------------------------------------------------
@st.cache_resource
def load_and_train_model():
    # Load the dataset (insurance.csv must be in the same directory)
    df = pd.read_csv("insurance.csv")

    # Map binary categorical columns to numbers
    df["sex"] = df["sex"].map({"male": 1, "female": 0})
    df["smoker"] = df["smoker"].map({"yes": 1, "no": 0})

    # One-hot encode 'region' (drop_first=True avoids the dummy variable trap)
    df = pd.get_dummies(df, columns=["region"], drop_first=True, dtype=int)

    # Split into features and target
    X = df.drop("charges", axis=1)
    y = df["charges"]

    # Train the Linear Regression model
    model = LinearRegression()
    model.fit(X, y)

    # Return the model and the exact column order used for training
    return model, list(X.columns)


try:
    model, feature_columns = load_and_train_model()
except FileNotFoundError:
    st.error("❌ 'insurance.csv' was not found. Place it in the same folder as app.py.")
    st.stop()


# ------------------------------------------------------------------
# Title and description
# ------------------------------------------------------------------
st.title("🏥 Medical Insurance Cost Predictor")
st.write(
    "Enter the details in the sidebar and click **Predict Charges** "
    "to estimate the yearly medical insurance cost."
)


# ------------------------------------------------------------------
# Sidebar: user inputs
# ------------------------------------------------------------------
st.sidebar.header("Enter Your Details")

age = st.sidebar.slider("Age", min_value=18, max_value=64, value=30)
sex = st.sidebar.selectbox("Sex", ["male", "female"])
bmi = st.sidebar.slider("BMI", min_value=15.0, max_value=55.0, value=25.0, step=0.1)
children = st.sidebar.slider("Number of Children", min_value=0, max_value=5, value=0)
smoker = st.sidebar.selectbox("Smoker", ["no", "yes"])
region = st.sidebar.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])


# ------------------------------------------------------------------
# Build the input row exactly the way the training data was processed
# ------------------------------------------------------------------
def build_input_dataframe():
    # Start with every training column set to 0
    input_df = pd.DataFrame(0, index=[0], columns=feature_columns)

    input_df["age"] = age
    input_df["sex"] = 1 if sex == "male" else 0
    input_df["bmi"] = bmi
    input_df["children"] = children
    input_df["smoker"] = 1 if smoker == "yes" else 0

    # One-hot region: 'northeast' is the dropped baseline, so all its columns stay 0
    region_col = f"region_{region}"
    if region_col in input_df.columns:
        input_df[region_col] = 1

    return input_df


# ------------------------------------------------------------------
# Prediction
# ------------------------------------------------------------------
if st.button("Predict Charges"):
    input_df = build_input_dataframe()
    prediction = model.predict(input_df)[0]

    # Linear regression can occasionally go below zero; charges can't be negative
    prediction = max(prediction, 0)

    st.subheader("Estimated Insurance Charges")
    st.success(f"💰 ${prediction:,.2f}")

    with st.expander("Show input summary"):
        st.write(
            {
                "Age": age,
                "Sex": sex,
                "BMI": bmi,
                "Children": children,
                "Smoker": smoker,
                "Region": region,
            }
        )
