
import streamlit as st
import pandas as pd
import pickle
import os

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

st.title("Credit Card Fraud Detection System")

st.write(
    "A machine learning system that identifies "
    "potentially fraudulent credit card transactions."
)

st.success("Application started successfully!")

# Load the trained machine learning model
with open("fraud_detection_model.pkl", "rb") as file:
    model = pickle.load(file)

# Load the model feature names
with open("model_features.pkl", "rb") as file:
    features = pickle.load(file)

st.success("Random Forest model loaded successfully!")

st.write("Machine Learning Algorithm: Random Forest")
st.write("Number of decision trees: 100")
st.write("Number of input features:", len(features))

st.header("Upload Transaction Data")

uploaded_file = st.file_uploader(
    "Choose a transaction CSV file",
    type=["csv"]
)

if uploaded_file is not None:
    uploaded_data = pd.read_csv(uploaded_file)

    st.success("CSV file uploaded successfully!")

    st.write("Uploaded transaction data:")
    st.dataframe(uploaded_data.head())

    st.write("Total transactions uploaded:", len(uploaded_data))

    # Check whether all required features exist
    missing_features = [
        feature for feature in features
        if feature not in uploaded_data.columns
    ]

    if missing_features:
        st.error("Your CSV file is missing required features.")
        st.write("Missing features:", missing_features)

    else:
        st.success("All 30 required features are available!")

        # Arrange features in the correct order
        transaction_data = uploaded_data[features]

        st.write("Validated transaction data:")
        st.dataframe(transaction_data.head())

        if st.button("Detect Fraud"):

            # Make fraud predictions
            predictions = model.predict(transaction_data)

            # Calculate fraud probabilities
            probabilities = model.predict_proba(transaction_data)[:, 1]

            # Add predictions to uploaded data
            results = uploaded_data.copy()

            results["Fraud_Prediction"] = predictions

            results["Fraud_Probability"] = probabilities

            results["Transaction_Status"] = results[
                "Fraud_Prediction"
            ].map({
                0: "Legitimate",
                1: "Potential Fraud"
            })

            st.session_state["fraud_results"] = results

            st.success("Fraud detection completed successfully!")

# Display prediction results
if "fraud_results" in st.session_state:

    results = st.session_state["fraud_results"]

    st.header("Fraud Detection Results")

    total_transactions = len(results)

    fraud_count = (
        results["Fraud_Prediction"] == 1
    ).sum()

    legitimate_count = (
        results["Fraud_Prediction"] == 0
    ).sum()

    # Display summary statistics
    col1, col2, col3 = st.columns(3)

    col1.metric("Total Transactions", total_transactions)

    col2.metric("Fraud Detected", fraud_count)

    col3.metric("Legitimate Transactions", legitimate_count)

    # Display transaction results
    st.subheader("Transaction Prediction Details")

    st.dataframe(results)

    # Convert results into CSV format
    csv_results = results.to_csv(index=False).encode("utf-8")

    # Download results button
    st.download_button(
        label="Download Fraud Detection Results",
        data=csv_results,
        file_name="fraud_detection_results.csv",
        mime="text/csv"
    )

    st.success("Your fraud detection report is ready to download!")
