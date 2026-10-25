
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Page settings
st.set_page_config(
    page_title="AI Payment Safety",
    page_icon="🛡️",
    layout="centered"
)

# Load trained model
MODEL_PATH = Path(__file__).parent / "payment_safety_model.pkl"

st.title("🛡️ AI Payment Safety")
st.caption("Machine Learning Based Credit Card Fraud Detection")

st.info(
    "This is an experimental project using a European credit-card "
    "dataset. It is not connected to real bank or UPI payments."
)

if not MODEL_PATH.exists():
    st.error("Model file not found. Please check payment_safety_model.pkl.")
    st.stop()

try:
    model = joblib.load(MODEL_PATH)
except Exception as error:
    st.error(f"Could not load model: {error}")
    st.stop()

st.subheader("🔍 Test Transaction")

amount = st.number_input(
    "Transaction amount",
    min_value=0.0,
    value=500.0,
    step=100.0
)

time_value = st.number_input(
    "Transaction time (seconds)",
    min_value=0.0,
    value=1000.0,
    step=100.0
)

st.warning(
    "For this demo, V1–V28 are set to zero. "
    "This is a sample input, not real transaction data."
)

if st.button("Analyze Sample Transaction", width="stretch"):
    columns = (
        ["Time"]
        + [f"V{i}" for i in range(1, 29)]
        + ["Amount"]
    )

    values = [time_value] + [0.0] * 28 + [amount]
    transaction = pd.DataFrame([values], columns=columns)

    prediction = model.predict(transaction)[0]
    probability = model.predict_proba(transaction)[0][1]

    st.subheader("Analysis Result")
    st.metric("Model-estimated fraud probability", f"{probability:.2%}")

    if prediction == 1:
        st.error("⚠️ Model flagged this sample as potentially fraudulent.")
    else:
        st.success("Model did not flag this sample as fraudulent.")

    st.caption(
        "This result applies only to the supplied sample features. "
        "It does not establish whether a real payment is safe."
    )

st.divider()
st.caption("AI Payment Safety | Educational ML Prototype")
st.divider()
st.header("📂 Bulk Transaction Analysis")

uploaded_file = st.file_uploader(
    "Upload a CSV file",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        required_columns = (
            ["Time"]
            + [f"V{i}" for i in range(1, 29)]
            + ["Amount"]
        )

        missing_columns = [
            col for col in required_columns
            if col not in df.columns
        ]

        if missing_columns:
            st.error(
                "CSV mein required columns missing hain. "
                "European credit-card dataset format ki file upload karo."
            )
            st.write("Missing columns:", missing_columns)

        elif df.empty:
            st.warning("CSV file empty hai.")

        else:
            features = df[required_columns].apply(
                pd.to_numeric, errors="coerce"
            )

            if features.isna().any().any():
                st.error(
                    "CSV mein missing ya invalid numeric values hain. "
                    "Pehle unhe correct karo."
                )
            else:
                predictions = model.predict(features)
                probabilities = model.predict_proba(features)[:, 1]

                results = df.copy()
                results["Predicted_Fraud"] = predictions
                results["Fraud_Probability"] = probabilities
                
                def risk_level(probability):
                    if probability >= 0.70:
                        return "High Risk"
                    elif probability >= 0.30:
                        return "Medium Risk"
                    else:
                        return "Low Risk"

                results["Risk_Level"] = results["Fraud_Probability"].apply(
                    risk_level
                )


                fraud_count = int((predictions == 1).sum())
                normal_count = int((predictions == 0).sum())

                col1, col2, col3 = st.columns(3)

                col1.metric("Total Transactions", len(results))
                col2.metric("Flagged as Fraud", fraud_count)
                col3.metric("Not Flagged", normal_count)
                
                st.divider()
                st.header("📊 Fraud Detection Dashboard")

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Total Transactions",
                    f"{len(results):,}"
                )

                col2.metric(
                    "Flagged as Fraud",
                    f"{fraud_count:,}"
                )

                col3.metric(
                    "Not Flagged",
                    f"{normal_count:,}"
                )

                st.subheader("📈 Prediction Distribution")

                chart_data = pd.DataFrame({
                    "Category": ["Flagged as Fraud", "Not Flagged"],
                    "Transactions": [fraud_count, normal_count]
                })

                st.bar_chart(
                    chart_data.set_index("Category")
                )
                
                if "Class" in df.columns:
                    st.divider()
                    st.header("🎯 Model Performance Analysis")

                    actual = pd.to_numeric(
                        df["Class"], errors="coerce"
                    )

                    if actual.isna().any() or not actual.isin([0, 1]).all():
                        st.warning(
                            "Class column mein sirf valid 0 aur 1 labels hone chahiye."
                        )
                    else:
                        actual = actual.astype(int)
                        predicted = pd.Series(predictions, index=df.index)

                        tp = int(((actual == 1) & (predicted == 1)).sum())
                        fn = int(((actual == 1) & (predicted == 0)).sum())
                        fp = int(((actual == 0) & (predicted == 1)).sum())
                        tn = int(((actual == 0) & (predicted == 0)).sum())

                        precision = tp / (tp + fp) if (tp + fp) else 0
                        recall = tp / (tp + fn) if (tp + fn) else 0
                        f1 = (
                            2 * precision * recall / (precision + recall)
                            if (precision + recall) else 0
                        )

                        m1, m2, m3 = st.columns(3)
                        m1.metric("Precision", f"{precision:.2%}")
                        m2.metric("Recall", f"{recall:.2%}")
                        m3.metric("F1 Score", f"{f1:.2%}")

                        st.subheader("📋 Prediction Breakdown")

                        performance = pd.DataFrame({
                            "Result": [
                                "Fraud correctly detected",
                                "Fraud missed by model",
                                "Normal incorrectly flagged",
                                "Normal correctly identified"
                            ],
                            "Transactions": [tp, fn, fp, tn]
                        })

                        st.dataframe(
                            performance,
                            use_container_width=True,
                            hide_index=True
                        )

                        st.caption(
                            "These metrics compare predictions with the Class labels "
                            "in the uploaded dataset. They are meaningful only when "
                            "those labels are genuine ground truth and the data was "
                            "not used to train the evaluated model."
                        )




                st.subheader("📊 Analysis Results")
                st.dataframe(results, use_container_width=True)

                csv_output = results.to_csv(index=False).encode("utf-8")

                st.download_button(
                    "⬇️ Download Analysis Report",
                    data=csv_output,
                    file_name="payment_analysis_report.csv",
                    mime="text/csv",
                    use_container_width=True
                )

                st.warning(
                    "Educational prototype: results are model predictions, "
                    "not proof of fraud or confirmation that a payment is safe."
                )

    except Exception as error:
        st.error(f"File process nahi ho payi: {error}")
