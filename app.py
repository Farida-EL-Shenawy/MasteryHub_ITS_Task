import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# Load trained model
model = joblib.load('best_rf_GridSearch_model.pkl')

st.title("📈 Customer Churn Prediction App")
st.write("Predict whether a customer will churn based on their account information.")

# -----------------------------
# USER INPUTS
# -----------------------------
st.header("📝 Enter Customer Details")

gender = st.selectbox('Gender', ['Male', 'Female'])
SeniorCitizen = st.selectbox('Senior Citizen', [0, 1])
Partner = st.selectbox('Partner', ['Yes', 'No'])
Dependents = st.selectbox('Dependents', ['Yes', 'No'])
tenure = st.slider('Tenure (months)', 0, 72, 12)
PhoneService = st.selectbox('Phone Service', ['Yes', 'No'])
PaperlessBilling = st.selectbox('Paperless Billing', ['Yes', 'No'])
MonthlyCharges = st.slider('Monthly Charges', 18.0, 120.0, 70.0)
TotalCharges = st.slider('Total Charges', 18.0, 9000.0, 2500.0)

MultipleLines = st.selectbox('Multiple Lines', ['No phone service', 'Yes', 'No'])
InternetService = st.selectbox('Internet Service', ['Fiber optic', 'DSL', 'No'])
OnlineSecurity = st.selectbox('Online Security', ['No internet service', 'Yes', 'No'])
OnlineBackup = st.selectbox('Online Backup', ['No internet service', 'Yes', 'No'])
DeviceProtection = st.selectbox('Device Protection', ['No internet service', 'Yes', 'No'])
TechSupport = st.selectbox('Tech Support', ['No internet service', 'Yes', 'No'])
StreamingTV = st.selectbox('Streaming TV', ['No internet service', 'Yes', 'No'])
StreamingMovies = st.selectbox('Streaming Movies', ['No internet service', 'Yes', 'No'])
Contract = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
PaymentMethod = st.selectbox(
    'Payment Method', 
    ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)']
)

# -----------------------------
# BUILD INPUT DATAFRAME
# -----------------------------
trained_columns = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
    'PhoneService', 'PaperlessBilling', 'MonthlyCharges', 'TotalCharges',
    'MultipleLines_No phone service', 'MultipleLines_Yes',
    'InternetService_Fiber optic', 'InternetService_No',
    'OnlineSecurity_No internet service', 'OnlineSecurity_Yes',
    'OnlineBackup_No internet service', 'OnlineBackup_Yes',
    'DeviceProtection_No internet service', 'DeviceProtection_Yes',
    'TechSupport_No internet service', 'TechSupport_Yes',
    'StreamingTV_No internet service', 'StreamingTV_Yes',
    'StreamingMovies_No internet service', 'StreamingMovies_Yes',
    'Contract_One year', 'Contract_Two year',
    'PaymentMethod_Credit card (automatic)',
    'PaymentMethod_Electronic check',
    'PaymentMethod_Mailed check'
]

def build_input_df():
    data = dict.fromkeys(trained_columns, 0)

    # Numeric & binary inputs
    data['gender'] = 1 if gender == 'Female' else 0
    data['SeniorCitizen'] = SeniorCitizen
    data['Partner'] = 1 if Partner == 'Yes' else 0
    data['Dependents'] = 1 if Dependents == 'Yes' else 0
    data['tenure'] = tenure
    data['PhoneService'] = 1 if PhoneService == 'Yes' else 0
    data['PaperlessBilling'] = 1 if PaperlessBilling == 'Yes' else 0
    data['MonthlyCharges'] = MonthlyCharges
    data['TotalCharges'] = TotalCharges

    # One-hot features
    if MultipleLines in ['No phone service', 'Yes']:
        data[f'MultipleLines_{MultipleLines}'] = 1

    if InternetService in ['Fiber optic', 'No']:
        data[f'InternetService_{InternetService}'] = 1

    for feat, val in {
        'OnlineSecurity': OnlineSecurity,
        'OnlineBackup': OnlineBackup,
        'DeviceProtection': DeviceProtection,
        'TechSupport': TechSupport,
        'StreamingTV': StreamingTV,
        'StreamingMovies': StreamingMovies
    }.items():
        if val in ['No internet service', 'Yes']:
            data[f'{feat}_{val}'] = 1

    if Contract in ['One year', 'Two year']:
        data[f'Contract_{Contract}'] = 1

    if PaymentMethod in [
        'Credit card (automatic)',
        'Electronic check',
        'Mailed check'
    ]:
        data[f'PaymentMethod_{PaymentMethod}'] = 1

    return pd.DataFrame([data])

# -----------------------------
# PREDICTION
# -----------------------------
st.header("🚀 Make a Prediction")

# 1️⃣ Let the user pick the threshold
st.write("Choose the probability threshold to classify a customer as CHURN or NO CHURN.")
threshold = st.slider(
    label='Prediction Threshold',
    min_value=0.1,
    max_value=0.9,
    value=0.5,
    step=0.01,
    help='If predicted probability >= threshold → CHURN; else → NO CHURN'
)
st.write(f"Selected Threshold: **{threshold:.2f}**")

# 2️⃣ Prediction button
if st.button("🔮 Predict Churn"):
    # Build input DataFrame
    input_data = build_input_df()

    # Predict churn probability
    predicted_proba = model.predict_proba(input_data)[0][1]

    # Apply user-selected threshold
    predicted_class = 1 if predicted_proba >= threshold else 0

    # Set label and color
    class_label = 'CHURN' if predicted_class == 1 else 'NO CHURN'
    bar_color = 'red' if predicted_class == 1 else 'lightblue'

    # Display textual prediction result
    st.subheader("🔍 Prediction Result")
    if predicted_class == 1:
        st.error(f"⚠️ Predicted: **CHURN** with probability {predicted_proba:.2%}")
    else:
        st.success(f"✅ Predicted: **NO CHURN** with probability {predicted_proba:.2%}")

    # Ensure minimum bar height for visibility
    display_prob = max(predicted_proba, 0.05)

    # Create Plotly bar chart
    import plotly.graph_objects as go
    fig = go.Figure(go.Bar(
        x=[class_label],
        y=[display_prob],
        marker_color=bar_color,
        text=[f"Predicted Probability: {predicted_proba:.2%}"],
        textposition='auto'
    ))

    fig.update_layout(
        title='Predicted Churn Probability for Customer',
        xaxis_title='Predicted Class',
        yaxis_title='Churn Probability',
        yaxis=dict(range=[0, 1]),
        template='plotly_white',
        showlegend=False
    )

    # Display the chart in Streamlit
    st.plotly_chart(fig, use_container_width=True)