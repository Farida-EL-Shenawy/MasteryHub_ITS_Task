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
if st.button("🚀 Predict Churn"):
    input_data = build_input_df()
    
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("🔍 Prediction Result")
    if prediction == 1:
        st.error(f"⚠️ Predicted: **CHURN** with probability {probability:.2%}")
    else:
        st.success(f"✅ Predicted: **NO CHURN** with probability {1-probability:.2%}")

    # Plot with plotly
    color = 'red' if prediction == 1 else 'lightblue'
    label = 'Churn' if prediction == 1 else 'No Churn'
    fig = px.bar(
        x=[label],
        y=[probability if prediction == 1 else 1-probability],
        labels={'x': 'Predicted Class', 'y': 'Churn Probability'},
        color_discrete_sequence=[color],
        text_auto='.2%'
    )
    fig.update_layout(title_text='Predicted Churn Probability for Customer')
    st.plotly_chart(fig, use_container_width=True)