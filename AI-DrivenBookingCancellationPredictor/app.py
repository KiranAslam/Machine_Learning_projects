import streamlit as st
import pandas as pd
import joblib 
import numpy as np

st.set_page_config(page_title="Hotel Booking Cancellation Predictor", layout="wide")

@st.cache_resource
def load_model():
    return joblib.load('models/booking_cancellation_predictor.pkl')

model = load_model()
st.title("Hotel Booking Cancellation Predictor")
st.markdown("Predict the likelihood of a booking cancellation to protect your revenue.")
st.divider()

st.sidebar.header("Booking details")
def user_input_features():
    lead_time = st.sidebar.number_input("Lead Time (Days before arrival)", min_value=0, value=30)
    adr = st.sidebar.number_input("ADR (Average Daily Rate)", min_value=0.0, value=100.0)
    total_special_requests = st.sidebar.slider("Special Requests", 0, 5, 0)
    car_parking = st.sidebar.selectbox("Required Car Parking?", [0, 1])
    
  
    hotel = st.sidebar.selectbox("Hotel Type", ['Resort Hotel', 'City Hotel'])
    market_segment = st.sidebar.selectbox("Market Segment", ['Direct', 'Corporate', 'Online TA', 'Offline TA/TO', 'Complementary', 'Groups'])
    deposit_type = st.sidebar.selectbox("Deposit Type", ['No Deposit', 'Refundable', 'Non Refund'])
    customer_type = st.sidebar.selectbox("Customer Type", ['Transient', 'Contract', 'Transient-Party', 'Group'])
    country = st.sidebar.text_input("Country Code (e.g. PRT, GBR, USA)", value="PRT").upper()

    data = {
        'lead_time': lead_time,
        'adr': adr,
        'total_of_special_requests': total_special_requests,
        'required_car_parking_spaces': car_parking,
        'hotel': hotel,
        'market_segment': market_segment,
        'deposit_type': deposit_type,
        'customer_type': customer_type,
        'country': country
    }
    return pd.DataFrame([data])

input_df = user_input_features()
col1,col2 = st.columns([1,1])
with col1:
    st.subheader("Summary of Input Data")
    st.write(input_df)

with col2:
    st.subheader("Prediction Result")
    if st.button("Analyze Booking"):
      
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)[0][1]

        if prediction[0] == 1:
            st.error(f" **High Risk:** This booking is likely to be **CANCELED**.")
            st.warning(f"Confidence Level: {prediction_proba*100:.2f}%")
        else:
            st.success(f" **Safe:** This booking is likely to be **CONFIRMED**.")
            st.info(f"Cancellation Probability: {prediction_proba*100:.2f}%")


st.divider()
st.caption("Developed by Kiran Aslam | AI-Data-Driven Booking Cancellation Predictor")
