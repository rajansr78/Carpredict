import pandas as pd
import numpy as np
import datetime
import xgboost as xgb
import streamlit as st

def main():
    st.title("Car Price Prediction")

    model = xgb.XGBRegressor()
    model.load_model("xgb_model.json")  # Ensure path is correct

    st.markdown("This app will help you to predict your car selling price")

    # User inputs
    p1 = st.number_input("Please enter ex-showroom price (In Lakhs)", 2.5, 25.0, step=1.0)
    p2 = st.number_input("Please enter car driven (In Kilometers)", 100, 500000, step=100)
    s1 = st.selectbox("Select the fuel_type", ("Petrol", "Diesel", "CNG"))

    # Mapping categorical inputs
    p3 = {'Petrol': 0, 'Diesel': 1, 'CNG': 2}.get(s1)
    s2 = st.selectbox("Select the seller_type", ("Dealer", "Individual"))
    p4 = {'Dealer': 0, 'Individual': 1}.get(s2)
    s3 = st.selectbox("Select the transmission", ("Manual", "Automatic"))
    p5 = {'Manual': 0, 'Automatic': 1}.get(s3)

    # Owner and age inputs
    p6 = st.slider("How many owners", 0, 3)
    date_time = datetime.datetime.now()
    years = st.number_input("Car purchased year", 1990, date_time.year, step=1)
    p7 = date_time.year - years

    # Data for prediction
    data_new = pd.DataFrame({
        'Present_Price': [p1],
        'Kms_Driven': [p2],
        'Fuel_Type': [p3],
        'Seller_Type': [p4],
        'Transmission': [p5],
        'Owner': [p6],
        'Age': [p7]
    })

    # Prediction button
    if st.button("Predict"):
        pred = model.predict(data_new)
        st.success(f"You can sell your car at {pred[0]:.2f} lakhs")

if __name__ == '__main__':
    main()
