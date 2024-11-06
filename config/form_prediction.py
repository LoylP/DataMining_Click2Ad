import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import joblib
from xgboost import XGBClassifier

label_encoders = {}

df_raw = pd.read_csv("./data/ad_10000records.csv")

for column in ['Ad Topic Line', 'City', 'Country']:
    label_encoders[column] = LabelEncoder()
    df_raw[column] = label_encoders[column].fit_transform(df_raw[column])

def predict(input):
    model_path_xgb = "models/best_xgb_model.joblib"
    best_model = joblib.load(model_path_xgb)
    predictions = best_model.predict(input)
    return predictions

def user_info_form():
    st.title("User Information Form")
    with st.form(key='user_info_form'):

        col1, col2 = st.columns(2)
        with col1:
            country = st.text_input("Country")
        with col2:
            city = st.text_input("City")

        col3, col4 = st.columns(2)
        with col3:
            age = st.number_input("Age", min_value=0.0, step=0.1) 
        with col4:
            gender = st.selectbox("Gender", ["Male", "Female"])

        area_income = st.number_input("Area Income", min_value=0.0, step=100.0)

        col5, col6 = st.columns(2)
        with col5:
            daily_time_spent = st.number_input("Daily Time Spent on Site (minutes)", min_value=0.0, step=0.1)
        with col6:
            daily_internet_usage = st.number_input("Daily Internet Usage (minutes)", min_value=0.0, step=0.1)

        ad_topic_line = st.text_input("Ad Topic Line")

        col7, col8 = st.columns(2)
        with col7:
            date = st.date_input("Date")
        with col8:
            hour = st.number_input("Time", min_value=0, max_value=24, step=1) 

        submit_button = st.form_submit_button(label="Submit")

    # Process form data if submitted
    if submit_button:
        # Encode categorical inputs
        ad_topic_encoded = label_encoders['Ad Topic Line'].transform([ad_topic_line])[0] if ad_topic_line in label_encoders['Ad Topic Line'].classes_ else 0
        city_encoded = label_encoders['City'].transform([city])[0] if city in label_encoders['City'].classes_ else 0
        country_encoded = label_encoders['Country'].transform([country])[0] if country in label_encoders['Country'].classes_ else 0
        gender_encoded = 1 if gender == "Male" else 0

        month = date.month
        day = date.day

        # Prepare input data for prediction
        new_data = {
            "Daily Time Spent on Site": [daily_time_spent],
            "Age": [age],
            "Area Income": [area_income],
            "Daily Internet Usage": [daily_internet_usage],
            "Ad Topic Line": [ad_topic_encoded],
            "City": [city_encoded],
            "Gender": [gender_encoded],
            "Country": [country_encoded],
            "month": [month],
            "day": [day],
            "hour": [hour]
        }
        X = pd.DataFrame(new_data)
        st.write(X)

        # Load model and make prediction
        predictions = predict(X)
        if predictions == 1:
            st.subheader("Predicting the Click-Through Rate: Yes")
        else:
            st.subheader("Predicting the Click-Through Rate: No")