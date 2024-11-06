import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import joblib
from xgboost import XGBClassifier
from EDA.show_chart import show_histogram, show_countplot, show_heatmap, show_scatter
from config.form_prediction import user_info_form

df_raw = pd.read_csv("data/ad_10000records.csv")

# Tạo LabelEncoder cho các cột danh mục
label_encoders = {}
for column in ['Ad Topic Line', 'City', 'Country']:
    label_encoders[column] = LabelEncoder()
    df_raw[column] = label_encoders[column].fit_transform(df_raw[column])

# Hàm hiển thị menu
def streamlit_menu():
    selected = option_menu(
        menu_title=None,  
        options=["Home", "EDA"],  
        icons=["house", "book"],  
        menu_icon="cast",  
        default_index=0,  
        orientation="horizontal",
    )
    return selected

# Hiển thị trang
selected = streamlit_menu()

if selected == "Home":
    st.title(f"Demo App Prediction")
    st.header("About Dataset: Click-Through Rate")
    df = pd.read_csv("data/ad_10000records.csv")
    st.write(df.head(10))
    user_info_form()

if selected == "EDA":
    st.title(f"Exploratory Data Analysis")
    st.header("Cleaned Dataset:")
    df = pd.read_csv("data/Clean_Data_V2.csv")

    st.write(df.head(10))

    show_histogram(df)
    show_scatter(df)
    show_heatmap(df)
    show_countplot(df)