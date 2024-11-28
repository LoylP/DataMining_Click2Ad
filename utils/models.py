import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

def select_data_for_prediction():
    if st.session_state.df_normalize is not None:
        data_options = ['df', 'df_smooth', 'df_normalize']
    elif st.session_state.df_smooth is not None:
        data_options = ['df', 'df_smooth']
    elif st.session_state.df is not None:
        data_options = ['df']
    else:
        st.warning("⚠️ Vui lòng tải và xử lý dữ liệu trước.")
        return None

    df_to_choose = st.selectbox("🔎 Chọn dữ liệu để dự đoán:", data_options)
    
    if df_to_choose == 'df_normalize':
        return st.session_state.df_normalize.copy()
    elif df_to_choose == 'df_smooth':
        return st.session_state.df_smooth.copy()
    else:
        return st.session_state.df.copy()

# Hàm chọn cột X và Y
def select_columns(data):
    columns = list(data.columns)
    X_columns = st.multiselect("Chọn cột X (biến độc lập):", columns)
    Y_column = st.selectbox("Chọn cột Y (biến mục tiêu):", columns)

    if not X_columns or not Y_column:
        X_columns = columns[:-1]
        Y_column = columns[-1]
    
    X = data[X_columns]
    y = data[Y_column]
    
    return X, y, X_columns, Y_column

# Hàm huấn luyện mô hình
def train_model(algorithm, X_train, y_train):
    if algorithm == "DecisionTree":
        model = DecisionTreeClassifier()
    elif algorithm == "KNN":
        model = KNeighborsClassifier()
    elif algorithm == "SVM":
        model = SVC(probability=True)
    elif algorithm == "XGBoost":
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    else:
        st.error("Thuật toán không hợp lệ!")
        return None
    
    model.fit(X_train, y_train)
    return model

# Hàm hiển thị và nhận đầu vào người dùng để dự đoán
def get_user_input(X_columns, data):
    input_data = {}
    for col in X_columns:
        dtype = data[col].dtype
        if np.issubdtype(dtype, np.number):
            input_data[col] = st.number_input(f"Nhập giá trị cho {col}:", value=float(data[col].mean()))
        else:
            input_data[col] = st.text_input(f"Nhập giá trị cho {col}:")
    
    return pd.DataFrame([input_data])

# Hàm hiển thị kết quả dự đoán
def show_prediction_result(input_df):
    if "trained_model" in st.session_state:
        model = st.session_state.trained_model
        prediction = model.predict(input_df)
        st.success(f"🔮 Kết quả dự đoán: {prediction[0]}")
    else:
        st.info("🔧 Vui lòng huấn luyện mô hình trước khi dự đoán.")