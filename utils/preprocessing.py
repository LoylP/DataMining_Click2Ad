import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

def handle_categorical_conversion(categorical_features):
    selected_columns = st.multiselect(
        "Chọn cột Categorical để chuyển đổi:",
        categorical_features,
        default=categorical_features  # Mặc định chọn tất cả các cột
    )
    
    if selected_columns:
        for column in selected_columns:
            label_encoder = LabelEncoder()
            st.session_state.df_preprocessing[column] = label_encoder.fit_transform(
                st.session_state.df_preprocessing[column].astype(str)
            )
        
        st.success("✅ Chuyển đổi dữ liệu thành công!")
        st.dataframe(st.session_state.df_preprocessing, use_container_width=True)

# Xử lý giá trị null
def handle_missing_values():
    null_columns = st.session_state.df_preprocessing.columns[
        st.session_state.df_preprocessing.isnull().any()
    ].tolist()
    st.write(f"Các cột chứa giá trị null: {null_columns}")
    
    fill_method = st.radio(
        "Chọn phương pháp điền giá trị null:",
        ["Điền số 0", "Điền giá trị trung bình", "Điền giá trị trung vị", "Điền giá trị phổ biến nhất"],
        index=3  # Mặc định chọn "Điền giá trị phổ biến nhất"
    )
    
    if null_columns:
        for selected_column in null_columns:
            col_type = st.session_state.df_preprocessing[selected_column].dtype
            if fill_method == "Điền số 0":
                st.session_state.df_preprocessing[selected_column] = st.session_state.df_preprocessing[selected_column].fillna(0)
            elif fill_method == "Điền giá trị trung bình":
                mean_value = st.session_state.df_preprocessing[selected_column].mean()
                st.session_state.df_preprocessing[selected_column] = st.session_state.df_preprocessing[selected_column].fillna(mean_value)
            elif fill_method == "Điền giá trị trung vị":
                median_value = st.session_state.df_preprocessing[selected_column].median()
                st.session_state.df_preprocessing[selected_column] = st.session_state.df_preprocessing[selected_column].fillna(median_value)
            elif fill_method == "Điền giá trị phổ biến nhất":
                mode_value = st.session_state.df_preprocessing[selected_column].mode()[0]
                st.session_state.df_preprocessing[selected_column] = st.session_state.df_preprocessing[selected_column].fillna(mode_value)
        
        st.success(f"✅ Điền giá trị null thành công bằng {fill_method}!")
        st.dataframe(st.session_state.df_preprocessing, use_container_width=True)
    
def detect_outliers(data):
    categorical_columns = data.select_dtypes(include=['object']).columns
    if len(categorical_columns) > 0:
        le = LabelEncoder()
        for col in categorical_columns:
            data[col] = le.fit_transform(data[col].astype(str))  # Chuyển đổi thành kiểu chuỗi rồi mã hóa
    
    k = 5
    nbrs = NearestNeighbors(n_neighbors=k)
    nbrs.fit(data)
    
    distances, _ = nbrs.kneighbors(data)
    
    mean_distances = distances.mean(axis=1)
    
    threshold = mean_distances.mean() + 2 * mean_distances.std()  # Ngưỡng = mean + 2 * std
    
    data['Outlier'] = mean_distances > threshold
    outliers = data[data['Outlier'] == True]
    
    return data, outliers, mean_distances, threshold