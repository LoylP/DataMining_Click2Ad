import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB

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

def show_algorithm_formula(algorithm):
    if algorithm == "NaiveBayes(Laplace Smoothing)":
        st.markdown("""
        **Áp dụng công thức Laplace smoothing:**
        - P(Xk|Ci) = (count(Ci, Xk) + 1) / (count(Ci) + r)
        - Trong đó:
          - count(Ci, Xk) là số lần xuất hiện của đặc trưng Xk trong lớp Ci.
          - count(Ci) là tổng số lần xuất hiện của tất cả các đặc trưng trong lớp Ci.
          - r là số lượng các đặc trưng.
        """)
    elif algorithm == "NaiveBayes":
        st.markdown("""
        **Áp dụng công thức:**
        - P(Y = y_i | X = x_j) = (P(X = x_j | Y = y_i) * P(Y = y_i)) / P(X = x_j)
        - Đây là công thức chuẩn của Naive Bayes, trong đó các đặc trưng được giả định là độc lập với nhau.
        """)
    elif algorithm == "DecisionTree(Entropy)":
        st.markdown("""
        **Công thức Entropy trong Decision Tree:**
        - Entropy(S) = - ∑ (p_i * log2(p_i))
        - Trong đó:
          - p_i là xác suất của lớp i trong tập S.
        """)
    elif algorithm == "DecisionTree(Gini index)":
        st.markdown("""
        **Công thức Gini Index:**
        - Gini(S) = 1 - ∑ (p_i^2)
        - Trong đó:
          - p_i là xác suất của lớp i trong tập S.
        """)
    elif algorithm == "RandomForest":
        st.markdown("""
        **Công thức Random Forest:**
        - Random Forest sử dụng nhiều cây quyết định (Decision Trees), mỗi cây được huấn luyện với một phần mẫu ngẫu nhiên của dữ liệu.
        - Dự đoán được lấy từ việc kết hợp các dự đoán của các cây.
        """)
    elif algorithm == "KNN":
        st.markdown("""
        **Công thức K-Nearest Neighbors (KNN):**
        - Dự đoán của KNN là lớp của K láng giềng gần nhất trong không gian đặc trưng.
        - Class(x) = majority_class(nearest_neighbors(x))
        """)
    elif algorithm == "SVM":
        st.markdown("""
        **Công thức Support Vector Machine (SVM):**
        - Dự đoán của SVM là hàm phân tách tối ưu nhất giữa các lớp trong không gian đặc trưng.
        - Công thức: f(x) = w^T x + b
        - Mục tiêu là tối đa hóa khoảng cách giữa các điểm dữ liệu và siêu phẳng phân cách.
        """)
    elif algorithm == "XGBoost":
        st.markdown("""
        **Công thức XGBoost:**
        - XGBoost là mô hình học máy gradient boosting, nơi mỗi cây quyết định mới cố gắng sửa chữa sai sót của cây trước đó.
        - Lớp cuối cùng được tính toán bằng cách cộng dồn kết quả của các cây quyết định: y_pred = ∑ f_t(x)
        """)
    else:
        st.warning("Công thức không có sẵn cho thuật toán này.")
        
# Hàm huấn luyện mô hình
def train_model(algorithm, X_train, y_train):
    if algorithm == "DecisionTree(Entropy)":
        model = DecisionTreeClassifier(criterion="entropy")
    elif algorithm == "DecisionTree(Gini index)":
        model = DecisionTreeClassifier(criterion="gini")
    elif algorithm == "RandomForest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif algorithm == "NaiveBayes":
        model = GaussianNB()
    elif algorithm == "NaiveBayes(Laplace Smoothing)":
        model = MultinomialNB(alpha=1.0)  # Laplace Smoothing
    elif algorithm == "KNN":
        model = KNeighborsClassifier()
    elif algorithm == "SVM":
        model = SVC(probability=True)
    elif algorithm == "XGBoost":
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    else:
        st.error("⚠️ Thuật toán không hợp lệ!")
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