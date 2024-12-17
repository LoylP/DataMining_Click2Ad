import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import random
from utils.preprocessing import handle_categorical_conversion, handle_missing_values, detect_outliers
from utils.apriori import get_numerical_columns, display_column_statistics, parse_bin_ranges, apriori
from utils.smoothing import smoothing
from utils.clustering import get_numerical_columns, plot_clusters_with_labels, kmeans_with_progress
from utils.EDA import show_Histogram, show_Boxplot, show_Scatter, show_Heatmap
from utils.models import select_data_for_prediction, select_columns, train_model, get_user_input, show_prediction_result, show_algorithm_formula
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import tree

def main():
    # Tiêu đề chính
    st.title("📊 Demo App Data-Mining")
    st.sidebar.title("📊 Menu")
    
    # Khởi tạo session_state để lưu dữ liệu
    if "df_raw" not in st.session_state:
        st.session_state.df_raw = None
    if "df" not in st.session_state:
        st.session_state.df = None
    if "df_preprocessing" not in st.session_state:
        st.session_state.df_preprocessing = None
    if "df_smooth" not in st.session_state:
        st.session_state.df_smooth = None
    if "df_normalize" not in st.session_state:
        st.session_state.df_normalize = None

    uploaded_file = st.sidebar.file_uploader("Tải lên tệp CSV:", type=["csv"])
    if uploaded_file:
        st.session_state.df_raw = pd.read_csv(uploaded_file)
        st.session_state.df_preprocessing = st.session_state.df_raw.copy()

    # Tạo menu tùy chọn với các nút bấm
    page = st.sidebar.radio(
        "Chọn một chức năng:",
        ["🏠 Trang chủ", "🛠 Xử lý dữ liệu", "🚫 Xác định Outliers", "🔗 Độ Tương Quan", "🌫 Binning/Smoothing", "🚀 Gom Cụm","🌳 Phân lớp", "📈 EDA", "🔎 Prediction"],
        label_visibility="visible"
    )

    if page == "🏠 Trang chủ":
        show_home_page()
    elif page == "🛠 Xử lý dữ liệu":
        show_data_processing_page()
    elif page == "🚫 Xác định Outliers":
        show_outlier_detection_page()
    elif page == "🔗 Độ Tương Quan":
        show_correlation_page()
    elif page == "🌫 Binning/Smoothing":
        show_binning_page()
    elif page == "🚀 Gom Cụm":
        show_clustering_page()
    elif page == "🌳 Phân lớp":
        show_classification_page()
    elif page == "📈 EDA":
        show_EDA_page()
    elif page == "🔎 Prediction":
        show_prediction_page()
    

# Trang chủ
def show_home_page():
    st.header("🏠 Dữ liệu ban đầu")
    
    if st.session_state.df_raw is not None:
        st.write("Dữ liệu hiện tại được tải từ file:")
        st.dataframe(st.session_state.df_raw, use_container_width=True)
        
        # Kiểm tra giá trị null
        st.subheader("⚠️ Kiểm tra giá trị null:")
        show_null_summary()
    else:
        st.warning("Vui lòng tải dữ liệu để bắt đầu.")

def show_null_summary():
    null_values = st.session_state.df_raw.isnull().sum()
    data_types = st.session_state.df_raw.dtypes
    unique_values = st.session_state.df_raw.nunique()
    summary_df = pd.DataFrame({
        "Feature": st.session_state.df_raw.columns,
        "null_values": null_values,
        "data_types": data_types,
        "unique_values": unique_values
    }).reset_index(drop=True)
    
    # Hiển thị bảng kết quả
    st.dataframe(summary_df, use_container_width=True)

def show_data_processing_page():
    st.header("📈 Xử lý dữ liệu")
    
    if st.session_state.df_preprocessing is not None:
        # Phân loại dữ liệu
        categorical_features = st.session_state.df_preprocessing.select_dtypes(include=['object']).columns.tolist()
        numerical_features = st.session_state.df_preprocessing.select_dtypes(include=['number']).columns.tolist()
        
        st.subheader("🔎 Phân loại dữ liệu")
        st.write("Các cột **Categorical**: ")
        st.write(categorical_features)
        st.write("Các cột **Numerical**: ")
        st.write(numerical_features)
        
        # Chuyển đổi dữ liệu categorical sang dữ liệu số
        if categorical_features:
            st.subheader("🔄 Chuyển đổi dữ liệu Categorical sang số")
            handle_categorical_conversion(categorical_features)
        
        st.subheader("🛠️ Xử lý giá trị null")
        handle_missing_values()

        save_button = st.button("💾 Lưu dữ liệu đã xử lý")
        if save_button:
            st.session_state.df = st.session_state.df_preprocessing.copy()
            st.success("Dữ liệu đã được lưu vào df!")

    else:
        st.warning("Vui lòng tải dữ liệu để bắt đầu.")

def show_outlier_detection_page():
    st.header("🚫 Xác định Outliers bằng KNN")
    
    if st.session_state.df is not None:
        data_new = st.session_state.df.copy()

        data_with_outliers, outliers, mean_distances, threshold = detect_outliers(data_new)

        st.subheader("📈 Các phần tử Outliers:")
        st.dataframe(outliers, use_container_width=True)

        plt.figure(figsize=(10, 6))
        plt.hist(mean_distances, bins=30, alpha=0.7, label='Distances')
        plt.axvline(threshold, color='r', linestyle='--', label='Threshold')
        plt.legend()
        plt.title('Phân phối khoảng cách trung bình')
        plt.xlabel('Mean Distance')
        plt.ylabel('Frequency')
        st.pyplot(plt.gcf())

        remove_outliers = st.checkbox("Loại bỏ Outliers khỏi dữ liệu")
        
        if remove_outliers:
            data_cleaned = data_with_outliers[data_with_outliers['Outlier'] == False].drop(columns=['Outlier'])
            st.subheader("📊 Dữ liệu sau khi loại bỏ Outliers:")
            st.dataframe(data_cleaned, use_container_width=True)
            
            save_button = st.button("💾 Lưu dữ liệu đã xử lý")
            if save_button:
                st.session_state.df = data_cleaned
                st.success("Dữ liệu đã được lưu vào df!")

        else:
            st.info("Dữ liệu hiện tại vẫn giữ lại các outliers.")
    else:
        st.warning("Vui lòng tải và xử lý dữ liệu trước.")

def show_correlation_page():
    st.header("🔗 Độ tương quan giữa các cột")
    
    if st.session_state.df is not None:
        corr_matrix = st.session_state.df.corr()
        
        f, ax = plt.subplots(figsize=(11, 11))
        sns.heatmap(corr_matrix, annot=True, linewidths=.5, fmt='.1f', ax=ax)
        st.pyplot(f)

        threshold = st.slider("Chọn mức tương quan tối thiểu:", 0.0, 1.0, 0.5, 0.05)
        st.write(f"Mức độ tương quan tối thiểu: {threshold}")
        
        high_corr_pairs = []
        for col1 in corr_matrix.columns:
            for col2 in corr_matrix.columns:
                if col1 != col2 and corr_matrix.loc[col1, col2] > threshold:
                    high_corr_pairs.append((col1, col2, corr_matrix.loc[col1, col2]))
        
        if high_corr_pairs:
            st.subheader("Các cặp cột có độ tương quan lớn hơn mức đã chọn:")
            for pair in high_corr_pairs:
                st.write(f"{pair[0]} và {pair[1]}: {pair[2]:.2f}")
        else:
            st.info("Không có cặp cột nào có độ tương quan lớn hơn mức đã chọn.")
        
        columns_to_remove = st.multiselect("Chọn các cột để loại bỏ:", st.session_state.df.columns.tolist())
        
        if columns_to_remove:
            st.subheader("Dữ liệu sau khi loại bỏ các cột đã chọn:")
            data_cleaned = st.session_state.df.drop(columns=columns_to_remove)
            st.dataframe(data_cleaned)
            
            save_button = st.button("💾 Lưu dữ liệu đã xử lý")
            if save_button:
                st.session_state.df = data_cleaned
                st.success("Dữ liệu đã được lưu vào df!")
        else:
            st.info("Không có cột nào được chọn để loại bỏ.")
    else:
        st.warning("Vui lòng tải và xử lý dữ liệu trước.")

def show_binning_page():
    st.header("🌫 Binning - Chia dữ liệu thành các nhóm")

    if st.session_state.df is not None:
        # Chọn cột để thực hiện binning
        numerical_columns = get_numerical_columns(st.session_state.df)
        df_temp = st.session_state.df.copy()
        
        if numerical_columns:
            column_to_bin = st.selectbox("Chọn cột số học để thực hiện Binning:", numerical_columns)
            display_column_statistics(df_temp, column_to_bin)
            # Nhập khoảng bin
            bin_ranges_input = st.text_input("Nhập các khoảng bin (ví dụ: [10, 20, 30, 40]):", "[10, 20, 30, 40]")
            
            if bin_ranges_input:
                bin_ranges = parse_bin_ranges(bin_ranges_input)
                bin_labels = [f"({bin_ranges[i]}-{bin_ranges[i+1]})" for i in range(len(bin_ranges) - 1)]
                
                if len(bin_ranges) - 1 == len(bin_labels):
                    # Thực hiện binning
                    df_temp['Binned'] = pd.cut(df_temp[column_to_bin], bins=bin_ranges, labels=bin_labels, include_lowest=True)
                    st.write("Dữ liệu đã được phân nhóm:")
                    st.dataframe(df_temp, use_container_width=True)
                    df_smooth = df_temp

                    df_one_hot = pd.get_dummies(df_temp['Binned'], prefix=column_to_bin)
                    for col in df_one_hot.columns:
                        df_one_hot[col] = df_one_hot[col].astype(int)
                    df_temp = pd.concat([df_temp, df_one_hot], axis=1)
                    df_temp = df_temp.loc[:, df_temp.nunique() == 2]
                    st.write("Dữ liệu binning:")
                    st.dataframe(df_temp, use_container_width=True)

                    # Thêm phần xử lý Apriori
                    st.subheader("📋 Tìm tập phổ biến và luật kết hợp")
                    
                    # Chọn min_support và min_confidence
                    min_support = st.slider("Chọn ngưỡng min_support:", 0.0, 1.0, 0.3, 0.05)
                    min_confidence = st.slider("Chọn ngưỡng min_confidence:", 0.0, 1.0, 0.5, 0.05)
                    
                    apriori(df_temp, min_support, min_confidence)

                    st.header("🌫 Smoothing - Làm trơn dữ liệu trong nhóm")
                    if st.button("Smoothing"):
                        # Làm trơn dữ liệu
                        smoothing(df_smooth, column_to_bin)
                        st.session_state.df_smooth = df_smooth
                    
                    st.header("Min Max Normalize")
                    if st.button("Normalize"):
                        data = st.session_state.df_smooth.copy()
                        data = data.drop(data.columns[-1], axis=1)
                        df_normalized = data.copy()
                        columns_to_normalize = data.columns
                        df_normalized[columns_to_normalize] = (data[columns_to_normalize] - data[columns_to_normalize].min()) / \
                                        (data[columns_to_normalize].max() - data[columns_to_normalize].min())

                        st.session_state.df_normalize = df_normalized
                        st.dataframe(st.session_state.df_normalize, use_container_width=True)

                else:
                    st.error("Đã xảy ra lỗi khi tạo nhãn từ các khoảng bin.")
        else:
            st.warning("Không có cột số học trong dữ liệu để thực hiện binning.")
    
    else:
        st.warning("Vui lòng tải và xử lý dữ liệu trước.")

def show_clustering_page():
    st.header("🚀 Gom Cụm với Thuật Toán K-Means")
    
    # Kiểm tra dữ liệu
    if st.session_state.df_normalize is not None:
        data_options = ['df', 'df_smooth', 'df_normalize']
    elif st.session_state.df_smooth is not None:
        data_options = ['df', 'df_smooth']
    elif st.session_state.df is not None:
        data_options = ['df']
    else:
        st.warning("⚠️ Vui lòng tải và xử lý dữ liệu trước.")
        return

    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Chọn dữ liệu
        df_to_choose = st.selectbox("📊 Chọn dữ liệu để gom cụm:", data_options)
        
        if df_to_choose == 'df_normalize':
            data = st.session_state.df_normalize.copy()
        elif df_to_choose == 'df_smooth':
            data = st.session_state.df_smooth.copy()
        else:
            data = st.session_state.df.copy()
            
        numerical_columns = get_numerical_columns(data)
        selected_columns = st.multiselect(
            "📈 Chọn các cột để gom cụm (2-3 cột):",
            numerical_columns,
            max_selections=3
        )
    
    with col2:
        k = st.slider("🎯 Số lượng cụm (k):", 2, 6, 3)
        start_clustering = st.button("▶️ Bắt đầu gom cụm", type="primary")

    if start_clustering and len(selected_columns) >= 2:
        with st.spinner('Đang thực hiện gom cụm...'):
            # Chuẩn bị dữ liệu
            X = data[selected_columns].values
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            # Khởi tạo centroid
            centroids = X_scaled[random.sample(range(len(X_scaled)), k)]
            
            # Hiển thị centroid ban đầu
            with st.expander("💫 Centroid ban đầu", expanded=True):
                for i, centroid in enumerate(centroids):
                    st.text(f"Centroid {i + 1}: {centroid}")
            
            # Thực hiện gom cụm
            progress_bar = st.progress(0)
            status_text = st.empty()
            # Thuật toán
            clusters, final_centroids = kmeans_with_progress(X_scaled, k, centroids, progress_bar, status_text)
            status_text.text("✅ Hoàn thành gom cụm!")
            progress_bar.progress(1.0)
            
            # Vẽ đồ thị nếu chọn 2 cột
            if len(selected_columns) == 2:
                st.subheader("📊 Biểu đồ phân cụm")
                plot_clusters_with_labels(X_scaled, clusters, final_centroids)
                st.pyplot(plt.gcf())
            else:
                st.info("ℹ️ Biểu đồ chỉ được hiển thị khi chọn 2 cột dữ liệu")
            
            # Hiển thị kết quả
            result_df = data.copy()
            cluster_labels = np.zeros(len(X))
            for i, cluster in enumerate(clusters):
                for point in cluster:
                    idx = np.where((X_scaled == point).all(axis=1))[0][0]
                    cluster_labels[idx] = i
            
            result_df['Cluster'] = cluster_labels.astype(int)
            st.subheader("📑 Kết quả phân cụm")
            st.dataframe(result_df, use_container_width=True)
            
    elif start_clustering:
        st.warning("⚠️ Vui lòng chọn ít nhất 2 cột để thực hiện gom cụm!")

def show_classification_page():
    st.header("🌳 Phân lớp cây quyết định (Decision Tree)")
    
    # Kiểm tra dữ liệu
    if st.session_state.df_normalize is not None:
        data_options = ['df', 'df_smooth', 'df_normalize']
    elif st.session_state.df_smooth is not None:
        data_options = ['df', 'df_smooth']
    elif st.session_state.df is not None:
        data_options = ['df']
    else:
        st.warning("⚠️ Vui lòng tải và xử lý dữ liệu trước.")
        return

    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Chọn dữ liệu
        df_to_choose = st.selectbox("📊 Chọn dữ liệu để phân lớp:", data_options)
        
        if df_to_choose == 'df_normalize':
            data = st.session_state.df_normalize.copy()
        elif df_to_choose == 'df_smooth':
            data = st.session_state.df_smooth.copy()
        else:
            data = st.session_state.df.copy()

        numerical_columns = get_numerical_columns(data)
        selected_columns = st.multiselect(
            "📈 Chọn các thuộc tính để phân lớp:",
            numerical_columns,
        )
        
        with col2:
            chart_type = st.selectbox("🛠️ Chọn kiểu tính information gain:", ["Entropy", "Gini index"])

            target_column = st.selectbox("🎯 Chọn cột mục tiêu (target):", data.columns)
            feature_columns = selected_columns

    X = data[selected_columns]
    y = data[target_column]

    # Tạo và huấn luyện mô hình
    if st.button("🔍 Phân lớp với Decision Tree"):
        criterion = "entropy" if chart_type == "Entropy" else "gini"
        model = DecisionTreeClassifier(criterion=criterion, random_state=42)
        model.fit(X, y)
        
        # Hiển thị kết quả
        st.subheader("🌟 Kết quả phân lớp:")
        
        # Vẽ cây quyết định với Matplotlib
        fig, ax = plt.subplots(figsize=(12, 6))
        plot_tree(model, feature_names=feature_columns, class_names=[str(c) for c in y.unique()],
                    filled=True, rounded=True, ax=ax)
        st.pyplot(fig)

        st.write(f"**Nút gốc của cây quyết định:** {model.tree_.value[0]}")
        st.write(f"**Độ sâu của cây quyết định:** {model.get_depth()}")
        st.write(f"**Số node của cây:** {model.get_n_leaves()}")


def show_EDA_page():
    st.header("📈 Exploratory Data Analysis (EDA)")

    # Kiểm tra dữ liệu
    if st.session_state.df_normalize is not None:
        data_options = ['df', 'df_smooth', 'df_normalize']
    elif st.session_state.df_smooth is not None:
        data_options = ['df', 'df_smooth']
    elif st.session_state.df is not None:
        data_options = ['df']
    else:
        st.warning("⚠️ Vui lòng tải và xử lý dữ liệu trước.")
        return

    df_to_choose = st.selectbox("📊 Chọn dữ liệu để phân tích:", data_options)

    if df_to_choose == 'df_normalize':
        data = st.session_state.df_normalize.copy()
    elif df_to_choose == 'df_smooth':
        data = st.session_state.df_smooth.copy()
    else:
        data = st.session_state.df.copy()

    # Hiển thị dataframe
    st.subheader("📋 Dữ liệu đã chọn")
    st.dataframe(data, use_container_width=True)
    numerical_columns = data.select_dtypes(include=['number']).columns.tolist()

    if not numerical_columns:
        st.warning("⚠️ Không tìm thấy cột số để phân tích.")
        return

    st.subheader("⚙️ Lựa chọn phân tích")
    chart_type = st.selectbox(
        "🛠️ Chọn loại biểu đồ phân tích:",
        ["Histogram", "Boxplot", "Scatter Plot", "Heatmap"]
    )

    if chart_type == "Histogram":
        show_Histogram(data, numerical_columns)
    
    elif chart_type == "Boxplot":
        show_Boxplot(data, numerical_columns)

    elif chart_type == "Scatter Plot":
        show_Scatter(data, numerical_columns)

    elif chart_type == "Heatmap":
        show_Heatmap(data, numerical_columns)

def show_prediction_page():

    st.header("🔎 Prediction")
    data = select_data_for_prediction()
    if data is None:
        return

    st.subheader("📋 Dữ liệu đã chọn")
    st.dataframe(data, use_container_width=True)

    X, y, X_columns, Y_column = select_columns(data)

    if X is None or y is None:
        st.warning("⚠️ Vui lòng chọn đầy đủ cột X và cột Y để tiếp tục.")
        return  

    st.write(f"🎯 Cột X đã chọn: {X_columns}")
    st.write(f"🎯 Cột Y đã chọn: {y.name}")

    selected_algorithm = st.selectbox(
        "📚 Chọn thuật toán dự đoán", 
        ["DecisionTree(Entropy)", "DecisionTree(Gini index)", "RandomForest", 
         "NaiveBayes", "NaiveBayes(Laplace Smoothing)", "KNN", "SVM", "XGBoost"]
    )
    test_size = st.slider("Tỷ lệ dữ liệu kiểm tra:", 0.1, 0.5, 0.2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    if st.button("🔍 Huấn luyện và đánh giá mô hình"):
        show_algorithm_formula(selected_algorithm)
        model = train_model(selected_algorithm, X_train, y_train)
        if model:
            # Đánh giá
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            st.success(f"🎯 Độ chính xác trên tập kiểm tra: {accuracy:.2%}")

            # Hiển thị Confusion Matrix
            st.subheader("📊 Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
            ax.set_xlabel('Dự đoán')
            ax.set_ylabel('Thực tế')
            st.pyplot(fig)

            # Hiển thị Classification Report với định dạng đẹp
            st.subheader("📈 Classification Report")
            report = classification_report(y_test, y_pred, target_names=[str(c) for c in model.classes_], output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            # Định dạng bảng đẹp
            st.write("**Precision, Recall, F1-Score và Support**")
            st.dataframe(report_df.style.format({
                'precision': '{:.2f}', 
                'recall': '{:.2f}', 
                'f1-score': '{:.2f}', 
                'support': '{:.0f}'
            }).background_gradient(axis=None, cmap='Blues'))

            st.session_state.trained_model = model

    # Form nhập liệu để dự đoán
    st.subheader("📋 Form nhập liệu dự đoán")
    input_df = get_user_input(X_columns, data)

    if st.button("📊 Dự đoán"):
        show_prediction_result(input_df)

if __name__ == "__main__":
    main()
