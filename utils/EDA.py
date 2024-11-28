import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

def show_Histogram(data, numerical_columns):

    column = st.selectbox("🔢 Chọn cột để hiển thị Histogram:", numerical_columns)
    if st.button("Hiển thị Histogram"):
        fig = plt.figure(figsize=(8, 6))
        sns.histplot(data[column], kde=True, color='skyblue', bins=30)
        plt.title(f"Histogram của {column}", fontsize=16)
        plt.xlabel(column, fontsize=12)
        plt.ylabel("Tần suất", fontsize=12)
        st.pyplot(fig)

def show_Boxplot(data, numerical_columns):

    column = st.selectbox("🔢 Chọn cột để hiển thị Boxplot:", numerical_columns)
    if st.button("Hiển thị Boxplot"):
        fig = plt.figure(figsize=(8, 6))
        sns.boxplot(data[column], color='orange')
        plt.title(f"Boxplot của {column}", fontsize=16)
        plt.xlabel(column, fontsize=12)
        st.pyplot(fig)

def show_Scatter(data, numerical_columns):

    x_col = st.selectbox("🟦 Chọn cột trục X:", numerical_columns)
    y_col = st.selectbox("🟥 Chọn cột trục Y:", numerical_columns)
    if st.button("Hiển thị Scatter Plot"):
        fig = plt.figure(figsize=(8, 6))
        sns.scatterplot(data=data, x=x_col, y=y_col, color='green')
        plt.title(f"Scatter Plot: {x_col} vs {y_col}", fontsize=16)
        plt.xlabel(x_col, fontsize=12)
        plt.ylabel(y_col, fontsize=12)
        st.pyplot(fig)

def show_Heatmap(data, numerical_columns):
    if len(numerical_columns) > 1:
        if st.button("Hiển thị Heatmap"):
            fig = plt.figure(figsize=(10, 8))
            sns.heatmap(data[numerical_columns].corr(), annot=True, cmap='coolwarm', fmt=".2f", square=True)
            plt.title("Heatmap - Mối tương quan giữa các biến", fontsize=16)
            st.pyplot(fig)
    else:
        st.warning("⚠️ Cần ít nhất hai cột số để hiển thị Heatmap.")