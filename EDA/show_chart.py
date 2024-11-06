import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

def show_histogram(df):
    st.header("Show Histogram:")
    cols_to_plot = df.columns.difference(['Gender', 'Clicked on Ad', 'Ad Topic Line'])
    selected_col = st.selectbox("Chọn cột để hiển thị histogram:", cols_to_plot)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(df[selected_col], bins=30, kde=True, ax=ax)
    ax.set_title(f'Histogram of {selected_col}')
    plt.xlabel(selected_col)
    plt.ylabel("Frequency")
    st.pyplot(fig)

def show_scatter(df):
    st.header("Show Scatter:")
    cols_to_plot = df.columns.difference(['Clicked on Ad'])
    col1, col2 = st.columns(2)

    with col1:
        x_col = st.selectbox("Chọn cột cho trục X:", cols_to_plot)
    with col2:
        y_col = st.selectbox("Chọn cột cho trục Y:", cols_to_plot)

    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(df[x_col], df[y_col], c=df['Clicked on Ad'], cmap='viridis', alpha=0.5)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f'{y_col} vs. {x_col}')

    # Thêm colorbar cho 'Clicked on Ad'
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Clicked on Ad')

    st.pyplot(fig)

def show_heatmap(df):
    st.header("Show Heatmap:")
    fig, ax = plt.subplots(figsize=(11, 11))
    sns.heatmap(df.iloc[:, 1:].corr(), annot=True, linewidths=.5, fmt='.1f', ax=ax)
    st.pyplot(fig)

def show_countplot(df):
    st.header("Show Countplot:")
    time_values = ['day', 'month', 'hour']
    selected_time = st.selectbox("Chọn loại thời gian:", time_values)
    plt.figure(figsize=(8, 6))
    sns.countplot(y=selected_time, hue='Clicked on Ad', data=df, palette='viridis', order=df[selected_time].value_counts().index)
    plt.title(f'Relationship between {selected_time} and Clicked on Ad')
    plt.xlabel('')
    st.pyplot(plt)