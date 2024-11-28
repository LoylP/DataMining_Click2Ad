import streamlit as st
import pandas as pd

def smoothing(df, column):

    bin_means = df.groupby('Binned')[column].transform('mean')

    original_dtype = st.session_state.df[column].dtype
    if original_dtype == 'int64':
        df[column] = bin_means.round().astype(int)
    else:
        df[column] = bin_means.round(1).astype(original_dtype)

    # Xóa cột 'Binned' sau khi làm trơn
    df = df.drop(columns=['Binned'])
    
    st.write(f"Dữ liệu cột {column} sau khi làm trơn (trung bình theo nhóm):")
    st.dataframe(df, use_container_width=True)