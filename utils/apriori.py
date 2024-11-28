import numpy as np
import pandas as pd
from itertools import combinations
import streamlit as st

def get_frequent_itemsets(df, min_support):
    n_transactions = len(df)
    frequent_itemsets = {}
    
    # Tìm tập phổ biến độ dài 1
    for column in df.columns:
        support = df[column].sum() / n_transactions
        if support >= min_support:
            frequent_itemsets[(column,)] = support
    
    # Tìm tập phổ biến độ dài 2
    for comb in combinations(df.columns, 2):
        support = (df[list(comb)].sum(axis=1) == len(comb)).sum() / n_transactions
        if support >= min_support:
            frequent_itemsets[comb] = support
            
    # Tìm tập phổ biến độ dài 3
    for comb in combinations(df.columns, 3):
        support = (df[list(comb)].sum(axis=1) == len(comb)).sum() / n_transactions
        if support >= min_support:
            frequent_itemsets[comb] = support
            
    return frequent_itemsets

def find_maximal_itemsets(frequent_itemsets):
    maximal_itemsets = {}
    sorted_itemsets = sorted(frequent_itemsets.items(), key=lambda x: len(x[0]), reverse=True)
    
    for itemset, support in sorted_itemsets:
        is_maximal = True
        itemset_set = set(itemset)
        
        for larger_itemset in maximal_itemsets:
            if itemset_set.issubset(set(larger_itemset)):
                is_maximal = False
                break
                
        if is_maximal:
            maximal_itemsets[itemset] = support
            
    return maximal_itemsets

def calculate_support(antecedent, consequent, df):
    """Hàm tính support cho một luật kết hợp."""
    combined = list(antecedent) + list(consequent)
    support = (df[combined].sum(axis=1) == len(combined)).sum() / len(df)
    return support

def calculate_confidence(antecedent, consequent, df):
    """Hàm tính confidence cho một luật kết hợp."""
    support_antecedent = calculate_support(antecedent, [], df)
    support_combined = calculate_support(antecedent, consequent, df)
    
    if support_antecedent == 0:
        return 0  # Tránh chia cho 0
    
    return support_combined / support_antecedent

def generate_rules(frequent_itemsets, min_confidence, df):
    """Sinh các luật kết hợp từ các tập phổ biến."""
    rules = []
    
    for itemset, support in frequent_itemsets.items():
        for i in range(1, len(itemset)):  # Tạo các luật kết hợp từ itemset
            antecedents = combinations(itemset, i)
            for antecedent in antecedents:
                consequent = tuple(set(itemset) - set(antecedent))
                confidence = calculate_confidence(antecedent, consequent, df)
                if confidence >= min_confidence:
                    rules.append((antecedent, consequent, support, confidence))
    
    return rules
def get_numerical_columns(df):
    return df.select_dtypes(include=['number']).columns.tolist()

def display_column_statistics(df, column):
    min_val = df[column].min()
    max_val = df[column].max()
    mean_val = df[column].mean()

    st.write(f"Giá trị nhỏ nhất: {min_val}")
    st.write(f"Giá trị lớn nhất: {max_val}")
    st.write(f"Giá trị trung bình: {mean_val}")

def parse_bin_ranges(bin_ranges_input):
    try:
        return eval(bin_ranges_input)
    except Exception as e:
        st.error(f"Lỗi khi phân tích khoảng bin: {e}")
        return []
        
def apriori(df, min_support, min_confidence):
    frequent_itemsets = get_frequent_itemsets(df, min_support)
    st.write(f"Các tập phổ biến thỏa ngưỡng **min_support = {min_support}**:")
    for itemset, support in frequent_itemsets.items():
        st.markdown(f" |---- {itemset}: {support:.2f}")

    # find_association_rules
    maximal_itemsets = find_maximal_itemsets(frequent_itemsets)
    st.write("Các tập phổ biến cực đại:")
    for itemset, support in maximal_itemsets.items():
        st.markdown(f" |---- {itemset}: {support:.2f}")

    # find_association_rules
    rules = generate_rules(frequent_itemsets, min_confidence, df)
    st.write(f"Các luật kết hợp thỏa ngưỡng **min_support = {min_support}** và **min_confidence = {min_confidence}**:")
    for antecedent, consequent, support, confidence in rules:
        st.markdown(f" |---- {antecedent} -> {consequent}: support={support:.2f}, confidence={confidence:.2f}")