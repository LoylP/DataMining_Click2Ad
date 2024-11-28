import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

def calculate_distance(point, centroid):
    return np.sqrt(np.sum((point - centroid) ** 2))

# Hàm lấy các cột số
def get_numerical_columns(df):
    return df.select_dtypes(include=[np.number]).columns.tolist()

# Hàm vẽ biểu đồ clusters
def plot_clusters_with_labels(X, clusters, centroids):
    colors = ['r', 'g', 'b', 'c', 'm', 'y']
    plt.figure(figsize=(10, 6))
    
    for i, cluster in enumerate(clusters):
        cluster = np.array(cluster)
        if len(cluster) > 0:
            plt.scatter(cluster[:, 0], cluster[:, 1], 
                       c=colors[i % len(colors)], 
                       marker='.', 
                       alpha=0.6,
                       label=f'Cụm {i + 1}')
            
            plt.scatter(centroids[i][0], centroids[i][1], 
                       c=colors[i % len(colors)], 
                       marker='*', 
                       s=300,
                       edgecolor='black',
                       linewidth=1,
                       label=f'Tâm cụm {i + 1}')
            
            plt.text(centroids[i][0], centroids[i][1] - 0.05, 
                    f"C{i + 1}", 
                    fontsize=10,
                    color='black',
                    ha='center')
    
    plt.xlabel('Chiều 1')
    plt.ylabel('Chiều 2')
    plt.title('Biểu đồ phân cụm K-Means')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

def kmeans_with_progress(X, k, centroids, progress_bar, status_text):
    prev_centroids = centroids.copy()
    iteration = 0
    
    while True:
        iteration += 1
        progress = min(iteration / 10, 1.0)  # Giả sử tối đa 10 lần lặp
        progress_bar.progress(progress)
        status_text.text(f"Đang thực hiện lần lặp thứ {iteration}...")
        
        # Tạo các cụm
        clusters = [[] for _ in range(k)]
        for point in X:
            distances = [calculate_distance(point, centroid) for centroid in centroids]
            closest_centroid = np.argmin(distances)
            clusters[closest_centroid].append(point)
        
        # Tính centroid mới
        new_centroids = []
        with st.expander(f"📍 Kết quả lần lặp {iteration}", expanded=True):
            for i in range(k):
                if len(clusters[i]) > 0:
                    new_centroid = np.mean(clusters[i], axis=0)
                else:
                    new_centroid = centroids[i]
                new_centroids.append(new_centroid)
                st.text(f"Centroid mới của cụm {i + 1}: {new_centroid}")
        
        if np.allclose(new_centroids, prev_centroids):
            break
            
        prev_centroids = new_centroids.copy()
        centroids = np.array(new_centroids)
    
    return clusters, centroids