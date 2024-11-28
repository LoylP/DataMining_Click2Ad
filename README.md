# 📊 Data Mining Demo App

A **Streamlit** app showcasing various data mining techniques, including data processing, outlier detection, correlation analysis, binning, smoothing, clustering, and prediction. The app serves as a demonstration of applying machine learning algorithms on real-world datasets and performing basic exploratory data analysis (EDA).

![](/data/config/home.png)
---

## Table of Contents

- [Introduction](#introduction)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Features](#features)
- [Usage](#usage)
  - [Upload Data](#upload-data)
  - [Data Processing](#data-processing)
  - [Outlier Detection](#outlier-detection)
  - [Clustering and Prediction](#clustering-and-prediction)
- [License](#license)

## Introduction

The **Data Mining Demo App** provides an interactive platform to explore and apply key data mining techniques. This app is designed to be user-friendly, with a simple interface allowing users to process, analyze, and visualize datasets in various ways. 

**Key Features:**
- Data preprocessing and cleaning
- Outlier detection using K-Nearest Neighbors (KNN)
- Correlation analysis between features
- Binning and smoothing techniques for noise reduction
- K-Means clustering and machine learning models for prediction

This tool is ideal for learners, data scientists, and anyone interested in performing data mining on small to medium datasets.

## Technologies Used

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Streamlit](https://img.shields.io/badge/Streamlit-0A0A0A?style=for-the-badge&logo=streamlit&logoColor=white) 
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white) 
![Jupyter Notebook](https://img.shields.io/badge/Jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white) 

## Installation
#### You can run with docker: 
```bash
docker build -t streamlit-app .
docker run -p 8080:8080 streamlit-app
```
#### Or follow these steps:
1. Clone the repository:
```bash
# 1. Clone the repository:
git clone https://github.com/LoylP/Data_Mining_App.git
# 2. Navigate to the project directory:
cd Data_Mining_App
```
2. (Optional) Create and activate a virtual environment:
 - For Ubuntu/macOS:
```bash
python3 -m venv venv
source venv/bin/activate
```
 - For Windows:
```bash
python -m venv venv
.\venv\Scripts\activate
```
3. Install the required dependencies:
```bash
pip install -r requirements.txt
```  
4. Run:
```bash
streamlit run app.py
```

## Features

### 🏠 Home

- **Overview**: Displays the dataset summary, including the shape and the number of missing values for each column.
- **File Upload**: Users can upload a CSV file for analysis, which is then displayed in a tabular format.

![](/data/config/upload.png)
### 🛠 Data Processing

- **Categorical and Numerical Columns**: Automatically detect and classify columns into categorical or numerical types.
- **Missing Value Handling**: Display and provide options to fill or drop missing values.
- **Encoding Categorical Data**: Convert categorical columns to numerical values using techniques like One-Hot Encoding.

![](/data/config/preprocessing.png)

### 🚫 Outlier Detection

- **KNN-based Outlier Detection**: Identify data points that significantly differ from the rest of the dataset.
- **Outlier Removal**: Optionally remove outliers based on a set threshold, improving the quality of analysis.
- **Distribution Visualization**: Visualize the distance distribution to help users set an appropriate threshold for outlier detection.

![](/data/config/outlier.png)

### 🔗 Correlation Analysis

- **Correlation Matrix**: Visualize correlations between numerical columns using a heatmap.
- **Threshold Selection**: Choose a minimum correlation threshold to filter out weak correlations, focusing on the most significant relationships.

![](/data/config/corr.png)

### 🌫 Binning/Smoothing

- **Data Binning**: Group continuous data into bins (e.g., age ranges, income groups).

![](/data/config/binning.png)

- **Smoothing**: Apply smoothing techniques to reduce noise in binned data, improving the quality of the analysis.

![](/data/config/smooth.png)

- **Apriori Algorithm**: Perform association rule mining to find frequent itemsets and learn associations between features.

![](/data/config/tapphobien.png)

### 🚀 Clustering

- **K-Means Clustering**: Apply K-Means clustering on numerical features to group similar data points.

![](/data/config/gomcum.png)

- **Cluster Visualization**: Visualize clustering results in 2D or 3D plots for better interpretation.

![](/data/config/clustering_visualize.png)

- **Cluster Count Selection**: Select the number of clusters using the Elbow method or manual input.

![](/data/config/clustering_visualize.png)

### 📈 EDA (Exploratory Data Analysis)

- **Descriptive Statistics**: Display key statistics like mean, median, standard deviation, and percentiles.
- **Visualizations**: Generate histograms, scatter plots, box plots, and pair plots for deeper insights into the data distribution and relationships.
- **Feature Correlation**: Visualize and analyze the correlation between selected features to understand dependencies.

![](/data/config/EDA.png)

### 🔎 Prediction

- **Model Training**: Train machine learning models such as linear regression, decision trees, and random forests.

- **Model Evaluation**: Display performance metrics like accuracy, precision, recall, F1 score, and confusion matrices.

![](/data/config/model.png)

- **Prediction Input**: Allow users to input new data and predict outcomes based on the trained model.

![](/data/config/form.png)

## Usage
#### Upload Data:
- Click on the "Upload CSV" button in the sidebar to upload a CSV dataset.
- After the file is uploaded, it will be displayed in a table format on the main page.
#### Data Processing:
- Categorization: View and manage categorical and numerical columns.
- Missing Value Handling: Choose to either drop or impute missing values.
- Encoding: Convert categorical columns to numerical format using one-hot encoding.
#### Outlier Detection:
- Run Outlier Detection: Detect outliers using the KNN-based method.
- Adjust Threshold: Set the threshold for outlier removal.
- Visualize: See the distribution of average distances and outliers highlighted on the graph.
#### Clustering and Prediction:
- K-Means Clustering: Apply K-Means clustering on selected numerical columns.
- Train Model: Train machine learning models and evaluate them using metrics like accuracy and precision.
- Prediction: Input new data points to make predictions based on the trained models.

## 📄 License
MIT License
