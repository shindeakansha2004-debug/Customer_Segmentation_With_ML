# Customer Segmentation Streamlit App

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")

# Page config
st.set_page_config(page_title="Customer Segmentation App", layout="wide")

st.title("📊 Customer Segmentation Using K-Means")
st.write("Upload customer data to predict customer segments.")

# Load model
with open("kmeans_model.pkl", "rb") as f:
    kmeans_model = pickle.load(f)

# Load scaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# File upload
uploaded_file = st.file_uploader(
    "Upload Customer Dataset (CSV or Excel)",
    type=["csv", "xlsx"]
)

if uploaded_file is not None:

    # Read file
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head())

    # Handle missing values
    df["Income"].fillna(df["Income"].median(), inplace=True)

    # Feature Engineering
    df["Total_Spend"] = (
        df["MntWines"]
        + df["MntMeatProducts"]
        + df["MntFishProducts"]
        + df["MntSweetProducts"]
        + df["MntGoldProds"]
    )

    df["Total_Purchases"] = (
        df["NumWebPurchases"] + df["NumStorePurchases"]
    )

    # Select features
    features = df[
        ["Income", "Recency", "Total_Spend", "Total_Purchases"]
    ]

    # Scale features
    scaled_features = scaler.transform(features)

    # Predict clusters
    df["Cluster"] = kmeans_model.predict(scaled_features)

    # Cluster summary
    st.subheader("📌 Cluster Summary")
    st.dataframe(
        df.groupby("Cluster")[
            ["Income", "Total_Spend", "Recency", "Total_Purchases"]
        ].mean()
    )

    # PCA visualization
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(scaled_features)

    pca_df = pd.DataFrame({
        "PC1": pca_data[:, 0],
        "PC2": pca_data[:, 1],
        "Cluster": df["Cluster"]
    })

    st.subheader("📈 Cluster Visualization (PCA)")

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        data=pca_df,
        x="PC1",
        y="PC2",
        hue="Cluster",
        palette="tab10",
        ax=ax
    )

    st.pyplot(fig)

    # Download result
    st.subheader("⬇️ Download Clustered Dataset")
    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download CSV",
        data=csv,
        file_name="clustered_customers.csv",
        mime="text/csv"
    )

else:
    st.info("Please upload a dataset to start segmentation.")
