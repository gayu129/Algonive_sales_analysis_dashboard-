# app.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import streamlit as st
from prophet import Prophet

# -------------------------------------------
# Streamlit Dashboard with Forecasting
# -------------------------------------------

st.set_page_config(page_title="Retail Sales Dashboard", layout="wide")

# Title
st.title("📊 Retail Sales Data Analysis & Forecasting Dashboard")

# Upload Dataset
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # ---------------------------
    # Data Cleaning & Preprocessing
    # ---------------------------
    st.subheader("🔹 Data Cleaning & Preprocessing")

    # Convert Date to datetime
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    # Fill missing values
    df['Gender'] = df['Gender'].fillna("Unknown")
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Quantity'] = df['Quantity'].fillna(1)
    df['Price per Unit'] = df['Price per Unit'].fillna(df['Price per Unit'].mean())

    # Recalculate Total Amount if inconsistent
    df['Total Amount'] = df['Quantity'] * df['Price per Unit']

    # Drop duplicates
    df = df.drop_duplicates()

    st.write("Dataset after cleaning:", df.head())

    # ---------------------------
    # Tabs for Dashboard Sections
    # ---------------------------
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📈 EDA", "⚙️ Feature Engineering", "📊 Insights", "🔮 Forecasting"]
    )

    # ---------------------------
    # 1. Exploratory Data Analysis (EDA)
    # ---------------------------
    with tab1:
        st.header("Exploratory Data Analysis")

        col1, col2 = st.columns(2)
        with col1:
            gender_count = df['Gender'].value_counts()
            fig = px.pie(names=gender_count.index, values=gender_count.values,
                         title="Customer Gender Distribution")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.histogram(df, x="Age", nbins=20, title="Customer Age Distribution")
            st.plotly_chart(fig, use_container_width=True)

        # Sales Trend
        sales_trend = df.groupby("Date")["Total Amount"].sum().reset_index()
        fig = px.line(sales_trend, x="Date", y="Total Amount",
                      title="Sales Trend Over Time")
        st.plotly_chart(fig, use_container_width=True)

        # Product Category Sales
        category_sales = df.groupby("Product Category")["Total Amount"].sum().reset_index()
        fig = px.bar(category_sales, x="Product Category", y="Total Amount",
                     title="Sales by Product Category")
        st.plotly_chart(fig, use_container_width=True)

    # ---------------------------
    # 2. Feature Engineering
    # ---------------------------
    with tab2:
        st.header("Feature Engineering")

        # Extract date features
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Weekday'] = df['Date'].dt.day_name()

        # Customer total spending
        customer_spending = df.groupby("Customer ID")["Total Amount"].sum().reset_index()
        customer_spending.rename(columns={"Total Amount": "Customer_Total_Spending"}, inplace=True)

        st.write("Top 5 Customers by Spending")
        st.dataframe(customer_spending.sort_values(by="Customer_Total_Spending", ascending=False).head())

    # ---------------------------
    # 3. Dashboard Insights
    # ---------------------------
    with tab3:
        st.header("Dashboard Insights")

        # Age vs Spending Scatter
        fig = px.scatter(df, x="Age", y="Total Amount", color="Gender",
                         title="Age vs Spending by Gender", opacity=0.7)
        st.plotly_chart(fig, use_container_width=True)

        # Weekly Sales Analysis
        weekday_sales = df.groupby("Weekday")["Total Amount"].sum().reset_index()
        fig = px.bar(weekday_sales, x="Weekday", y="Total Amount",
                     title="Sales by Weekday")
        st.plotly_chart(fig, use_container_width=True)

    # ---------------------------
    # 4. Forecasting with Prophet
    # ---------------------------
    with tab4:
        st.header("🔮 Sales Forecasting")

        # Frequency selection
        freq = st.radio("Select forecast frequency:", ["Daily", "Weekly", "Monthly"])

        # Resample based on frequency
        if freq == "Daily":
            forecast_data = df.groupby("Date")["Total Amount"].sum().reset_index()
            freq_code = "D"
        elif freq == "Weekly":
            forecast_data = df.resample("W", on="Date")["Total Amount"].sum().reset_index()
            freq_code = "W"
        else:  # Monthly
            forecast_data = df.resample("M", on="Date")["Total Amount"].sum().reset_index()
            freq_code = "M"

        forecast_data = forecast_data.rename(columns={"Date": "ds", "Total Amount": "y"})

        # Select forecast horizon
        periods = st.slider("Select forecast period", min_value=1, max_value=24, value=6, step=1)
        st.caption(f"Forecast horizon = {periods} {freq.lower()}s")

        # Build Prophet Model
        model = Prophet(daily_seasonality=True, yearly_seasonality=True)
        model.fit(forecast_data)

        future = model.make_future_dataframe(periods=periods, freq=freq_code)
        forecast = model.predict(future)

        # Plot forecast
        fig1 = px.line(forecast, x="ds", y="yhat", title=f"{freq} Sales Forecast")
        fig1.add_scatter(x=forecast_data["ds"], y=forecast_data["y"], mode="markers", name="Actual Sales")
        st.plotly_chart(fig1, use_container_width=True)

        # Components (trend + seasonality)
        st.write("Forecast Components:")
        fig2 = model.plot_components(forecast)
        st.pyplot(fig2)

        # Download Button for Forecast Data
        st.subheader("⬇️ Download Forecast Results")
        csv = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Forecast CSV",
            data=csv,
            file_name="sales_forecast.csv",
            mime="text/csv"
        )

        st.success(f"✅ Forecast generated for next {periods} {freq.lower()}s. Download available below.")
else:
    st.warning("👆 Please upload a dataset CSV file to start.")


