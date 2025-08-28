# 📊 Retail Sales Data Analysis & Forecasting Dashboard

An interactive dashboard built with Python, Streamlit, Plotly, and Prophet for analyzing retail sales data, generating insights, and forecasting future sales.

This project is useful for retail, e-commerce, and finance businesses to optimize:
- ✅ Inventory planning
- ✅ Marketing strategies
- ✅ Revenue forecasting

---

## 🚀 Features

- 📂 Upload your own dataset (CSV)
- 🔹 Data Cleaning & Preprocessing (handle missing values, recalculate totals, drop duplicates)
- 📈 Exploratory Data Analysis (EDA) with interactive charts:
  - Customer demographics (gender, age distribution)
  - Sales trends over time
  - Product category performance
- ⚙️ Feature Engineering:
  - Extract time features (year, month, weekday)
  - Customer-level spending
- 📊 Dashboard Insights:
  - Age vs Spending scatter
  - Weekly sales performance
- 🔮 Sales Forecasting using Facebook Prophet:
  - Daily, weekly, or monthly forecasts
  - Interactive slider for forecast horizon (1–24 periods)
  - Trend and seasonality analysis
  - Downloadable forecast results as CSV

---

## 🛠️ Tech Stack

- Python 3.8+
- Streamlit – Interactive dashboard
- Pandas, NumPy – Data manipulation
- Plotly Express, Matplotlib, Seaborn – Visualizations
- Prophet (Facebook) – Forecasting model

---

## 📂 Project Structure

```
📦 Retail-Sales-Forecasting
├── app.py                  # Main Streamlit app
├── requirements.txt        # Project dependencies
├── README.md               # Documentation
└── sample_dataset.csv      # Example dataset (optional)
```

---

## ⚙️ Installation

**Clone this repository:**
```bash
git clone https://github.com/your-username/retail-sales-forecasting.git
cd retail-sales-forecasting
```

**Create a virtual environment (recommended):**
```bash
python -m venv venv
# Mac/Linux
source venv/bin/activate
# Windows
venv\Scripts\activate
```

**Install dependencies:**
```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

**Run the Streamlit app:**
```bash
streamlit run app.py
```

Then, open your browser at [http://localhost:8501](http://localhost:8501)

**Steps:**
1. Upload your sales dataset (CSV format).
2. Explore EDA, feature engineering, and insights.
3. Use the Forecasting tab to predict future sales.
4. Download forecast results as CSV.

---

## 📊 Example Dataset

**Columns typically required:**
- Date (YYYY-MM-DD)
- Customer ID
- Gender
- Age
- Product Category
- Quantity
- Price per Unit
- Total Amount

*(You can use Kaggle retail datasets such as Walmart Sales Forecasting)*

---

## 📸 Screenshots

- Dashboard Home
- Forecasting Example

*(Add screenshots after running your app)*

---

## 📌 Future Improvements

- Add ARIMA & LSTM models for comparison
- Deploy dashboard on Streamlit Cloud or Heroku
- Add anomaly detection for unusual sales trends

---

## 📄 License

This project is licensed under the MIT License – feel free to use.
