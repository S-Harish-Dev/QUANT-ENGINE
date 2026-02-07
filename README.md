# 🇮🇳 QUANT-ENGINE: Indian Market Price Inference Dashboard

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Gemini](https://img.shields.io/badge/Gemini_2.5_Flash-8E75B2?style=for-the-badge&logo=google-gemini&logoColor=white)

**QUANT-ENGINE** is a professional-grade stock price inference dashboard specially **tuned for the Indian Stock Market (NSE/BSE)**. It leverages **Meta-Ensemble Deep Learning** combined with **Macro-Economic Intelligence** powered by Gemini 2.5 Flash.

---

## 🚀 Key Features

- **Indian Market Optimization**: Specially configured to handle NSE tickers and track Indian multi-sector conglomerates.
- **Macro-Economic Intelligence**: Integrated analysis of Indian-specific drivers like **Union Budgets, Corporate Taxes, Tariffs, and RBI Repo Rates**.
- **Hybrid Price Inference Engine**: Combines XGBoost and Attention-LSTM architectures for high-precision technical analysis.
- **Gemini 2.5 Logic**: Performs "Bigger Picture" sentiment analysis using dual-fetch news strategies (Sector + Macro).
- **Weighted Forecasting**: Blends technical signals (60%) with macro/news sentiment (40%) for a robust average target price.
- **Full Signal Persistence**: Database stores real-time XGB, LSTM, and Gemini sentiment metrics for every inference.
- **Deep Historical Context**: Caches and displays original news articles and sentiment reasoning even for historical lookups.
- **Mobile-First UX**: Responsive dashboard design with auto-scroll to results and touch-optimized controls.
- **Live News Integration**: Real-time news aggregation via Newsdata.io.
- **Professional Dashboard**: High-fidelity UI with interactive Plotly charts, performance gauges, and automated accuracy tracking.
- **Live News Integration**: Real-time news aggregation via Newsdata.io.
- **Professional Dashboard**: High-fidelity UI with interactive Plotly charts, performance gauges, and automated accuracy tracking.
- **Cloud Architecture**: Optimized for Streamlit Cloud hosting with persistent **Neon PostgreSQL** integration.
- **Instant UI Rendering**: Optimized asynchronous layout rendering (buttons load immediately) with background data synchronization.

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit (with refined CSS glassmorphism & Auto-scroll JS)
- **Deep Learning**: PyTorch (Attention-LSTM)
- **Machine Learning**: XGBoost & Scikit-learn
- **API Intelligence**: Google Gemini 2.0 Flash (Specially tuned for Indian Economy)
- **News Source**: Newsdata.io
- **Data Source**: YFinance
- **Database**: Neon PostgreSQL (Cloud Persistence System with JSON Support)

---

## 📦 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/LazzyKiller007/Stock_ML.git
cd Stock_ML
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

> [!TIP]
> **GPU Support**: If you are in a GPU-accelerated environment, use `pip install -r requirements_gpu.txt` instead for optimal performance.

### 3. Configure Secrets
#### Local Testing
Create a `.streamlit/secrets.toml` file:
```toml
GEMINI_API_KEY = "your_key"
NEWSDATA_API_KEY = "your_key"

[connections.postgresql]
url = "postgresql://user:password@your-neon-host/neondb?sslmode=require"
```

#### Streamlit Cloud Deployment
Paste the above TOML configuration into the **Secrets** section of your Streamlit Cloud dashboard.

### 4. Run the Application
```bash
streamlit run app.py
```

---

## 📊 How it Works

1. **Data Fetching**: The engine syncs with YFinance and updates the Neon PostgreSQL cloud cache.
2. **Technical Inference**: The model processes the last 15 days of price action through the Meta-Ensemble (XGBoost + DL).
3. **Sentiment Analysis**: In parallel, Newsdata.io fetches the latest news, which Gemini analyzes to determine a "Sentiment Move %".
4. **Weighted Averaging**: The system calculates the final price inference:
   `Final = (Model_Inference * 0.6) + (Sentiment_Impact * 0.4)`
5. **Persistence**: All signals, news snippets, and reasoning are stored as JSON in the cloud DB for permanent, cross-session retrieval.
6. **Visualization**: Results are rendered on a high-fidelity dashboard with interactive diamond-point targets on the price chart.

---

## 🛡️ License
Distributed under the MIT License. See `LICENSE` for more information.

---
**Quant-Engine v3.5** | Developed for Indian Markets (NSE)
