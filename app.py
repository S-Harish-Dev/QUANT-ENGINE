# Standard & Third-Party Library Imports
import os
import json
import gc
from datetime import datetime, timedelta

import streamlit as st
st.set_page_config(
    page_title="QUANT-ENGINE | Indian Market Price Inference Dashboard",
    page_icon="🇮🇳",
    layout="wide",
    initial_sidebar_state="collapsed"
)

import plotly.graph_objects as go

# Internal Architecture
import db_manager
import news_manager

# --- State Management ---
if 'view_mode' not in st.session_state:
    st.session_state.view_mode = None
if 'scroll_trigger' not in st.session_state:
    st.session_state.scroll_trigger = 0
if 'force_refresh' not in st.session_state:
    st.session_state.force_refresh = False

# --- UI Styling & Core Branding ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    
    /* Global Button Styling */
    div.stButton > button {
        width: 100% !important;
        background: linear-gradient(90deg, #0ea5e9 0%, #2563eb 100%) !important;
        color: white !important;
        border: none !important;
        height: 55px !important;
        font-weight: 700 !important;
        border-radius: 12px !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-size: 0.85rem !important;
        white-space: nowrap !important;
    }
    div.stButton > button:hover {
        box-shadow: 0 0 20px rgba(14, 165, 233, 0.4) !important;
        transform: scale(1.02) !important;
    }

    /* Red Reset Button Styling for Sidebar */
    [data-testid="stSidebar"] .stButton button {
        background: linear-gradient(90deg, #ef4444 0%, #b91c1c 100%) !important;
        color: white !important;
        height: 45px !important;
        font-weight: 800 !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        white-space: normal !important; /* Allow wrapping if needed */
        line-height: 1.2 !important;
        margin-top: 15px !important;
    }
    [data-testid="stSidebar"] .stButton button:hover {
        box-shadow: 0 0 15px rgba(239, 68, 68, 0.5) !important;
    }
    
    /* Remove top padding and hide header */
    /* Restore sidebar visibility (uncollapse button) */
    .block-container { padding-top: 1rem !important; padding-bottom: 0rem !important; }
    header { visibility: visible !important; height: auto !important; }
    .stAppDeployButton { display: none !important; }
    
    /* Hide Streamlit footer and remove wasted space */
    footer { visibility: hidden; height: 0px; }
    .element-container { margin-bottom: 0rem !important; }
    
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    
    .main { background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%); color: #f8fafc; }
    .stApp { background: transparent; }

    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 20px;
        border-radius: 14px;
        backdrop-filter: blur(10px);
        margin-bottom: 15px;
        transition: transform 0.3s ease, border-color 0.3s ease, box-shadow 0.3s ease;
        cursor: default;
        
        /* GPU acceleration fixes for backdrop-filter ghosting on scroll */
        transform: translateZ(0);
        backface-visibility: hidden;
        will-change: transform, backdrop-filter;
    }
    .metric-card:hover {
        transform: translateY(-8px);
        border-color: #38bdf8;
        box-shadow: 0 10px 30px rgba(56, 189, 248, 0.2);
    }

    .stSelectbox div[data-baseweb="select"] {
        background-color: #1e293b !important;
        border: 1px solid #334155 !important;
        border-radius: 12px !important;
        height: 50px !important;
        font-size: 1rem !important;
    }
    .stSelectbox label { color: #94a3b8 !important; font-weight: 600 !important; }

    .status-up { color: #10b981; font-weight: 800; }
    .status-down { color: #ef4444; font-weight: 800; }
    
    .header-box { text-align: center; padding: 40px 0 20px 0; }
    .header-box h1 {
        font-weight: 800; font-size: 3rem;
        background: linear-gradient(90deg, #38bdf8, #818cf8);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
    }
    .header-box p { color: #94a3b8; font-size: 1rem; margin-top: -5px; }

    .accuracy-badge {
        background: rgba(14, 165, 233, 0.15);
        color: #38bdf8;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 700;
        border: 1px solid rgba(56, 189, 248, 0.3);
        margin: 0 5px;
    }
    .market-badge {
        background: rgba(16, 185, 129, 0.15);
        color: #10b981;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 800;
        border: 1px solid rgba(16, 185, 129, 0.3);
        margin-bottom: 12px;
        display: inline-block;
        letter-spacing: 0.5px;
    }

    .gauge-wrapper { position: relative; margin: 30px auto 40px auto; max-width: 700px; }
    .gauge-bar { height: 18px; background: #334155; border-radius: 9px; position: relative; overflow: visible; }
    
    .gauge-fill { 
        height: 100%; border-radius: 9px; 
        background: linear-gradient(90deg, #10b981, #38bdf8); 
        width: 0%;
    }
    
    .gauge-pointer {
        position: absolute; top: -35px; transform: translateX(-50%);
        background: #38bdf8; color: #0f172a; padding: 5px 12px;
        border-radius: 8px; font-weight: 800; font-size: 0.9rem;
        box-shadow: 0 0 15px #38bdf8;
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    .gauge-pointer::after {
        content: ''; position: absolute; bottom: -6px; left: 50%; transform: translateX(-50%);
        border-left: 6px solid transparent; border-right: 6px solid transparent; border-top: 6px solid #38bdf8;
    }
    .gauge-wrapper:hover .gauge-pointer { opacity: 1; }
    
    .gauge-labels { display: flex; justify-content: space-between; margin-top: 8px; color: #94a3b8; font-size: 0.8rem; font-weight: 700; }
    .arrow-down { color: #ef4444; }
    .arrow-up { color: #10b981; }

    .chart-container {
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        padding-top: 25px;
        margin-top: 40px;
    }
    
    /* Responsive News Images */
    [data-testid="stImage"] {
        border-radius: 16px !important;
        overflow: hidden !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
    }

    /* Center the Spinner */
    [data-testid="stSpinner"] {
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        width: 100% !important;
        margin: 40px 0 !important;
    }
    [data-testid="stSpinner"] > div {
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        gap: 10px;
        width: 100%;
    }

    /* News Card Interactivity */
    .news-item {
        transition: transform 0.3s ease, box-shadow 0.3s ease !important;
        border: 1px solid transparent !important; /* Invisible border unless hovered */
        border-left: 4px solid #38bdf8 !important; /* Maintain left accent */
    }
    .news-item:hover {
        transform: translateY(-5px) !important;
        box-shadow: 0 10px 20px -5px rgba(56, 189, 248, 0.4) !important;
        border: 1px solid rgba(56, 189, 248, 0.3) !important;
        border-left: 4px solid #38bdf8 !important;
    }

    /* Mobile Optimization */
    @media (max-width: 768px) {
        .header-box h1 { font-size: 2.2rem !important; }
        .metric-card { padding: 10px 15px !important; margin-bottom: 8px !important; }
        
        /* Mobile Grid Layout - The Ultimate Fix */
        [data-testid="stVerticalBlock"] > div:has(#mobile_btn_anchor) + div [data-testid="stHorizontalBlock"] {
            display: grid !important;
            grid-template-columns: 1fr 1fr !important;
            gap: 2px !important;
            width: 92% !important;
            margin: -20px auto 0 auto !important;
            flex-direction: row !important; /* Fallback */
        }
        
        [data-testid="stVerticalBlock"] > div:has(#mobile_btn_anchor) + div [data-testid="column"] {
            width: 100% !important;
            flex: 1 1 auto !important;
            min-width: 0 !important;
        }

        [data-testid="stVerticalBlock"] > div:has(#mobile_btn_anchor) + div button {
            width: 100% !important;
            min-width: 0 !important;
            padding: 0px 4px !important;
            font-size: 0.75rem !important;
            height: 50px !important;
            line-height: 1.1 !important;
            white-space: normal !important;
            overflow: hidden !important;
            text-overflow: ellipsis !important;
        }

        .gauge-wrapper { max-width: 100%; padding: 0 10px; }
        .gauge-pointer { font-size: 0.75rem !important; padding: 3px 8px !important; }
        
        /* Reduce 'No News' image height by 50% on mobile */
        [data-testid="stImage"] img {
            max-height: 200px !important;
            object-fit: cover !important;
        }
        
        /* Add padding space for Technical Engine Specs expander */
        [data-testid="stExpander"] {
            margin-top: 25px !important;
            padding: 10px !important;
        }

        /* Mobile Chart Scaling */
        .mobile-chart {
            transform: scale(0.5);
            transform-origin: top center;
        }
    }
</style>

<script>
    function scrollToResults() {
        const resultsHeader = Array.from(document.querySelectorAll('h3, h2, h1')).find(el => el.textContent.includes('Result Analysis'));
        if (resultsHeader) {
            resultsHeader.scrollIntoView({ behavior: 'smooth', block: 'start' });
            return true;
        }
        return false;
    }

    // Check immediately and then observe
    if (!scrollToResults()) {
        const scrollObserver = new MutationObserver((mutations) => {
            if (scrollToResults()) {
                scrollObserver.disconnect();
            }
        });
        scrollObserver.observe(document.body, { childList: true, subtree: true });
        
        // Fallback for fast rendering
        setTimeout(scrollToResults, 500);
        setTimeout(scrollToResults, 2000);
    }
</script>
""", unsafe_allow_html=True)

# --- Deep Learning Ensemble Architecture ---
def get_meta_dl_model(input_dim):
    import torch.nn as nn
    class MetaDL(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.conv = nn.Conv1d(input_dim, 32, 3, padding=1)
            self.lstm = nn.LSTM(32, 64, batch_first=True, bidirectional=True)
            self.attn = nn.MultiheadAttention(128, 4, batch_first=True)
            self.fc = nn.Linear(128, 2)
            self.drop = nn.Dropout(0.3)

        def forward(self, x):
            out = self.conv(x.permute(0, 2, 1)).permute(0, 2, 1)
            out, _ = self.lstm(out)
            attn_out, _ = self.attn(out, out, out)
            out = attn_out.mean(1)
            return self.fc(self.drop(out))
    return MetaDL(input_dim)

# --- Feature Engineering ---
@st.cache_data
def engineer_features(df):
    df = df.copy()
    df['Body'] = (df['Close'] - df['Open']) / df['Open']
    df['Upper_Shadow'] = (df['High'] - df[['Open', 'Close']].max(axis=1)) / df['Open']
    df['Lower_Shadow'] = (df[['Open', 'Close']].min(axis=1) - df['Low']) / df['Open']
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + gain / (loss + 1e-10)))
    df['VWAP'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / df['Volume'].cumsum()
    df['Dist_VWAP'] = (df['Close'] - df['VWAP']) / df['VWAP']
    for lag in [1, 2, 3, 5]:
        df[f'Ret_{lag}'] = df['Close'].pct_change(lag)
    df['Vol_5'] = df['Close'].pct_change().rolling(5).std()
    return df.dropna()

# --- Optimization: Cached Model Loader ---
@st.cache_resource
def load_pkl_model(ticker):
    import torch
    import pickle
    import io
    file_path = os.path.join(os.path.dirname(__file__), "Stock_Models", f"{ticker}_model.pkl")
    if not os.path.exists(file_path): return None
    
    # Custom Unpickler for high-speed CPU loading of GPU-saved models
    class CPU_Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == 'torch.storage' and name == '_load_from_bytes':
                return lambda b: torch.load(io.BytesIO(b), map_location='cpu', weights_only=False)
            return super().find_class(module, name)

    try:
        with open(file_path, "rb") as f:
            return CPU_Unpickler(f).load()
    except Exception as e:
        print(f"Error loading model {ticker}: {e}")
        return None

# --- Sidebar Controls ---
with st.sidebar:
    st.markdown("### 🛠️ Engine Controls")
    st.info("Bypass local cache")
    if st.button("🔥 RESET", use_container_width=True):
        st.cache_data.clear()
        st.session_state.force_refresh = True
        st.toast("Engine Reset: Refreshing all data...", icon="🔥")

# --- Dynamic Header Placeholder ---
header_placeholder = st.empty()

# Instant header render to prevent frame-shutter while DB syncs
header_placeholder.markdown(f"""
<div class='header-box' style='padding-bottom: 10px;'>
    <div class='market-badge'>🇮🇳 SPECIALLY TUNED FOR INDIAN MARKETS</div>
    <h1>QUANT-ENGINE</h1>
    <p>Hybrid ML Forecasting & Macro-Economic Sentiment Intelligence</p>
    <div style='margin-top: 15px;'>
        <span class='accuracy-badge'>Signal Hit Rate: ...</span>
        <span class='accuracy-badge'>Avg. Price Error: ...</span>
    </div>
</div>
""", unsafe_allow_html=True)

# --- Primary Asset Selector ---
INDIAN_STOCKS = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS"]
st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
c1, c2, c3 = st.columns([1, 1.5, 1])
with c2:
    selected_ticker = st.selectbox("Market Asset", INDIAN_STOCKS)

# --- View Control Callbacks ---
def set_daily_view():
    st.session_state.view_mode = "daily"
    st.session_state.scroll_trigger += 1

def set_weekly_view():
    st.session_state.view_mode = "weekly"
    st.session_state.scroll_trigger += 1

# --- Action Controls (Prefetched for Instant UI) ---
with st.container():
    _, mid_col, _ = st.columns([1, 1.5, 1])
    with mid_col:
        # Anchor for mobile CSS targeting
        st.markdown('<div id="mobile_btn_anchor"></div>', unsafe_allow_html=True)
        col1, col2 = st.columns([1,1], gap="small")

        with col1:
            st.button("⚡ NEXT DAY", use_container_width=True, on_click=set_daily_view)

        with col2:
            st.button("📈 WEEKLY TREND", use_container_width=True, on_click=set_weekly_view)

# --- Background Data Orchestration (Silent Update) ---
db_manager.update_stock_cache_efficient(selected_ticker)
db_manager.update_inference_actuals(selected_ticker)
stats = db_manager.get_accuracy_stats(selected_ticker)

# --- Update Header with Real Stats ---
trend_acc = f"{stats['trend_accuracy']:.1f}% ✓" if stats['trend_accuracy'] is not None else "N/A"
mae_val = f"₹{stats['avg_mae']:.2f}" if stats['avg_mae'] is not None else "N/A"

header_placeholder.markdown(f"""
<div class='header-box' style='padding-bottom: 10px;'>
    <div class='market-badge'>🇮🇳 SPECIALLY TUNED FOR INDIAN MARKETS</div>
    <h1>QUANT-ENGINE</h1>
    <p>Hybrid ML Forecasting & Macro-Economic Sentiment Intelligence</p>
    <div style='margin-top: 15px;'>
        <span class='accuracy-badge'>Signal Hit Rate: {trend_acc}</span>
        <span class='accuracy-badge'>Avg. Price Error: {mae_val}</span>
    </div>
</div>
""", unsafe_allow_html=True)


if st.session_state.view_mode:
    model_data = load_pkl_model(selected_ticker)
    
    if model_data is None:
        st.error(f"Engine Core [ {selected_ticker} ] not found in system storage.")
    else:
        # --- Quantitative Inference Engine ---
        with st.spinner("Executing Quant Algorithm..."):
            # Fetch from cloud persistence
            df = db_manager.get_cached_data(selected_ticker)
            
            if df.empty:
                st.error(f"Insufficient market data for {selected_ticker}. Please try again later.")
                st.stop()
            
            latest_price = df['Close'].iloc[-1]
            last_date = df.index[-1]
            
            # Determine target timeline
            target_days = 1 if st.session_state.view_mode == "daily" else 5
            target_date = last_date + timedelta(days=target_days)
            if target_date.weekday() >= 5: target_date += timedelta(days=2) # Skip weekends
            
            target_date_str = target_date.strftime("%Y-%m-%d")
            
            # Use cached inference if available within TTL window (Skip if forcing refresh)
            recent_inference = db_manager.get_recent_inference(selected_ticker, target_date_str)
            created_time = "N/A"
            from_cache = False
            
            if recent_inference:
                try:
                    created_dt = datetime.fromisoformat(recent_inference['created_at'])
                    created_time = created_dt.strftime("%H:%M")
                except:
                    created_time = "Recent"

            if recent_inference and not st.session_state.force_refresh:
                from_cache = True
                final_weighted_price = recent_inference['target_price']
                combined_move_pct = ((final_weighted_price / latest_price) - 1) * 100
                
                # RETRIEVE GENUINE METRICS FROM CACHED INFERENCE
                final_prob = recent_inference.get('final_prob')
                xgb_prob = recent_inference.get('xgb_prob')
                dl_prob = recent_inference.get('dl_prob')
                sentiment_move_pct = recent_inference.get('sentiment_move')
                
                # Robust null-handling for data consistency
                if sentiment_move_pct is None: sentiment_move_pct = 0.0
                if final_prob is None:
                    final_prob = 0.75 if recent_inference['direction'] == "UP" else 0.25
                
                est_price = final_weighted_price
                
                news_json = recent_inference.get('news_json')
                sentiment_json = recent_inference.get('sentiment_json')
                
                try:
                    news = json.loads(news_json) if news_json else []
                except Exception as e:
                    news = []
                    
                try:
                    sentiment_data = json.loads(sentiment_json) if sentiment_json else None
                    if sentiment_data is None:
                        raise ValueError("Empty sentiment data")
                    
                    # SURFACE CACHED ERRORS: Check if the cached reason contains an error message
                    cached_reason = sentiment_data.get('reason', '')
                    if "Usage Limit" in cached_reason or "Service Error" in cached_reason:
                        st.warning(f"🤖 **Stored Gemini AI Status**: {cached_reason}")
                        
                except Exception as e:
                    # Generic fallback for corrupted or legacy JSON data
                    sentiment_data = {
                        "reason": "Referencing cached institutional inference.", 
                        "relevant": False,
                        "trend": recent_inference['direction'],
                        "expectation": sentiment_move_pct
                    }
            else:
                import torch
                import torch.nn.functional as F
                
                df_feat = engineer_features(df)
                X_scaled = model_data['scaler'].transform(df_feat[model_data['features']].values)
                
                seq_len = 15
                latest_seq = X_scaled[-seq_len:]
                latest_flat = latest_seq.flatten().reshape(1, -1)
                
                xgb_prob = model_data['xgb'].predict_proba(latest_flat)[0, 1]
                dl_model = get_meta_dl_model(model_data['input_dim']).to("cpu")
                state_dict = {k: v.cpu() for k, v in model_data['dl_state'].items()}
                dl_model.load_state_dict(state_dict)
                dl_model.eval()
                
                with torch.no_grad():
                    dl_out = dl_model(torch.FloatTensor(latest_seq).unsqueeze(0))
                    dl_prob = F.softmax(dl_out, dim=1)[0, 1].item()
                
                final_prob = float((xgb_prob + dl_prob) / 2)
                
                # Price Inference Logic
                inference_dir = "UP" if final_prob > 0.5 else "DOWN"
                avg_vol = df['Close'].pct_change().abs().mean()
                est_move = latest_price * avg_vol * target_days
                est_price = latest_price + est_move if inference_dir == "UP" else latest_price - est_move
                
                # Fetch Sentiment & News
                news = news_manager.fetch_news(selected_ticker)
                
                # Report News API Errors
                if news_manager.API_ERROR_STATE["news_api_error"]:
                    st.error(f"📡 **News API Error**: {news_manager.API_ERROR_STATE['news_api_error']}")
                
                sentiment_data = news_manager.analyze_sentiment(selected_ticker, news)
                
                # Report Gemini API Errors
                if news_manager.API_ERROR_STATE["gemini_api_error"]:
                    st.warning(f"🤖 **Gemini AI Intelligence Error**: {news_manager.API_ERROR_STATE['gemini_api_error']}")
                
                # Weighted Ensemble Logic: 60% Technical Signal, 40% Macro Sentiment
                ml_move_pct = ((est_price / latest_price) - 1) * 100
                sentiment_move_pct = sentiment_data['expectation'] if sentiment_data.get('relevant') else 0.0
                
                combined_move_pct = (ml_move_pct * 0.6) + (sentiment_move_pct * 0.4)
                final_weighted_price = latest_price * (1 + combined_move_pct / 100)
                
                # Log inference for historical accuracy tracking
                db_manager.log_price_inference(
                    ticker=selected_ticker,
                    inference_date=last_date.strftime("%Y-%m-%d"),
                    target_date=target_date_str,
                    direction="UP" if combined_move_pct > 0 else "DOWN",
                    target_price=float(final_weighted_price),
                    final_prob=float(final_prob),
                    xgb_prob=float(xgb_prob),
                    dl_prob=float(dl_prob),
                    sentiment_move=float(sentiment_move_pct),
                    news_json=json.dumps(news),
                    sentiment_json=json.dumps(sentiment_data)
                )
                
                # Reset force flag after successful execution
                st.session_state.force_refresh = False
                
                # Cleanup local variables to optimize memory usage
                if 'dl_model' in locals(): del dl_model
                gc.collect()

        # --- Results UI (Moved OUT of spinner for responsiveness) ---
        if from_cache:
            st.info(f"✨ Using recent Price Inference from cache (Created: {created_time})")

        # Result Analysis Header
        st.markdown("<div id='result_anchor'></div>", unsafe_allow_html=True)
        st.markdown(f"### Result Analysis: `{selected_ticker}`")
        
        # Robust Auto-scroll with Retry Logic
        # Robust Auto-scroll with Retry Logic (Triggered by unique ID)
        st.components.v1.html(
            f"""
            <script>
                // Trigger ID: {st.session_state.scroll_trigger}
                var count = 0;
                var interval = setInterval(function() {{
                    var element = window.parent.document.getElementById('result_anchor');
                    if (element) {{
                        element.scrollIntoView({{behavior: 'smooth', block: 'start'}});
                        clearInterval(interval);
                    }}
                    count++;
                    if (count > 50) clearInterval(interval);
                }}, 100);
            </script>
            """, height=0, width=0
        )
        
        top_col1, top_col2, top_col3 = st.columns(3)
        
        with top_col1:
            st.markdown(f"""
            <div class='metric-card'>
                <small style='color:#94a3b8'>Current Price</small>
                <div style='font-size:1.9rem; font-weight:800;'>₹{latest_price:,.2f}</div>
                <small style='color:{"#10b981" if df['Close'].pct_change().iloc[-1] > 0 else "#ef4444"}'>
                    {df['Close'].pct_change().iloc[-1]*100:+.2f}% Today
                </small>
            </div>
            """, unsafe_allow_html=True)

        with top_col2:
            weighted_inference = "UP" if combined_move_pct > 0 else "DOWN"
            inference_text = "BULLISH" if weighted_inference == "UP" else "BEARISH"
            status_class = "status-up" if weighted_inference == "UP" else "status-down"
            st.markdown(f"""
            <div class='metric-card'>
                <small style='color:#94a3b8'>Signal Strength</small>
                <div class='{status_class}' style='font-size:1.9rem;'>{inference_text} {'▲' if inference_text=='BULLISH' else '▼'}</div>
                <small style='color:#38bdf8'>Sentiment Impact: {sentiment_move_pct:+.2f}%</small>
            </div>
            """, unsafe_allow_html=True)
        
        with top_col3:
            st.markdown(f"""
            <div class='metric-card'>
                <small style='color:#94a3b8'>{'Next Day' if st.session_state.view_mode == 'daily' else 'Weekly'} Target</small>
                <div style='font-size:1.9rem; font-weight:800; color:#38bdf8'>₹{final_weighted_price:,.2f}</div>
                <small style='color:#94a3b8'>Weighted Est: {combined_move_pct:+.2f}%</small>
            </div>
            """, unsafe_allow_html=True)

        # --- Dashboard Metrics Layout ---
        st.markdown(f"<h4 style='text-align:center; color:#f8fafc; margin-bottom:5px; margin-top:25px;'>Signal Strength & Timeline</h4>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align:center; color:#94a3b8; font-weight:600; margin-bottom:0;'>{last_date.strftime('%d/%m/%y')} ➔ <b>{target_date.strftime('%d/%m/%y')}</b></p>", unsafe_allow_html=True)
        
        prob_pct = final_prob * 100
        anim_id = f"fill_{int(prob_pct)}_{st.session_state.scroll_trigger}"
        
        st.markdown(f"""
        <style>
            @keyframes {anim_id} {{
                from {{ width: 0%; }}
                to {{ width: {prob_pct:.1f}%; }}
            }}
            
            @keyframes pop_tooltip_{st.session_state.scroll_trigger} {{
                0% {{ opacity: 0; transform: translateX(-50%) translateY(10px); }}
                10% {{ opacity: 1; transform: translateX(-50%) translateY(0); }} /* Pop up quickly */
                80% {{ opacity: 1; transform: translateX(-50%) translateY(0); }} /* Stay visible */
                100% {{ opacity: 0; transform: translateX(-50%) translateY(10px); }} /* Fade out */
            }}

            .gauge-fill-active {{
                animation: {anim_id} 1.2s cubic-bezier(0.16, 1, 0.3, 1) 0.5s forwards;
            }}
            
            .gauge-pointer-anim {{
                /* Delay = 0.2s (gauge delay) + 1.2s (gauge duration) = 1.4s total wait */
                animation: pop_tooltip_{st.session_state.scroll_trigger} 2.5s ease 1.4s forwards;
            }}
            
            /* Ensure it shows on hover regardless of animation state */
            .gauge-bar:hover .gauge-pointer {{ 
                opacity: 1 !important; 
                transform: translateX(-50%) translateY(0) !important;
                animation: none !important;
            }}
        </style>
        <div class='gauge-wrapper'>
            <div class='gauge-bar'>
                <div class='gauge-fill gauge-fill-active'></div>
                <div class='gauge-pointer gauge-pointer-anim' style='left: {prob_pct:.1f}%;'>{prob_pct:.1f}%</div>
            </div>
            <div class='gauge-labels'>
                <span><span class='arrow-down'>▼</span> 0%</span>
                <span>100% <span class='arrow-up'>▲</span></span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # --- Plotly Forecasting Visualization ---
        fig = go.Figure()
        
        fig.add_trace(go.Candlestick(x=df.index[-150:],
            open=df['Open'].tail(150), high=df['High'].tail(150),
            low=df['Low'].tail(150), close=df['Close'].tail(150),
            name='Market Context'))
            
        history = db_manager.get_inference_history(selected_ticker)
        
        # Prepare points for the continuous AI line
        ai_x = []
        ai_y = []
        
        if history:
            # History comes sorted by target_date DESC from DB, reverse it
            history_sorted = history[::-1]
            ai_x = []
            for h in history_sorted:
                td = h['target_date']
                if isinstance(td, str):
                    ai_x.append(datetime.strptime(td, "%Y-%m-%d"))
                else:
                    # Handle date/datetime objects from SQLAlchemy/Pandas
                    ai_x.append(datetime(td.year, td.month, td.day))
            ai_y = [h['target_price'] for h in history_sorted]
        
        # Add the current prediction if it's newer than the last history point
        if not ai_x or target_date > ai_x[-1]:
            ai_x.append(target_date)
            ai_y.append(est_price)
        elif ai_x and target_date == ai_x[-1]:
            # Update last point if it's the same date (e.g. fresh prediction for same day)
            ai_y[-1] = est_price

        # Plot the continuous Forecasting line
        if ai_x:
            fig.add_trace(go.Scatter(
                x=ai_x, 
                y=ai_y, 
                mode='markers+lines',
                name='AI Forecast Path',
                line=dict(color='#38bdf8', width=2, dash='dot'),
                marker=dict(color='#38bdf8', size=6),
                hovertemplate="<b>Forecast:</b> ₹%{y:,.2f}<br><b>Date:</b> %{x|%b %d, %Y}<extra></extra>"
            ))

        # Highlight the final target Diamond
        fig.add_trace(go.Scatter(
            x=[target_date], 
            y=[est_price], 
            mode='markers+text',
            name='AI Target', 
            text=[f"₹{est_price:,.0f}"], 
            textposition="top center",
            marker=dict(color='#38bdf8', size=14, symbol='diamond'),
            hovertemplate=f"<b>Target:</b> ₹{est_price:,.2f}<br><b>Date:</b> {target_date.strftime('%b %d, %Y')}<extra></extra>"
        ))

        start_zoom = df.index[-22] if len(df) > 22 else df.index[0]
        end_zoom = target_date + timedelta(days=2)

        fig.update_layout(
            template="plotly_dark",
            dragmode='pan',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis_rangeslider_visible=False,
            height=500,
            margin=dict(l=10, r=10, t=20, b=10),
            xaxis=dict(range=[start_zoom, end_zoom], showgrid=False, linecolor='#334155', linewidth=1, mirror=True),
            yaxis=dict(showgrid=True, gridcolor='rgba(51,65,85,0.3)', linecolor='#334155', linewidth=1, mirror=True),
            legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99, bgcolor="rgba(0,0,0,0)")
        )
        
        st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
        st.markdown('<div class="mobile-chart">', unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # --- Intelligence: News & Sentiment Cards ---
        st.markdown("### 📰 Market Sentiment & Intelligence")
        
        if not news:
            import random
            img_dir = os.path.join(os.path.dirname(__file__), "images")
            try:
                img_files = [f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
                random_img = random.choice(img_files) if img_files else None
            except:
                random_img = None
            
            n_col1, n_col2 = st.columns([1, 2])
            with n_col1:
                if random_img:
                    st.image(os.path.join("images", random_img), use_container_width=True)
                else:
                    st.markdown("<div style='font-size: 5rem; text-align: center; margin-bottom: 20px;'>🔍</div>", unsafe_allow_html=True)
            
            with n_col2:
                st.markdown(f"""
                <div style='background: rgba(30, 41, 59, 0.4); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 16px; padding: 30px; height: 100%; display: flex; flex-direction: column; justify-content: center; min-height: 200px;'>
                    <h3 style='color: #f8fafc; margin-bottom: 12px; font-size: 1.3rem; font-weight: 700;'>No Relevant News Detected</h3>
                    <p style='color: #94a3b8; font-size: 1rem; line-height: 1.6; margin: 0;'>
                        Our engine scanned for <b>{selected_ticker}</b> news in the last 72 hours but found no high-impact market data. 
                    </p>
                </div>
                """, unsafe_allow_html=True)
        else:
            s1, s2 = st.columns([1, 2])
            with s1:
                st.markdown(f"""
                <div class='metric-card'>
                    <small style='color:#94a3b8'>Gemini 2.5 Intelligence</small>
                    <div style='font-size:1.4rem; font-weight:700; color:{"#10b981" if sentiment_data["trend"]=="UP" else "#ef4444" if sentiment_data["trend"]=="DOWN" else "#94a3b8"};'>
                        {sentiment_data["trend"]} ({sentiment_data["expectation"]:+.1f}%)
                    </div>
                    <p style='font-size:0.85rem; color:#f8fafc; margin-top:10px;'>{sentiment_data["reason"]}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with s2:
                if news:
                    for n in news[:3]:
                        st.markdown(f"""
                        <a href='{n.get("link")}' target='_blank' style='text-decoration: none; color: inherit;'>
                            <div style='background:rgba(30,41,59,0.3); padding:15px; border-radius:12px; margin-bottom:12px; border-left:4px solid #38bdf8; cursor: pointer;' class='news-item'>
                                <div style='font-weight:600; font-size:0.95rem; margin-bottom: 5px; color: #f1f5f9;'>{n.get('title')}</div>
                                <div style='display: flex; justify-content: space-between; align-items: center;'>
                                    <small style='color:#38bdf8; font-weight: 600;'>{n.get('source').upper()}</small>
                                    <small style='color:#64748b;'>{n.get('date')}</small>
                                </div>
                            </div>
                        </a>
                        """, unsafe_allow_html=True)

        st.markdown("<div style='height: 25px;'></div>", unsafe_allow_html=True)
        with st.expander("🔧 Technical Engine Specs"):
            if 'xgb_prob' in locals() and xgb_prob is not None:
                st.write(f"**XGBoost Probability:** {xgb_prob:.4f}")
            if 'dl_prob' in locals() and dl_prob is not None:
                st.write(f"**Attention-LSTM Probability:** {dl_prob:.4f}")
            st.write(f"**Meta-Learner Probability:** {final_prob:.4f}")
            st.write(f"**Cached Samples:** {len(df)} days")

        st.markdown("<div style='text-align:center; color:#64748b; font-size:0.75rem; margin-top:30px;'>Quant-Engine v3.5 | Hybrid Cloud Caching Active</div>", unsafe_allow_html=True)
