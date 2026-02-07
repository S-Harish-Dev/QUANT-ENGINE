import os
import pandas as pd
import sqlalchemy
from datetime import datetime, timedelta

try:
    import streamlit as st
except ImportError:
    st = None


def get_connection():
    """Establish PostgreSQL connection via Streamlit's SQL utility."""
    if st:
        return st.connection("postgresql", type="sql")
    
    raise RuntimeError("Missing Streamlit environment. Database requires st.connection support.")


def init_db():
    """Initialize schema for stock data and prediction history."""
    conn = get_connection()
    with conn.session as session:
        session.execute(st.text("""
            CREATE TABLE IF NOT EXISTS stock_data (
                id SERIAL PRIMARY KEY,
                ticker VARCHAR(20) NOT NULL,
                date DATE NOT NULL,
                open DOUBLE PRECISION,
                high DOUBLE PRECISION,
                low DOUBLE PRECISION,
                close DOUBLE PRECISION,
                volume BIGINT,
                UNIQUE(ticker, date)
            )
        """))
        
        session.execute(st.text("""
            CREATE TABLE IF NOT EXISTS predictions (
                id SERIAL PRIMARY KEY,
                ticker VARCHAR(20) NOT NULL,
                inference_date DATE NOT NULL,
                target_date DATE NOT NULL,
                direction VARCHAR(10),
                target_price DOUBLE PRECISION,
                actual_price DOUBLE PRECISION,
                was_correct INTEGER,
                mae DOUBLE PRECISION,
                final_prob DOUBLE PRECISION,
                xgb_prob DOUBLE PRECISION,
                dl_prob DOUBLE PRECISION,
                sentiment_move DOUBLE PRECISION,
                news_json TEXT,
                sentiment_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(ticker, inference_date, target_date)
            )
        """))
        session.commit()


CACHE_DAYS = 90

def update_stock_cache(ticker: str) -> "pd.DataFrame":
    """Sync cloud cache with latest market data via yfinance."""
    conn = get_connection()
    
    df_max = conn.query(
        "SELECT MAX(date) as max_date FROM stock_data WHERE ticker = :ticker",
        params={"ticker": ticker},
        ttl=0
    )
    last_cached_date = df_max['max_date'].iloc[0] if not df_max.empty and pd.notnull(df_max['max_date'].iloc[0]) else None
    
    today = datetime.now().date()
    
    if last_cached_date:
        last_date = last_cached_date if not isinstance(last_cached_date, str) else datetime.strptime(last_cached_date, "%Y-%m-%d").date()
        days_missing = (today - last_date).days
        
        if days_missing > 0:
            import yfinance as yf
            start_date = last_date + timedelta(days=1)
            df_new = yf.download(ticker, start=start_date.strftime("%Y-%m-%d"), end=(today + timedelta(days=1)).strftime("%Y-%m-%d"), progress=False)
            
            if not df_new.empty:
                if isinstance(df_new.columns, pd.MultiIndex):
                    df_new.columns = df_new.columns.get_level_values(0)
                
                with conn.session as session:
                    for idx, row in df_new.iterrows():
                        session.execute(st.text("""
                            INSERT INTO stock_data (ticker, date, open, high, low, close, volume)
                            VALUES (:ticker, :date, :open, :high, :low, :close, :volume)
                            ON CONFLICT (ticker, date) DO UPDATE SET
                                open = EXCLUDED.open, high = EXCLUDED.high, low = EXCLUDED.low,
                                close = EXCLUDED.close, volume = EXCLUDED.volume
                        """), {
                            "ticker": ticker, "date": idx.date(), "open": float(row['Open']),
                            "high": float(row['High']), "low": float(row['Low']),
                            "close": float(row['Close']), "volume": int(row['Volume'])
                        })
                    
                    # Enforce 90-day rolling window
                    session.execute(st.text("DELETE FROM stock_data WHERE ticker = :ticker AND date < :cutoff"), 
                                    {"ticker": ticker, "cutoff": today - timedelta(days=CACHE_DAYS)})
                    session.commit()
    else:
        import yfinance as yf
        start_date = today - timedelta(days=CACHE_DAYS)
        df_new = yf.download(ticker, start=start_date.strftime("%Y-%m-%d"), end=(today + timedelta(days=1)).strftime("%Y-%m-%d"), progress=False)
        
        if not df_new.empty:
            if isinstance(df_new.columns, pd.MultiIndex):
                df_new.columns = df_new.columns.get_level_values(0)
            
            with conn.session as session:
                for idx, row in df_new.iterrows():
                    session.execute(st.text("""
                        INSERT INTO stock_data (ticker, date, open, high, low, close, volume)
                        VALUES (:ticker, :date, :open, :high, :low, :close, :volume)
                        ON CONFLICT (ticker, date) DO NOTHING
                    """), {
                        "ticker": ticker, "date": idx.date(), "open": float(row['Open']),
                        "high": float(row['High']), "low": float(row['Low']),
                        "close": float(row['Close']), "volume": int(row['Volume'])
                    })
                session.commit()
    
    return get_cached_data(ticker)


def get_cached_data(ticker: str, days: int = CACHE_DAYS) -> "pd.DataFrame":
    """Retrieve cached market data as a pandas DataFrame."""
    conn = get_connection()
    
    cutoff_date = datetime.now().date() - timedelta(days=days)
    
    df = conn.query(
        """
        SELECT date, open as "Open", high as "High", low as "Low", close as "Close", volume as "Volume"
        FROM stock_data
        WHERE ticker = :ticker AND date >= :cutoff
        ORDER BY date ASC
        """,
        params={"ticker": ticker, "cutoff": cutoff_date}
    )
    
    if not df.empty:
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
    
    return df


def log_price_inference(ticker: str, inference_date: str, target_date: str, 
                   direction: str, target_price: float, 
                   final_prob: float = None, xgb_prob: float = None, 
                   dl_prob: float = None, sentiment_move: float = None,
                   news_json: str = None, sentiment_json: str = None):
    """Upsert model inference and sentiment metadata to persistence layer."""
    conn = get_connection()
    
    if news_json is None: news_json = "[]"
    elif not isinstance(news_json, str):
        import json
        news_json = json.dumps(news_json)

    if sentiment_json is None: sentiment_json = "{}"
    elif not isinstance(sentiment_json, str):
        import json
        sentiment_json = json.dumps(sentiment_json)

    with conn.session as session:
        session.execute(st.text("""
            INSERT INTO predictions 
            (ticker, inference_date, target_date, direction, target_price, 
             final_prob, xgb_prob, dl_prob, sentiment_move, news_json, sentiment_json, created_at)
            VALUES (:ticker, :inf_date, :target_date, :direction, :target_price, 
             :final_prob, :xgb_prob, :dl_prob, :sentiment_move, :news_json, :sentiment_json, :created_at)
            ON CONFLICT (ticker, inference_date, target_date) DO UPDATE SET
                direction = EXCLUDED.direction,
                target_price = EXCLUDED.target_price,
                final_prob = EXCLUDED.final_prob,
                xgb_prob = EXCLUDED.xgb_prob,
                dl_prob = EXCLUDED.dl_prob,
                sentiment_move = EXCLUDED.sentiment_move,
                news_json = EXCLUDED.news_json,
                sentiment_json = EXCLUDED.sentiment_json,
                created_at = EXCLUDED.created_at
        """), {
            "ticker": ticker,
            "inf_date": inference_date,
            "target_date": target_date,
            "direction": direction,
            "target_price": target_price,
            "final_prob": final_prob,
            "xgb_prob": xgb_prob,
            "dl_prob": dl_prob,
            "sentiment_move": sentiment_move,
            "news_json": news_json,
            "sentiment_json": sentiment_json,
            "created_at": datetime.now()
        })
        session.commit()


# Wrapper to avoid double-syncing in some environments
def update_stock_cache_efficient(ticker: str):
    if st:
        @st.cache_data(ttl=3600, show_spinner=False)
        def MarketScan(t):
            return update_stock_cache(t)
        return MarketScan(ticker)
    return update_stock_cache(ticker)


def update_inference_actuals(ticker: str):
    """Calculate and update historical accuracy by comparing targets with actual close prices."""
    conn = get_connection()
    
    pending = conn.query("""
        SELECT id, target_date, direction, target_price
        FROM predictions
        WHERE ticker = :ticker AND actual_price IS NULL
    """, params={"ticker": ticker}, ttl=0)
    
    if pending.empty:
        return

    with conn.session as session:
        for _, pred in pending.iterrows():
            actual = conn.query("""
                SELECT close FROM stock_data
                WHERE ticker = :ticker AND date = :tdate
            """, params={"ticker": ticker, "tdate": pred['target_date']})
            
            if not actual.empty:
                actual_price = actual['close'].iloc[0]
                
                prev_row = conn.query("""
                    SELECT close FROM stock_data
                    WHERE ticker = :ticker AND date < :tdate
                    ORDER BY date DESC LIMIT 1
                """, params={"ticker": ticker, "tdate": pred['target_date']})
                
                if not prev_row.empty:
                    actual_direction = "UP" if actual_price > prev_row['close'].iloc[0] else "DOWN"
                    was_correct = 1 if actual_direction == pred['direction'] else 0
                else:
                    was_correct = None
                
                mae = abs(pred['target_price'] - actual_price)
                
                session.execute(st.text("""
                    UPDATE predictions
                    SET actual_price = :actual, was_correct = :correct, mae = :mae
                    WHERE id = :id
                """), {"actual": actual_price, "correct": was_correct, "mae": mae, "id": int(pred['id'])})
        session.commit()


def get_accuracy_stats(ticker: str) -> dict:
    """Get accuracy statistics for a ticker."""
    conn = get_connection()
    
    df = conn.query("""
        SELECT 
            COUNT(*) as total,
            SUM(CASE WHEN was_correct = 1 THEN 1 ELSE 0 END) as correct,
            AVG(mae) as avg_mae
        FROM predictions
        WHERE ticker = :ticker AND was_correct IS NOT NULL
    """, params={"ticker": ticker})
    
    if df.empty or df['total'].iloc[0] == 0:
        return {'trend_accuracy': None, 'trend_total': 0, 'trend_correct': 0, 'avg_mae': None}
    
    trend_total = int(df['total'].iloc[0])
    trend_correct = int(df['correct'].iloc[0])
    trend_accuracy = (trend_correct / trend_total * 100)
    avg_mae = df['avg_mae'].iloc[0]
    
    return {
        'trend_accuracy': trend_accuracy,
        'trend_total': trend_total,
        'trend_correct': trend_correct,
        'avg_mae': avg_mae
    }


def get_inference_history(ticker: str, limit: int = 30) -> list:
    """Get recent price inference history for charting."""
    conn = get_connection()
    
    df = conn.query("""
        SELECT target_date, target_price, actual_price, direction, was_correct
        FROM predictions
        WHERE ticker = :ticker
        ORDER BY target_date DESC
        LIMIT :limit
    """, params={"ticker": ticker, "limit": limit})
    
    return df.to_dict('records')


def get_recent_inference(ticker: str, target_date: str):
    """Check if an inference exists for this ticker and target date."""
    conn = get_connection()
    
    df = conn.query("""
        SELECT direction, target_price, final_prob, xgb_prob, dl_prob, sentiment_move, news_json, sentiment_json, created_at
        FROM predictions
        WHERE ticker = :ticker AND target_date = :tdate
        ORDER BY created_at DESC LIMIT 1
    """, params={"ticker": ticker, "tdate": target_date}, ttl=0)
    
    return df.iloc[0].to_dict() if not df.empty else None


# Schema initialization on module load
init_db()
