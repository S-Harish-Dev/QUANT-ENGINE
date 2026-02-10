import os
import json
import requests
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

load_dotenv()

# --- API Configuration ---
def get_secret(key, default=None):
    """Resolve API keys from st.secrets or environment variables."""
    try:
        import streamlit as st
        if key in st.secrets:
            return st.secrets[key]
    except:
        pass
    return os.getenv(key, default)

GEMINI_API_KEY = get_secret("GEMINI_API_KEY")
NEWSDATA_API_KEY = get_secret("NEWSDATA_API_KEY")

# --- Global API Error Tracking ---
API_ERROR_STATE = {
    "news_api_error": None,
    "gemini_api_error": None
}

def get_gemini_client():
    """Initialize Gemini client with lazy loading for efficiency."""
    if not GEMINI_API_KEY:
        return None
    from google import genai
    return genai.Client(api_key=GEMINI_API_KEY)

def fetch_news(ticker):
    """Aggregate market news using parallel company, sector, and macro queries."""
    ticker_info = {
        "RELIANCE.NS": {"name": "Reliance Industries", "keywords": "Reliance OR Ambani OR Jio OR RIL", "sector": "Energy OR Retail OR Telecom"},
        "TCS.NS": {"name": "Tata Consultancy Services", "keywords": "TCS OR Tata Consultancy OR Tata Group", "sector": "IT sector OR Software India"},
        "HDFCBANK.NS": {"name": "HDFC Bank", "keywords": "HDFC OR HDFC Bank", "sector": "Banking India OR Finance India"},
        "ICICIBANK.NS": {"name": "ICICI Bank", "keywords": "ICICI OR ICICI Bank", "sector": "Banking India OR Finance India"}
    }
    info = ticker_info.get(ticker, {"name": ticker.split('.')[0], "keywords": ticker.split('.')[0], "sector": ""})
    
    queries = [
        (f"\"{info['name']}\" OR {info['keywords']} stock news India", 1),
        (f"({info['sector']}) AND (market news OR stock news) India" if info['sector'] else f"{info['keywords']} India", 2),
        ("Nifty 50 OR Sensex OR Indian Economy OR RBI Policy news", 3)
    ]
    
    # Reset error state for new fetch
    API_ERROR_STATE["news_api_error"] = None

    def fetch_single_query(q_data):
        q, priority = q_data
        url = f"https://newsdata.io/api/1/news?apikey={NEWSDATA_API_KEY}&q={urllib.parse.quote(q)}&language=en"
        try:
            response = requests.get(url, timeout=10)
            data = response.json()
            if data.get("status") == "success":
                results = data.get("results", [])
                for r in results:
                    r["_priority"] = priority
                return results
            else:
                # Track specific API error message if provided by NewsData.io
                error_msg = data.get("results", {}).get("message") or data.get("results", {}).get("code") or "Unknown API Error"
                if "apikey" in error_msg.lower() or "unauthorized" in error_msg.lower():
                    API_ERROR_STATE["news_api_error"] = "Invalid NewsData API Key"
                elif "rate limit" in error_msg.lower() or "too many requests" in error_msg.lower():
                    API_ERROR_STATE["news_api_error"] = "NewsData API Rate Limit Reached"
                else:
                    API_ERROR_STATE["news_api_error"] = error_msg
        except Exception as e:
            print(f"Error fetching {q}: {e}")
            API_ERROR_STATE["news_api_error"] = str(e)
        return []

    all_results = []
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = executor.map(fetch_single_query, queries)
        for results in futures:
            all_results.extend(results)

    # Deduplicate results based on URL
    unique_links = set()
    dedup_news = []
    for r in all_results:
        link = r.get("link")
        if link and link not in unique_links:
            unique_links.add(link)
            dedup_news.append({
                "title": r.get("title"),
                "description": r.get("description", ""),
                "link": link,
                "source": r.get("source_id"),
                "date": r.get("pubDate"),
                "priority": r.get("_priority", 3)
            })

    # Hierarchy Fallback Grouping:
    # Group 1: Company News (Priority 1)
    # Group 2: Sector News (Priority 2)
    # Group 3: Macro News (Priority 3 - Fallback only)
    company_news = [n for n in dedup_news if n['priority'] == 1]
    sector_news = [n for n in dedup_news if n['priority'] == 2]
    macro_news = [n for n in dedup_news if n['priority'] == 3]

    # Strict Waterfall Selection:
    # Priority 1 + 2 are the 'Primary' signals. 
    # Macro (Priority 3) is ONLY used if no primary signals are found.
    if company_news or sector_news:
        final_list = company_news + sector_news
        # Secondary sort to ensure company news stays at the top
        final_list.sort(key=lambda x: x['priority'])
        return final_list[:8]
    
    # Fallback to macro if asset/sector data is completely missing
    return macro_news[:8]

def analyze_sentiment(ticker, news_list):
    """Apply Gemini LLM to interpret macro and asset-specific news impact."""
    # Reset Gemini error state
    API_ERROR_STATE["gemini_api_error"] = None

    if not news_list:
        return {"trend": "NEUTRAL", "expectation": 0.0, "reason": "No high-impact news or macro data found.", "relevant": False}

    try:
        processed_items = []
        for n in news_list:
            if not isinstance(n, dict):
                continue
            source = n.get('source') or "Unknown"
            title = n.get('title') or "No Title"
            desc = n.get('description') or ""
            processed_items.append(f"- [{source}] {title}: {desc[:200]}...")
        
        news_text = "\n".join(processed_items)
    except Exception as e:
        print(f"Error processing news list: {e}")
        return {"trend": "NEUTRAL", "expectation": 0.0, "reason": "Error processing news data.", "relevant": False}

    prompt = f"""
Analyze the following news articles for the stock '{ticker}'. 
These include macro-economic (tariffs, taxes, budget) and sector-specific news.

CORE TASK: 
1. Evaluate how global/national macro factors (trade, budget, taxes) and sector trends impact {ticker}.
2. Disregard general market noise. Only use news that has a direct or indirect price-moving logical link to {ticker}.
3. If the news is entirely irrelevant to {ticker} or its sector, set "relevant": false.

Expected JSON Output:
{{
  "trend": "UP" | "DOWN" | "NEUTRAL",
  "expectation": float (e.g. 1.2 for +1.2% impact),
  "reason": "Explain the logical price-moving link between this news and {ticker}.",
  "relevant": true | false
}}

Market Data for {ticker} Context:
{news_text}

JSON Output:
"""

    try:
        client = get_gemini_client()
        if not client:
            API_ERROR_STATE["gemini_api_error"] = "Gemini API Key missing"
            return {"trend": "NEUTRAL", "expectation": 0.0, "reason": "Gemini API key not configured.", "relevant": False}
            
        response = client.models.generate_content(
            model='gemini-2.5-flash', 
            contents=prompt
        )
        result_text = response.text.strip()
        
        # Extract JSON block
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0].strip()
        elif "```" in result_text:
            result_text = result_text.split("```")[1].strip()
        
        data = json.loads(result_text)
        
        # Handle if Gemini returned a list instead of an object
        if isinstance(data, list) and len(data) > 0:
            data = data[0]
            
        if not isinstance(data, dict):
            raise ValueError("Parsed JSON is not a dictionary")

        # Ensure it has the relevant flag
        if "relevant" not in data:
            data["relevant"] = data.get("trend") != "NEUTRAL"
        return data
    except Exception as e:
        error_msg = str(e)
        print(f"Error in Gemini analysis: {error_msg}")
        
        # Check for quota/limit errors
        if "429" in error_msg or "quota" in error_msg.lower() or "exhausted" in error_msg.lower():
            reason_msg = "AI Usage Limit Reached (429)"
            API_ERROR_STATE["gemini_api_error"] = reason_msg
            return {"trend": "NEUTRAL", "expectation": 0.0, "reason": reason_msg, "relevant": False}
        
        reason_msg = f"AI Service Error: {error_msg[:50]}"
        API_ERROR_STATE["gemini_api_error"] = reason_msg
        return {"trend": "NEUTRAL", "expectation": 0.0, "reason": reason_msg, "relevant": False}

if __name__ == "__main__":
    # Test
    ticker = "RELIANCE.NS"
    news = fetch_news(ticker)
    print(f"Fetched {len(news)} news articles.")
    sentiment = analyze_sentiment(ticker, news)
    print(f"Sentiment for {ticker}: {sentiment}")
