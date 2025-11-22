import streamlit as st
import pandas as pd
import yfinance as yf
import requests
from bs4 import BeautifulSoup
import math
import time
import os
import itertools

# ==========================================
# 0. 頁面設定與初始化
# ==========================================
st.set_page_config(page_title="當沖戰略室 V8 (網路版)", page_icon="⚡", layout="wide")

# --- 🔥 修復錯誤關鍵：在此處初始化 Session State ---
if 'stock_data' not in st.session_state:
    st.session_state.stock_data = pd.DataFrame()

# --- 側邊欄設定 ---
with st.sidebar:
    st.header("⚙️ 設定")
    
    # 字體大小調整
    font_size = st.slider("字體大小 (表格)", min_value=12, max_value=24, value=16)
    
    hide_etf = st.checkbox("隱藏 ETF (00開頭)", value=True)
    st.markdown("---")
    limit_rows = st.number_input("顯示筆數", min_value=1, value=50)
    st.caption("功能說明")
    st.info("🗑️ **如何刪除股票？**\n\n勾選左側框框後按 `Delete` 鍵。")

# --- 動態 CSS ---
st.markdown(f"""
    <style>
    .block-container {{ padding-top: 0.5rem; padding-bottom: 1rem; }}
    
    /* 調整表格字體大小 */
    div[data-testid="stDataFrame"] * {{ 
        font-size: {font_size}px !important; 
        font-family: 'Microsoft JhengHei', sans-serif !important;
    }}
    
    /* 命中標籤樣式 */
    .hit-tag {{ background-color: #ffff00; color: black; padding: 2px 6px; border-radius: 4px; font-weight: bold; }}
    
    /* 修正輸入跳動問題: 強制表格容器穩定 */
    div[data-testid="stDataFrame"] {{
        min-height: 200px;
    }}
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 1. 資料庫與網路功能
# ==========================================

@st.cache_data
def load_local_stock_names():
    """讀取本地 stock_names.csv"""
    code_map = {}
    name_map = {}
    if os.path.exists("stock_names.csv"):
        try:
            df = pd.read_csv("stock_names.csv", header=None, names=["code", "name"], dtype=str)
            for _, row in df.iterrows():
                c = str(row['code']).strip()
                n = str(row['name']).strip()
                code_map[c] = n
                name_map[n] = c
        except Exception as e:
            pass
    return code_map, name_map

@st.cache_data(ttl=86400)
def get_stock_name_online(code):
    code = str(code).strip()
    if not code.isdigit(): return code
    code_map, _ = load_local_stock_names()
    if code in code_map: return code_map[code]
    
    try:
        url = f"https://tw.stock.yahoo.com/quote/{code}.TW"
        headers = {'User-Agent': 'Mozilla/5.0'}
        r = requests.get(url, headers=headers, timeout=2)
        soup = BeautifulSoup(r.text, "html.parser")
        if soup.title and "(" in soup.title.string:
            return soup.title.string.split('(')[0].strip()
        
        url_two = f"https://tw.stock.yahoo.com/quote/{code}.TWO"
        r_two = requests.get(url_two, headers=headers, timeout=2)
        soup_two = BeautifulSoup(r_two.text, "html.parser")
        if soup_two.title and "(" in soup_two.title.string:
            return soup_two.title.string.split('(')[0].strip()
        return code
    except:
        return code

@st.cache_data(ttl=86400)
def search_code_online(query):
    query = query.strip()
    if query.isdigit(): return query
    _, name_map = load_local_stock_names()
    if query in name_map: return name_map[query]
    
    try:
        url = f"https://tw.stock.yahoo.com/h/kimosearch/search_list.html?keyword={query}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        r = requests.get(url, headers=headers, timeout=2)
        soup = BeautifulSoup(r.text, "html.parser")
        links = soup.find_all('a', href=True)
        for link in links:
            if "/quote/" in link['href'] and ".TW" in link['href']:
                parts = link['href'].split("/quote/")[1].split(".")
                if parts[0].isdigit(): return parts[0]
    except:
        pass
    return None

# ==========================================
# 2. 核心計算邏輯
# ==========================================

def get_tick_size(price):
    if price < 10: return 0.01
    if price < 50: return 0.05
    if price < 100: return 0.1
    if price < 500: return 0.5
    if price < 1000: return 1.0
    return 5.0

def calculate_limits(price):
    try:
        p = float(price)
        tick = get_tick_size(p)
        limit_up = math.floor((p * 1.10) / tick) * tick
        limit_down = math.ceil((p
