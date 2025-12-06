import streamlit as st
import pandas as pd
import yfinance as yf
import requests
from bs4 import BeautifulSoup
import math
import time
import os
import itertools
import json
from datetime import datetime, time as dt_time
import pytz
from decimal import Decimal, ROUND_HALF_UP
import io

# ==========================================
# 0. 頁面設定與初始化
# ==========================================
st.set_page_config(page_title="當沖戰略室", page_icon="⚡", layout="wide")

# [新增] CSS 修復側邊欄圖標錯誤 (強制替換為箭頭)
st.markdown("""
    <style>
    /* 隱藏側邊欄切換按鈕內的所有預設內容 (包含錯誤的文字圖標) */
    [data-testid="stSidebarCollapsedControl"] * {
        display: none !important;
    }
    
    /* 使用偽元素插入一個簡單的箭頭符號 */
    [data-testid="stSidebarCollapsedControl"]::after {
        content: "➤";  /* 這裡可以改成您喜歡的任何箭頭符號，如 ➜, ➡, ▷ */
        font-size: 24px;
        color: #555;  /* 箭頭顏色，可依需求調整 */
        display: block;
        margin-top: 5px;
        margin-left: 5px;
        cursor: pointer;
    }

    /* 調整一下按鈕區域的大小，確保箭頭顯示完整 */
    [data-testid="stSidebarCollapsedControl"] {
        width: 40px !important;
        height: 40px !important;
        align-items: center;
        justify-content: center;
    }

    /* 其他樣式保持不變 */
    .block-container { padding-top: 4.5rem; padding-bottom: 1rem; }
    div[data-testid="stDataFrame"] table, td, th, input, div, span, p {
        font-family: 'Microsoft JhengHei', sans-serif !important;
    }
    [data-testid="stMetricValue"] { font-size: 1.2em; }
    thead tr th:first-child { display:none }
    tbody th { display:none }
    </style>
""", unsafe_allow_html=True)

# 1. 標題
st.title("⚡ 當沖戰略室 ⚡")

CONFIG_FILE = "config.json"
DATA_CACHE_FILE = "data_cache.json"

def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f: return json.load(f)
        except: return {}
    return {}

def save_config(font_size, limit_rows):
    try:
        config = {"font_size": font_size, "limit_rows": limit_rows}
        with open(CONFIG_FILE, "w") as f: json.dump(config, f)
        return True
    except: return False

def save_data_cache(df, ignored_set):
    try:
        df_save = df.fillna("") 
        data_to_save = {
            "stock_data": df_save.to_dict(orient='records'),
            "ignored_stocks": list(ignored_set)
        }
        with open(DATA_CACHE_FILE, "w", encoding='utf-8') as f:
            json.dump(data_to_save, f, ensure_ascii=False, indent=4)
    except: pass

def load_data_cache():
    if os.path.exists(DATA_CACHE_FILE):
        try:
            with open(DATA_CACHE_FILE, "r", encoding='utf-8') as f:
                data = json.load(f)
            df = pd.DataFrame(data.get('stock_data', []))
            ignored = set(data.get('ignored_stocks', []))
            return df, ignored
        except: return pd.DataFrame(), set()
    return pd.DataFrame(), set()

# --- 初始化 Session State ---
if 'stock_data' not in st.session_state:
    cached_df, cached_ignored = load_data_cache()
    st.session_state.stock_data = cached_df
    st.session_state.ignored_stocks = cached_ignored

if 'ignored_stocks' not in st.session_state:
    st.session_state.ignored_stocks = set()

if 'calc_base_price' not in st.session_state:
    st.session_state.calc_base_price = 100.0

if 'calc_view_price' not in st.session_state:
    st.session_state.calc_view_price = 100.0

if 'cloud_url' not in st.session_state:
    st.session_state.cloud_url = ""

saved_config = load_config()

if 'font_size' not in st.session_state:
    st.session_state.font_size = saved_config.get('font_size', 15)

if 'limit_rows' not in st.session_state:
    st.session_state.limit_rows = saved_config.get('limit_rows', 5)

# --- 側邊欄設定 ---
with st.sidebar:
    st.header("⚙️ 設定")
    current_font_size = st.slider("字體大小 (表格)", 12, 72, value=st.session_state.font_size, key='font_size_slider')
    st.session_state.font_size = current_font_size
    
    hide_non_stock = st.checkbox("隱藏非個股 (ETF/權證/債券)", value=True)
    
    st.markdown("---")
    
    current_limit_rows = st.number_input(
        "顯示筆數", 
        min_value=1, 
        value=st.session_state.limit_rows,
        key='limit_rows_input'
    )
    st.session_state.limit_rows = current_limit_rows
    
    if st.button("💾 儲存設定"):
        if save_config(current_font_size, current_limit_rows):
            st.toast("設定已儲存！", icon="✅")
            
    st.markdown("### 資料管理")
    st.write(f"🚫 已忽略 **{len(st.session_state.ignored_stocks)}** 檔")
    
    col_restore, col_clear = st.columns([1, 1])
    with col_restore:
        if st.button("♻️ 復原", use_container_width=True):
            st.session_state.ignored_stocks.clear()
            save_data_cache(st.session_state.stock_data, st.session_state.ignored_stocks)
            st.toast("已重置忽略名單。", icon="🔄")
            st.rerun()
    with col_clear:
        if st.button("🗑️ 清空", type="primary", use_container_width=True):
            st.session_state.stock_data = pd.DataFrame()
            st.session_state.ignored_stocks = set()
            if os.path.exists(DATA_CACHE_FILE):
                os.remove(DATA_CACHE_FILE)
            st.toast("資料已全部清空", icon="🗑️")
            st.rerun()
    
    st.caption("功能說明")
    st.info("🗑️ **如何刪除股票？**\n\n在表格左側勾選「刪除」框，該股票將被隱藏。")

# --- 動態 CSS (表格縮放) ---
font_px = f"{st.session_state.font_size}px"
zoom_level = current_font_size / 14.0
st.markdown(f"""
    <style>
    div[data-testid="stDataFrame"] {{ width: 100%; zoom: {zoom_level}; }}
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 1. 資料庫與網路功能
# ==========================================

@st.cache_data
def load_local_stock_names():
    code_map = {}
    name_map = {}
    if os.path.exists("stock_names.csv"):
        try:
            df = pd.read_csv("stock_names.csv", header=None, names=["code", "
