import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import math
import time
import os
import itertools
import json
from datetime import datetime, timedelta, time as dt_time
import pytz
from decimal import Decimal, ROUND_HALF_UP

# ==========================================
# 0. 頁面設定與套件檢查
# ==========================================
st.set_page_config(page_title="當沖戰略室", page_icon="⚡", layout="wide")

# [除錯] 嘗試匯入並顯示詳細錯誤
try:
    from FinMind.data import DataLoader
    import twstock
except ImportError as e:
    st.error(f"⚠️ 套件載入失敗，錯誤訊息：{e}")
    st.warning("請確認 requirements.txt 檔案中已包含 FinMind 與 twstock，並嘗試在 Streamlit Cloud 後台重啟 (Reboot) 應用程式。")
    st.stop()

# 1. 標題
st.title("⚡ 當沖戰略室 ⚡")

CONFIG_FILE = "config.json"
DATA_CACHE_FILE = "data_cache.json"

def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_config(font_size, limit_rows):
    try:
        config = {"font_size": font_size, "limit_rows": limit_rows}
        with open(CONFIG_FILE, "w") as f:
            json.dump(config, f)
        return True
    except:
        return False

def save_data_cache(df, ignored_set):
    try:
        df_save = df.fillna("") 
        data_to_save = {
            "stock_data": df_save.to_dict(orient='records'),
            "ignored_stocks": list(ignored_set)
        }
        with open(DATA_CACHE_FILE, "w", encoding='utf-8') as f:
            json.dump(data_to_save, f, ensure_ascii=False, indent=4)
    except Exception as e:
        pass

def load_data_cache():
    if os.path.exists(DATA_CACHE_FILE):
        try:
            with open(DATA_CACHE_FILE, "r", encoding='utf-8') as f:
                data = json.load(f)
            df = pd.DataFrame(data.get('stock_data', []))
            ignored = set(data.get('ignored_stocks', []))
            return df, ignored
        except Exception as e:
            return pd.DataFrame(), set()
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
    st.session_state.font_size = saved_config.get('font_size', 18)

if 'limit_rows' not in st.session_state:
    st.session_state.limit_rows = saved_config.get('limit_rows', 5)

# --- 側邊欄設定 ---
with st.sidebar:
    st.header("⚙️ 設定")
    
    current_font_size = st.slider("字體大小 (表格)", 12, 72, key='font_size')
    hide_non_stock = st.checkbox("隱藏非個股 (ETF/權證/債券)", value=True)
    st.markdown("---")
    current_limit_rows = st.number_input("顯示筆數", min_value=1, key='limit_rows')
    
    if st.button("💾 儲存設定"):
        if save_config(current_font_size, current_limit_rows):
            st.toast("設定已儲存！", icon="✅")
            
    st.markdown("### 資料管理")
    st.write(f"🚫 已忽略 **{len(st.session_state.ignored_stocks)}** 檔")
    
    if st.button("♻️ 復原忽略", use_container_width=True):
        st.session_state.ignored_stocks.clear()
        save_data_cache(st.session_state.stock_data, st.session_state.ignored_stocks)
        st.toast("已重置忽略名單。", icon="🔄")
        st.rerun()
            
    if st.button("🗑️ 清空資料", type="primary", use_container_width=True):
        st.session_state.stock_data = pd.DataFrame()
        st.session_state.ignored_stocks = set()
        if os.path.exists(DATA_CACHE_FILE):
            os.remove(DATA_CACHE_FILE)
        st.toast("資料已全部清空", icon="🗑️")
        st.rerun()
    
    st.caption("功能說明")
    st.info("🗑️ **如何刪除股票？**\n\n在表格左側勾選「移除」框，該股票將被隱藏。")

# --- 動態 CSS ---
font_px = f"{st.session_state.font_size}px"
zoom_level = current_font_size / 14.0

st.markdown(f"""
    <style>
    .block-container {{ padding-top: 4.5rem; padding-bottom: 1rem; }}
    div[data-testid="stDataFrame"] {{ width: 100%; zoom: {zoom_level}; }}
    div[data-testid="stDataFrame"] table, td, th, input, div, span, p {{
        font-family: 'Microsoft JhengHei', sans-serif !important;
    }}
    [data-testid="stMetricValue"] {{ font-size: 1.2em; }}
    thead tr th:first-child {{ display:none }}
    tbody th {{ display:none }}
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
            df = pd.read_csv("stock_names.csv", header=None, names=["code", "name"], dtype=str)
            for _, row in df.iterrows():
                c = str(row['code']).strip()
                n = str(row['name']).strip()
                code_map[c] = n
                name_map[n] = c
        except: pass
    return code_map, name_map

@st.cache_data(ttl=86400)
def get_stock_name_online(code):
    code = str(code).strip()
    if code in twstock.codes:
        return twstock.codes[code].name
    code_map, _ = load_local_stock_names()
    if code in code_map: return code_map[code]
    return code

@st.cache_data(ttl=86400)
def search_code_online(query):
    query = query.strip()
    if query.isdigit(): return query
    _, name_map = load_local_stock_names()
    if query in name_map: return name_map[query]
    return None

# ==========================================
# 2. 核心計算邏輯
# ==========================================

def get_tick_size(price):
    try: price = float(price)
    except: return 0.01
    if pd.isna(price) or price <= 0: return 0.01
    if price < 10: return 0.01
    if price < 50: return 0.05
    if price < 100: return 0.1
    if price < 500: return 0.5
    if price < 1000: return 1.0
    return 5.0

def calculate_limits(price):
    try:
        p = float(price)
        if math.isnan(p) or p <= 0: return 0, 0
        raw_up = p * 1.10
        tick_up = get_tick_size(raw_up) 
        limit_up = math.floor(raw_up / tick_up) * tick_up
        raw_down = p * 0.90
        tick_down = get_tick_size(raw_down) 
        limit_down = math.ceil(raw_down / tick_down) * tick_down
        return float(f"{limit_up:.2f}"), float(f"{limit_down:.2f}")
    except: return 0, 0

def apply_tick_rules(price):
    try:
        p = float(price)
        if math.isnan(p): return 0.0
        tick = get_tick_size(p)
        rounded = (Decimal(str(p)) / Decimal(str(tick))).quantize(Decimal("1"), rounding=ROUND_HALF_UP) * Decimal(str(tick))
        return float(rounded)
    except: return price

def move_tick(price, steps):
    try:
        curr = float(price)
        if steps > 0:
            for _ in range(steps):
                tick = get_tick_size(curr)
                curr = round(curr + tick, 2)
        elif steps < 0:
            for _ in range(abs(steps)):
                tick = get_tick_size(curr - 0.0001)
                curr = round(curr - tick, 2)
        return curr
    except: return price

def apply_sr_rules(price, base_price):
    try:
        p = float(price)
        if math.isnan(p): return 0.0
        tick = get_tick_size(p)
        d_val = Decimal(str(p))
        d_tick = Decimal(str(tick))
        if p < base_price: return float(math.ceil(d_val / d_tick) * d_tick)
        elif p > base_price: return float(math.floor(d_val / d_tick) * d_tick)
        else: return apply_tick_rules(p)
    except: return price

def fmt_price(v):
    try:
        if pd.isna(v) or v == "": return ""
        return f"{float(v):.2f}".rstrip('0').rstrip('.')
    except: return str(v)

def calculate_note_width(series, font_size):
    def get_width(s):
        w = 0
        for c in str(s): w += 2.0 if ord(c) > 127 else 1.0
        return w
    if series.empty: return 50
    max_w = series.apply(get_width).max()
    if pd.isna(max_w): max_w = 0
    # 係數維持 0.44
    pixel_width = int(max_w * (font_size * 0.44))
    return max(50, pixel_width)

def recalculate_row(row):
    custom_price = row.get('自訂價(可
