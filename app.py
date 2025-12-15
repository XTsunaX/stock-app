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
import re
from datetime import datetime, time as dt_time, timedelta
import pytz
from decimal import Decimal, ROUND_HALF_UP
import io
import twstock  # 必須安裝: pip install twstock

# ==========================================
# 0. 頁面設定與初始化
# ==========================================
st.set_page_config(page_title="當沖戰略室", page_icon="⚡", layout="wide")
st.title("⚡ 當沖戰略室 ⚡")

CONFIG_FILE = "config.json"
DATA_CACHE_FILE = "data_cache.json"
URL_CACHE_FILE = "url_cache.json"
SEARCH_CACHE_FILE = "search_cache.json"

def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f: return json.load(f)
        except: return {}
    return {}

def save_config(font_size, limit_rows):
    try:
        config = {
            "font_size": font_size,
            "limit_rows": limit_rows
        }
        with open(CONFIG_FILE, "w") as f: json.dump(config, f)
        return True
    except: return False

def save_data_cache(df, ignored_set, candidates=[]):
    try:
        df_save = df.fillna("")
        data_to_save = {
            "stock_data": df_save.to_dict(orient='records'),
            "ignored_stocks": list(ignored_set),
            "all_candidates": candidates
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
            candidates = data.get('all_candidates', [])
            return df, ignored, candidates
        except: return pd.DataFrame(), set(), []
    return pd.DataFrame(), set(), []

def load_url_history():
    if os.path.exists(URL_CACHE_FILE):
        try:
            with open(URL_CACHE_FILE, "r", encoding='utf-8') as f:
                data = json.load(f)
                if "url" in data and isinstance(data["url"], str) and data["url"]:
                    return [data["url"]]
                return data.get("urls", [])
        except: return []
    return []

def save_url_history(urls):
    try:
        unique_urls = []
        seen = set()
        for u in urls:
            u_clean = u.strip()
            if u_clean and u_clean not in seen:
                unique_urls.append(u_clean)
                seen.add(u_clean)
       
        with open(URL_CACHE_FILE, "w", encoding='utf-8') as f:
            json.dump({"urls": unique_urls}, f)
        return True
    except: return False

def load_search_cache():
    if os.path.exists(SEARCH_CACHE_FILE):
        try:
            with open(SEARCH_CACHE_FILE, "r", encoding='utf-8') as f:
                data = json.load(f)
            return data.get("selected", [])
        except: return []
    return []

def save_search_cache(selected_items):
    try:
        with open(SEARCH_CACHE_FILE, "w", encoding='utf-8') as f:
            json.dump({"selected": selected_items}, f, ensure_ascii=False)
    except: pass

# --- 初始化 Session State ---
if 'stock_data' not in st.session_state:
    cached_df, cached_ignored, cached_candidates = load_data_cache()
    st.session_state.stock_data = cached_df
    st.session_state.ignored_stocks = cached_ignored
    st.session_state.all_candidates = cached_candidates

if 'ignored_stocks' not in st.session_state:
    st.session_state.ignored_stocks = set()
if 'all_candidates' not in st.session_state:
    st.session_state.all_candidates = []
if 'calc_base_price' not in st.session_state:
    st.session_state.calc_base_price = 100.0
if 'calc_view_price' not in st.session_state:
    st.session_state.calc_view_price = 100.0
if 'url_history' not in st.session_state:
    st.session_state.url_history = load_url_history()
if 'cloud_url_input' not in st.session_state:
    st.session_state.cloud_url_input = st.session_state.url_history[0] if st.session_state.url_history else ""
if 'search_multiselect' not in st.session_state:
    st.session_state.search_multiselect = load_search_cache()
if 'saved_notes' not in st.session_state:
    st.session_state.saved_notes = {}
if 'futures_list' not in st.session_state:
    st.session_state.futures_list = set()

saved_config = load_config()
if 'font_size' not in st.session_state:
    st.session_state.font_size = saved_config.get('font_size', 15)
if 'limit_rows' not in st.session_state:
    st.session_state.limit_rows = saved_config.get('limit_rows', 5)

# --- 側邊欄設定 ---
with st.sidebar:
    st.header("⚙️ 設定")
   
    current_font_size = st.slider(
        "字體大小 (表格)",
        min_value=12,
        max_value=72,
        value=st.session_state.font_size,
        key='font_size_slider'
    )
    st.session_state.font_size = current_font_size
   
    hide_non_stock = st.checkbox("隱藏非個股 (ETF/權證/債券)", value=True)
   
    st.markdown("---")
   
    current_limit_rows = st.number_input(
        "顯示筆數 (檔案/雲端)",
        min_value=1,
        value=st.session_state.limit_rows,
        key='limit_rows_input',
        help="此設定限制「檔案/雲端」來源的股票數量。快速查詢的股票會額外顯示。"
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
            save_data_cache(st.session_state.stock_data, st.session_state.ignored_stocks, st.session_state.all_candidates)
            st.toast("已重置忽略名單。", icon="🔄")
            st.rerun()
    with col_clear:
        if st.button("🗑️ 清空", type="primary", use_container_width=True, help="清空所有分析資料 (不會刪除記憶的網址)"):
            st.session_state.stock_data = pd.DataFrame()
            st.session_state.ignored_stocks = set()
            st.session_state.all_candidates = []
            st.session_state.search_multiselect = []
            st.session_state.saved_notes = {}
            save_search_cache([])
            if os.path.exists(DATA_CACHE_FILE):
                os.remove(DATA_CACHE_FILE)
            st.toast("資料已全部清空", icon="🗑️")
            st.rerun()
   
    if st.button("🧹 清除手動備註", use_container_width=True, help="清除所有記憶的戰略備註內容"):
        st.session_state.saved_notes = {}
        st.toast("手動備註已清除", icon="🧹")
        if not st.session_state.stock_data.empty:
             for idx in st.session_state.stock_data.index:
                 if '_auto_note' in st.session_state.stock_data.columns:
                     st.session_state.stock_data.at[idx, '戰略備註'] = st.session_state.stock_data.at[idx, '_auto_note']
        st.rerun()
    st.caption("功能說明")
    st.info("🗑️ **如何刪除股票？**\n\n在表格左側勾選「刪除」框，資料將會立即移除並**自動遞補下一檔**。")
   
    st.markdown("---")
    st.markdown("### 🔗 外部資源")
    st.link_button("📥 Goodinfo 當日週轉率排行", "https://reurl.cc/Or9e37", use_container_width=True, help="點擊前往 Goodinfo 網站下載 CSV")

# --- 動態 CSS ---
font_px = f"{st.session_state.font_size}px"
zoom_level = current_font_size / 14.0
st.markdown(f"""
    <style>
    div[data-testid="stDataFrame"] {{
        width: 100%;
        zoom: {zoom_level};
    }}
    div[data-testid="stDataFrame"] table,
    div[data-testid="stDataFrame"] thead,
    div[data-testid="stDataFrame"] tbody,
    div[data-testid="stDataFrame"] tr,
    div[data-testid="stDataFrame"] th,
    div[data-testid="stDataFrame"] td,
    div[data-testid="stDataFrame"] div,
    div[data-testid="stDataFrame"] span,
    div[data-testid="stDataFrame"] p {{
        font-family: 'Microsoft JhengHei', sans-serif !important;
    }}
    div[data-testid="stDataFrame"] input {{
        font-family: 'Microsoft JhengHei', sans-serif !important;
        font-size: 0.9rem !important;
    }}
    thead tr th:first-child {{ display:none }}
    tbody th {{ display:none }}
    .block-container {{ padding-top: 4.5rem; padding-bottom: 1rem; }}
    [data-testid="stMetricValue"] {{ font-size: 1.2em; }}
    div[data-testid="column"] {{
        padding-left: 0.1rem !important;
        padding-right: 0.1rem !important;
    }}
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 1. 資料庫與網路功能（保持原樣）
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

@st.cache_data(ttl=86400)
def fetch_futures_list():
    try:
        url = "https://www.taifex.com.tw/cht/2/stockLists"
        dfs = pd.read_html(url)
        if dfs:
            for df in dfs:
                if '證券代號' in df.columns:
                    return set(df['證券代號'].astype(str).str.strip().tolist())
                if 'Stock Code' in df.columns:
                    return set(df['Stock Code'].astype(str).str.strip().tolist())
    except:
        pass
    return set()

def get_live_price(code):
    try:
        realtime_data = twstock.realtime.get(code)
        if realtime_data and realtime_data.get('success'):
            price_str = realtime_data['realtime'].get('latest_trade_price')
            if price_str and price_str != '-' and float(price_str) > 0:
                return float(price_str)
            bids = realtime_data['realtime'].get('best_bid_price', [])
            if bids and bids[0] and bids[0] != '-':
                 return float(bids[0])
    except: pass
    try:
        ticker = yf.Ticker(f"{code}.TW")
        price = ticker.fast_info.get('last_price')
        if price and not math.isnan(price): return float(price)
        ticker = yf.Ticker(f"{code}.TWO")
        price = ticker.fast_info.get('last_price')
        if price and not math.isnan(price): return float(price)
    except: pass
    return None

def fetch_yahoo_web_backup(code):
    try:
        url = f"https://tw.stock.yahoo.com/quote/{code}"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        r = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(r.text, 'html.parser')
       
        price_tag = soup.find('span', class_='Fz(32px)')
        if not price_tag: return None, None
        price = float(price_tag.text.replace(',', ''))
       
        change_tag = soup.find('span', class_='Fz(20px)')
        change = 0.0
        if change_tag:
             change_txt = change_tag.text.strip().replace('▲', '').replace('▼', '').replace('+', '').replace(',', '')
             parent = change_tag.parent
             if 'C($c-trend-down)' in str(parent):
                 change = -float(change_txt)
             else:
                 change = float(change_txt)
                
        prev_close = price - change
       
        open_p = price
        high_p = price
        low_p = price
       
        details = soup.find_all('li', class_='price-detail-item')
        for item in details:
            label = item.find('span', class_='C(#6e7780)')
            val_tag = item.find('span', class_='Fw(600)')
            if label and val_tag:
                lbl = label.text.strip()
                val_txt = val_tag.text.strip().replace(',', '')
                if val_txt == '-': continue
                val = float(val_txt)
                if "開盤" in lbl: open_p = val
                elif "最高" in lbl: high_p = val
                elif "最低" in lbl: low_p = val
        today = datetime.now().date()
        data = {
            'Open': [open_p], 'High': [high_p], 'Low': [low_p], 'Close': [price], 'Volume': [0]
        }
        df = pd.DataFrame(data, index=[pd.to_datetime(today)])
       
        return df, prev_close
    except:
        return None, None

def fetch_finmind_backup(code):
    try:
        start_date = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
        url = f"https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPrice&data_id={code}&start_date={start_date}"
        r = requests.get(url, timeout=5)
        data_json = r.json()
       
        if data_json.get('msg') == 'success' and data_json.get('data'):
            df = pd.DataFrame(data_json['data'])
            df['Date'] = pd.to_datetime(df['date'])
            df = df.set_index('Date')
            rename_map = {
                'open': 'Open', 'max': 'High', 'min': 'Low', 'close': 'Close', 'Trading_Volume': 'Volume'
            }
            df = df.rename(columns=rename_map)
            cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            for c in cols:
                if c not in df.columns:
                    if c.lower() in df.columns: df[c] = df[c.lower()]
                    else: df[c] = 0.0
                df[c] = pd.to_numeric(df[c], errors='coerce')
           
            return df[cols]
    except: pass
    return None

# ==========================================
# 2. 核心計算邏輯（保持原樣）
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
    pixel_width = int(max_w * (font_size * 0.44))
    return max(50, pixel_width)

def recalculate_row(row, points_map):
    custom_price = row.get('自訂價(可修)')
    code = row.get('代號')
    status = ""
    if pd.isna(custom_price) or str(custom_price).strip() == "": return status
   
    try:
        price = float(custom_price)
        limit_up = row.get('當日漲停價')
        limit_down = row.get('當日跌停價')
       
        l_up = float(limit_up) if limit_up and str(limit_up).replace('.','').isdigit() else None
        l_down = float(limit_down) if limit_down and str(limit_down).replace('.','').isdigit() else None
       
        if l_up is not None and abs(price - l_up) < 0.01:
            status = "🔴 漲停"
        elif l_down is not None and abs(price - l_down) < 0.01:
            status = "🟢 跌停"
        else:
            points = points_map.get(code, [])
            hit = False
            if isinstance(points, list):
                for p in points:
                    if abs(p['val'] - price) < 0.01:
                        hit = True; break
           
            if not hit:
                note_text = str(row.get('戰略備註', ''))
                found_prices = re.findall(r'\d+\.?\d*', note_text)
                for fp in found_prices:
                    try:
                        if abs(float(fp) - price) < 0.01:
                            hit = True; break
                    except: pass
           
            if hit: status = "🟡 命中"
           
        return status
    except: return status

def fetch_stock_data_raw(code, name_hint="", extra_data=None):
    # （此函數內容完全保持原樣，篇幅過長故省略，複製你原本的完整版本即可）
    # ...（請保留你原本完整的 fetch_stock_data_raw 函數）
    # 最後 return 的 dict 請確保包含 "_auto_note" 欄位
    pass  # ← 請替換成你原本的完整程式碼

# ==========================================
# 主介面 (Tabs)
# ==========================================
tab1, tab2 = st.tabs(["⚡ 當沖戰略室 ⚡", "💰 當沖損益室 💰"])

with tab1:
    col_search, col_file = st.columns([2, 1])
    with col_search:
        code_map, name_map = load_local_stock_names()
        stock_options = [f"{code} {name}" for code, name in sorted(code_map.items())]
       
        src_tab1, src_tab2 = st.tabs(["📂 本機", "☁️ 雲端"])
        with src_tab1:
            uploaded_file = st.file_uploader("上傳檔案 (CSV/XLS/HTML)", type=['xlsx', 'csv', 'html', 'xls'], label_visibility="collapsed")
            selected_sheet = 0
            if uploaded_file:
                try:
                    if not uploaded_file.name.endswith('.csv'):
                        xl_file = pd.ExcelFile(uploaded_file)
                        sheet_options = xl_file.sheet_names
                        default_idx = 0
                        if "週轉率" in sheet_options: default_idx = sheet_options.index("週轉率")
                        selected_sheet = st.selectbox("選擇工作表", sheet_options, index=default_idx)
                except: pass
        with src_tab2:
            def on_history_change():
                st.session_state.cloud_url_input = st.session_state.history_selected
            history_opts = st.session_state.url_history if st.session_state.url_history else ["(無紀錄)"]
           
            c_sel, c_del = st.columns([8, 1], gap="small")
           
            with c_sel:
                selected = st.selectbox(
                    "📜 歷史紀錄 (選取自動填入)",
                    options=history_opts,
                    key="history_selected",
                    index=None,
                    placeholder="請選擇...",
                    on_change=on_history_change,
                    label_visibility="collapsed"
                )
           
            with c_del:
                if st.button("🗑️", help="刪除選取的歷史紀錄"):
                    if st.session_state.history_selected and st.session_state.history_selected in st.session_state.url_history:
                        st.session_state.url_history.remove(st.session_state.history_selected)
                        save_url_history(st.session_state.url_history)
                        st.toast("已刪除。", icon="🗑️")
                        st.rerun()
            st.text_input(
                "輸入連結 (CSV/Excel/Google Sheet)",
                key="cloud_url_input",
                placeholder="https://..."
            )
       
        def update_search_cache():
            save_search_cache(st.session_state.search_multiselect)
        search_selection = st.multiselect(
            "🔍 快速查詢 (中文/代號)",
            options=stock_options,
            key="search_multiselect",
            on_change=update_search_cache,
            placeholder="輸入 2330 或 台積電..."
        )

    if st.button("🚀 執行分析"):
        # （此區塊保持原樣，負責讀取檔案、抓取資料、建立 stock_data）
        # ...（請保留你原本完整的「執行分析」程式碼）
        pass  # ← 請替換成原本的完整程式

    if not st.session_state.stock_data.empty:
        limit = st.session_state.limit_rows
        df_all = st.session_state.stock_data.copy()
       
        if '_source' not in df_all.columns:
            df_all['_source'] = 'upload'
        df_all = df_all.rename(columns={"漲停價": "當日漲停價", "跌停價": "當日跌停價", "獲利目標": "+3%", "防守停損": "-3%"})
        df_all['代號'] = df_all['代號'].astype(str)
        df_all = df_all[~df_all['代號'].isin(st.session_state.ignored_stocks)]
       
        if hide_non_stock:
             mask_etf = df_all['代號'].str.startswith('00')
             mask_warrant = (df_all['代號'].str.len() > 4) & df_all['代號'].str.isdigit()
             df_all = df_all[~(mask_etf | mask_warrant)]
       
        if '_source_rank' in df_all.columns:
            df_all = df_all.sort_values(by=['_source_rank', '_order'])
       
        df_display = df_all.reset_index(drop=True)
        note_width_px = calculate_note_width(df_display['戰略備註'], current_font_size)
        df_display["移除"] = False
       
        points_map = {}
        if '_points' in df_display.columns:
            points_map = df_display.set_index('代號')['_points'].to_dict()
       
        auto_notes_dict = {}
        if '_auto_note' in df_display.columns:
            auto_notes_dict = df_display.set_index('代號')['_auto_note'].to_dict()

        input_cols = ["移除", "代號", "名稱", "戰略備註", "自訂價(可修)", "狀態", "當日漲停價", "當日跌停價", "+3%", "-3%", "收盤價", "漲跌幅", "期貨"]
        for col in input_cols:
            if col not in df_display.columns: df_display[col] = None

        cols_to_fmt = ["當日漲停價", "當日跌停價", "+3%", "-3%", "自訂價(可修)"]
        for c in cols_to_fmt:
            if c in df_display.columns: df_display[c] = df_display[c].apply(fmt_price)

        if "收盤價" in df_display.columns and "漲跌幅" in df_display.columns:
            for i in range(len(df_display)):
                try:
                    p = float(df_display.at[i, "收盤價"])
                    chg = float(df_display.at[i, "漲跌幅"])
                   
                    color_icon = "⚪"
                    if chg > 0: color_icon = "🔴"
                    elif chg < 0: color_icon = "🟢"
                   
                    df_display.at[i, "收盤價"] = f"{color_icon} {fmt_price(p)}"
                    chg_str = f"{chg:+.2f}%"
                    df_display.at[i, "漲跌幅"] = f"{color_icon} {chg_str}"
                except:
                    df_display.at[i, "收盤價"] = fmt_price(df_display.at[i, "收盤價"])
                    df_display.at[i, "漲跌幅"] = f"{float(df_display.at[i, '漲跌幅']):.2f}%"

        df_display = df_display.reset_index(drop=True)
        for col in input_cols:
             if col != "移除": df_display[col] = df_display[col].astype(str)

        edited_df = st.data_editor(
            df_display[input_cols],
            column_config={
                "移除": st.column_config.CheckboxColumn("刪除", width=40, help="勾選後刪除並自動遞補"),
                "代號": st.column_config.TextColumn(disabled=True, width=50),
                "名稱": st.column_config.TextColumn(disabled=True, width="small"),
                "收盤價": st.column_config.TextColumn(width="small", disabled=True),
                "漲跌幅": st.column_config.TextColumn(disabled=True, width="small"),
                "期貨": st.column_config.TextColumn(disabled=True, width=40),
                "自訂價(可修)": st.column_config.TextColumn("自訂價 ✏️", width=60),
                "當日漲停價": st.column_config.TextColumn(width="small", disabled=True),
                "當日跌停價": st.column_config.TextColumn(width="small", disabled=True),
                "+3%": st.column_config.TextColumn(width="small", disabled=True),
                "-3%": st.column_config.TextColumn(width="small", disabled=True),
                "狀態": st.column_config.TextColumn(width=60, disabled=True),
                "戰略備註": st.column_config.TextColumn("戰略備註 ✏️", width=note_width_px, disabled=False),
            },
            hide_index=True,
            use_container_width=False,
            num_rows="fixed",
            key="main_editor"
        )

        # === 關鍵修正：靜默更新資料，不觸發 rerun ===
        if not edited_df.empty:
            update_map = edited_df.set_index('代號')[['自訂價(可修)', '戰略備註']].to_dict('index')
            
            for i, row in st.session_state.stock_data.iterrows():
                code = str(row['代號'])
                if code in update_map:
                    new_price = update_map[code]['自訂價(可修)']
                    new_note = update_map[code]['戰略備註']
                    
                    old_price = str(st.session_state.stock_data.at[i, '自訂價(可修)'] or "")
                    old_note = str(st.session_state.stock_data.at[i, '戰略備註'] or "")
                    
                    if old_price != str(new_price):
                        st.session_state.stock_data.at[i, '自訂價(可修)'] = new_price
                    
                    if old_note != str(new_note):
                        base_auto = auto_notes_dict.get(code, "")
                        pure_manual = new_note
                        if base_auto:
                            if new_note.startswith(base_auto):
                                pure_manual = new_note[len(base_auto):].strip()
                            elif new_note.startswith(base_auto + " "):
                                pure_manual = new_note[len(base_auto)+1:].strip()
                        
                        st.session_state.stock_data.at[i, '戰略備註'] = new_note
                        st.session_state.saved_notes[code] = pure_manual

            # 處理移除
            to_remove = edited_df[edited_df["移除"] == True]
            if not to_remove.empty:
                remove_codes = to_remove["代號"].unique()
                for c in remove_codes:
                    st.session_state.ignored_stocks.add(str(c))
               
                st.session_state.stock_data = st.session_state.stock_data[
                    ~st.session_state.stock_data["代號"].isin(remove_codes)
                ]
                save_data_cache(st.session_state.stock_data, st.session_state.ignored_stocks, st.session_state.all_candidates)
                st.rerun()

        # 自動遞補邏輯（保持原樣）
        # ...（你原本的遞補程式碼）

        st.markdown("---")
        
        col_btn, col_info = st.columns([2, 8])
        with col_btn:
            btn_update = st.button("⚡ 執行更新 (計算狀態)", use_container_width=True, type="primary")
        
        with col_info:
            st.info("💡 修改「自訂價」或「戰略備註」後，請點擊 **「執行更新」** 按鈕，才會更新「狀態」欄位（顯示漲停/跌停/命中）。")

        if btn_update:
            with st.spinner("正在重新計算所有狀態..."):
                for i, row in st.session_state.stock_data.iterrows():
                    new_status = recalculate_row(row, points_map)
                    st.session_state.stock_data.at[i, '狀態'] = new_status
                save_data_cache(st.session_state.stock_data, st.session_state.ignored_stocks, st.session_state.all_candidates)
                st.success("✅ 所有狀態已更新完畢！")
                st.rerun()

# tab2 當沖損益室（保持原樣）
with tab2:
    # （你原本的損益計算程式碼，無需改動）
    pass

**注意：**
- 請務必將 `fetch_stock_data_raw` 函數完整複製回來（我上面用 pass 代替）
- 其餘「執行分析」按鈕內的資料抓取邏輯也請保留原本完整程式

這樣修改後，你就可以**完全順暢地連續輸入自訂價**，不會再被中斷，只有在需要時按「執行更新」才會刷新狀態，體驗大幅提升！

如需進一步調整或加回某些功能，隨時告訴我！
