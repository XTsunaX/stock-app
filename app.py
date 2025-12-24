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
st.set_page_config(page_title="當沖戰略室", page_icon="⚡", layout="wide", initial_sidebar_state="collapsed")

# 1. 標題
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

def save_config(font_size, limit_rows, auto_update, delay_sec):
    try:
        config = {
            "font_size": font_size, 
            "limit_rows": limit_rows,
            "auto_update": auto_update,
            "delay_sec": delay_sec
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

if 'auto_update_last_row' not in st.session_state:
    st.session_state.auto_update_last_row = saved_config.get('auto_update', True)

if 'update_delay_sec' not in st.session_state:
    st.session_state.update_delay_sec = saved_config.get('delay_sec', 1.0) 

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
    code_map, _ = load_local_stock_names()
    if code in code_map: return code_map[code]
    return code

# --- 側邊欄設定 ---
with st.sidebar:
    st.header("⚙️ 設定")
    
    current_font_size = st.slider("字體大小 (表格)", min_value=12, max_value=72, value=st.session_state.font_size, key='font_size_slider')
    st.session_state.font_size = current_font_size
    
    hide_non_stock = st.checkbox("隱藏非個股 (ETF/權證/債券)", value=True)
    
    # [需求 3] 新增近3日高低點選項
    show_3d_hilo = st.checkbox("近3日高低點 (戰略備註)", value=False, help="勾選後將在備註中加入 3H(近3日最高) 與 3L(近3日最低)")
    
    st.markdown("---")
    
    current_limit_rows = st.number_input("顯示筆數 (檔案/雲端)", min_value=1, value=st.session_state.limit_rows, key='limit_rows_input')
    st.session_state.limit_rows = current_limit_rows
    
    if st.button("💾 儲存設定"):
        if save_config(current_font_size, current_limit_rows, st.session_state.auto_update_last_row, st.session_state.update_delay_sec):
            st.toast("設定已儲存！", icon="✅")
            
    st.markdown("### 資料管理")
    # [需求 1] 忽略名單改為可視狀態，方便加回
    if st.session_state.ignored_stocks:
        st.write("🚫 忽略名單 (取消勾選以復原):")
        ignored_list = sorted(list(st.session_state.ignored_stocks))
        # 呈現格式為 代號+股名
        options_display = [f"{c} {get_stock_name_online(c)}" for c in ignored_list]
        options_map = {f"{c} {get_stock_name_online(c)}": c for c in ignored_list}
        
        selected_remains = st.multiselect("管理忽略股票", options=options_display, default=options_display, label_visibility="collapsed")
        
        current_codes = set(options_map[item] for item in selected_remains)
        if current_codes != st.session_state.ignored_stocks:
            st.session_state.ignored_stocks = current_codes
            save_data_cache(st.session_state.stock_data, st.session_state.ignored_stocks, st.session_state.all_candidates)
            st.rerun()
    else:
        st.write("🚫 目前無忽略股票")

    col_restore, col_clear = st.columns([1, 1])
    with col_restore:
        if st.button("♻️ 全部復原", use_container_width=True):
            st.session_state.ignored_stocks.clear()
            save_data_cache(st.session_state.stock_data, st.session_state.ignored_stocks, st.session_state.all_candidates)
            st.rerun()
    with col_clear:
        if st.button("🗑️ 全部清空", type="primary", use_container_width=True):
            st.session_state.stock_data = pd.DataFrame()
            st.session_state.ignored_stocks = set()
            st.session_state.all_candidates = []
            st.session_state.saved_notes = {}
            if os.path.exists(DATA_CACHE_FILE): os.remove(DATA_CACHE_FILE)
            st.rerun()

    st.markdown("---")
    st.markdown("### 🔗 外部資源")
    st.link_button("📥 Goodinfo 當日週轉率排行", "https://reurl.cc/Or9e37", use_container_width=True)

# --- 動態 CSS ---
zoom_level = current_font_size / 14.0
st.markdown(f"""
    <style>
    div[data-testid="stDataFrame"] {{ width: 100%; zoom: {zoom_level}; }}
    div[data-testid="stDataFrame"] * {{ font-family: 'Microsoft JhengHei', sans-serif !important; }}
    thead tr th:first-child {{ display:none }}
    tbody th {{ display:none }}
    .block-container {{ padding-top: 4.5rem; padding-bottom: 1rem; }}
    div[data-testid="column"] {{ padding-left: 0.1rem !important; padding-right: 0.1rem !important; }}
    </style>
""", unsafe_allow_html=True)

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
    pixel_width = int(max_w * (font_size * 0.44))
    return max(150, pixel_width)

def recalculate_row(row, points_map):
    custom_price = row.get('自訂價(可修)')
    code = row.get('代號')
    status = ""
    if pd.isna(custom_price) or str(custom_price).strip() == "": return status
    try:
        price = float(custom_price)
        l_up = float(row.get('當日漲停價', 0))
        l_down = float(row.get('當日跌停價', 0))
        strat_values = [p['val'] for p in points_map.get(code, [])]
        note_text = str(row.get('戰略備註', ''))
        found_prices = re.findall(r'\d+\.?\d*', note_text)
        for fp in found_prices:
            try: strat_values.append(float(fp))
            except: pass
        if abs(price - l_up) < 0.01: status = "🔴 漲停"
        elif abs(price - l_down) < 0.01: status = "🟢 跌停"
        elif strat_values:
            max_val, min_val = max(strat_values), min(strat_values)
            if price > max_val: status = "🔴 強"
            elif price < min_val: status = "🟢 弱"
            else:
                hit = any(abs(v - price) < 0.01 for v in strat_values)
                if hit: status = "🟡 命中"
        return status
    except: return status

@st.cache_data(ttl=86400)
def fetch_futures_list():
    try:
        url = "https://www.taifex.com.tw/cht/2/stockLists"
        dfs = pd.read_html(url)
        for df in dfs:
            if '證券代號' in df.columns: return set(df['證券代號'].astype(str).str.strip().tolist())
    except: pass
    return set()

def fetch_stock_data_raw(code, name_hint="", extra_data=None):
    code = str(code).strip()
    hist = pd.DataFrame()
    try:
        ticker = yf.Ticker(f"{code}.TW")
        hist = ticker.history(period="10d")
        if hist.empty:
            ticker = yf.Ticker(f"{code}.TWO")
            hist = ticker.history(period="10d")
    except: pass
    if hist.empty: return None

    hist['High'] = hist[['High', 'Close']].max(axis=1)
    hist['Low'] = hist[['Low', 'Close']].min(axis=1)
    
    # 判斷交易時間 (以台北時間 13:30 前後切換基準日)
    tz = pytz.timezone('Asia/Taipei')
    now = datetime.now(tz)
    is_during_trading = (now.time() < dt_time(13, 30))
    hist_strat = hist.iloc[:-1] if is_during_trading and (hist.index[-1].date() == now.date()) else hist
    
    strategy_base_price = hist_strat.iloc[-1]['Close']
    prev_close = hist_strat.iloc[-2]['Close'] if len(hist_strat) >= 2 else strategy_base_price
    pct_change = ((strategy_base_price - prev_close) / prev_close) * 100
    
    limit_up, limit_down = calculate_limits(strategy_base_price)
    target_price = apply_sr_rules(strategy_base_price * 1.03, strategy_base_price)
    stop_price = apply_sr_rules(strategy_base_price * 0.97, strategy_base_price)
    
    # 點位計算
    points = []
    # MA5
    if len(hist_strat) >= 5:
        ma5_raw = float(hist_strat['Close'].tail(5).mean())
        ma5 = apply_sr_rules(ma5_raw, strategy_base_price)
        points.append({"val": ma5, "tag": "多" if ma5_raw < strategy_base_price else "空", "force": True})
    # 高低點
    h90 = apply_tick_rules(hist_strat['High'].max()); l90 = apply_tick_rules(hist_strat['Low'].min())
    points.append({"val": h90, "tag": "高"}); points.append({"val": l90, "tag": "低"})
    
    # 格式化系統備註
    points.sort(key=lambda x: x['val'])
    note_parts = []
    for p in points:
        v_str = fmt_price(p['val'])
        tag = p['tag']
        note_parts.append(f"{tag}{v_str}" if tag in ["高", "低"] else f"{v_str}{tag}")
    auto_note = "-".join(note_parts)
    
    # [需求 3] 直接將近3日高低點加到戰略備註邏輯裡
    h3, l3 = 0.0, 0.0
    if len(hist_strat) >= 3:
        h3, l3 = hist_strat['High'].tail(3).max(), hist_strat['Low'].tail(3).min()
    
    manual_note = st.session_state.saved_notes.get(code, "")
    final_name = f"{('🔴' if '多' in auto_note else '🟢' if '空' in auto_note else '⚪')} {name_hint or get_stock_name_online(code)}"
    
    return {
        "代號": code, "名稱": final_name, "收盤價": round(strategy_base_price, 2),
        "漲跌幅": pct_change, "期貨": "✅" if code in st.session_state.futures_list else "",
        "當日漲停價": limit_up, "當日跌停價": limit_down, "自訂價(可修)": None,
        "+3%": target_price, "-3%": stop_price, "戰略備註": auto_note, "狀態": "",
        "_points": points, "_auto_note": auto_note, "_3d_h": h3, "_3d_l": l3
    }

# ==========================================
# 主介面
# ==========================================
tab1, tab2 = st.tabs(["⚡ 當沖戰略室 ⚡", "💰 當沖損益室 💰"])

with tab1:
    code_map, _ = load_local_stock_names()
    stock_options = [f"{c} {n}" for c, n in sorted(code_map.items())]
    
    c_src, c_quick = st.columns([1, 2])
    with c_src:
        uploaded_file = st.file_uploader("上傳 CSV/Excel", type=['xlsx', 'csv'], label_visibility="collapsed")
    with c_quick:
        search_selection = st.multiselect("🔍 快速查詢", options=stock_options, key="search_multiselect", placeholder="輸入代號或名稱...")

    # [需求 2] 調整按鈕寬度符合文字長度
    c_btn1, c_btn2, c_btn3, _ = st.columns([0.15, 0.1, 0.15, 0.6])
    with c_btn1: btn_run = st.button("🚀 執行分析")
    with c_btn2: btn_save = st.button("💾 儲存")
    with c_btn3: btn_clear_note = st.button("🧹 清除手動備註")

    if btn_clear_note:
        st.session_state.saved_notes = {}; st.rerun()
    if btn_save:
        save_data_cache(st.session_state.stock_data, st.session_state.ignored_stocks, st.session_state.all_candidates)
        st.toast("已儲存！")

    if btn_run:
        if not st.session_state.futures_list: st.session_state.futures_list = fetch_futures_list()
        # 執行分析邏輯 (省略重複上傳讀取部分，直接進行 fetch)
        targets = []
        if search_selection:
            for item in search_selection: targets.append((item.split(' ')[0], item.split(' ')[1], 'search', 0))
        
        results = []
        bar = st.progress(0)
        for i, (code, name, src, order) in enumerate(targets):
            data = fetch_stock_data_raw(code, name)
            if data: results.append(data)
            bar.progress((i+1)/len(targets))
        st.session_state.stock_data = pd.DataFrame(results)
        st.rerun()

    if not st.session_state.stock_data.empty:
        df_disp = st.session_state.stock_data.copy()
        
        # [需求 3] 即時處理戰略備註顯示 (3H/3L 直接併入系統備註)
        for i, row in df_disp.iterrows():
            code = row['代號']
            base = row.get('_auto_note', '')
            
            if show_3d_hilo:
                h3, l3 = row.get('_3d_h', 0), row.get('_3d_l', 0)
                h3_str = f"3H{fmt_price(h3)}" if h3 > 0 else ""
                l3_str = f"3L{fmt_price(l3)}" if l3 > 0 else ""
                extra = "-".join(filter(None, [h3_str, l3_str]))
                if extra: base = f"{base}-{extra}" if base else extra
            
            manual = st.session_state.saved_notes.get(code, "")
            df_disp.at[i, "戰略備註"] = f"{base} {manual}".strip()

        # 格式化顯示
        df_disp["移除"] = False
        cols = ["移除", "代號", "名稱", "戰略備註", "自訂價(可修)", "狀態", "當日漲停價", "當日跌停價", "+3%", "-3%", "收盤價", "漲跌幅", "期貨"]
        
        edited_df = st.data_editor(
            df_disp[cols],
            column_config={
                "移除": st.column_config.CheckboxColumn("刪除", width=40),
                "戰略備註": st.column_config.TextColumn("戰略備註 ✏️", width=calculate_note_width(df_disp["戰略備註"], st.session_state.font_size)),
                "自訂價(可修)": st.column_config.TextColumn("自訂價 ✏️", width=80),
            },
            hide_index=True, key="main_editor"
        )
        
        # 更新與儲存邏輯 (略)
