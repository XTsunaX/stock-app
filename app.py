import streamlit as st
import pandas as pd
import yfinance as yf
import requests
import math
import time
import os
import itertools
import json
import re
from datetime import datetime, time as dt_time, timedelta
import pytz
from decimal import Decimal, ROUND_HALF_UP
import twstock

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
        config = {"font_size": font_size, "limit_rows": limit_rows}
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

if 'ignored_stocks' not in st.session_state: st.session_state.ignored_stocks = set()
if 'all_candidates' not in st.session_state: st.session_state.all_candidates = []
if 'calc_base_price' not in st.session_state: st.session_state.calc_base_price = 100.0
if 'calc_view_price' not in st.session_state: st.session_state.calc_view_price = 100.0
if 'url_history' not in st.session_state: st.session_state.url_history = load_url_history()
if 'cloud_url_input' not in st.session_state:
    st.session_state.cloud_url_input = st.session_state.url_history[0] if st.session_state.url_history else ""
if 'search_multiselect' not in st.session_state: st.session_state.search_multiselect = load_search_cache()
if 'saved_notes' not in st.session_state: st.session_state.saved_notes = {}
if 'futures_list' not in st.session_state: st.session_state.futures_list = set()

saved_config = load_config()
if 'font_size' not in st.session_state: st.session_state.font_size = saved_config.get('font_size', 15)
if 'limit_rows' not in st.session_state: st.session_state.limit_rows = saved_config.get('limit_rows', 5)
if 'auto_update_last_row' not in st.session_state: st.session_state.auto_update_last_row = True # 預設開啟

# --- 側邊欄設定 ---
with st.sidebar:
    st.header("⚙️ 設定")
    current_font_size = st.slider("字體大小 (表格)", 12, 72, st.session_state.font_size, key='font_size_slider')
    st.session_state.font_size = current_font_size
    hide_non_stock = st.checkbox("隱藏非個股 (ETF/權證/債券)", value=True)
    st.markdown("---")
    current_limit_rows = st.number_input("顯示筆數 (檔案/雲端)", min_value=1, value=st.session_state.limit_rows, key='limit_rows_input')
    st.session_state.limit_rows = current_limit_rows
    
    # [Restored] 恢復自動更新開關，但移除緩衝秒數 (改為內建邏輯)
    auto_update = st.checkbox("啟用最後一列自動更新", value=st.session_state.auto_update_last_row)
    st.session_state.auto_update_last_row = auto_update

    if st.button("💾 儲存設定"):
        if save_config(current_font_size, current_limit_rows): st.toast("設定已儲存！", icon="✅")
            
    st.markdown("### 資料管理")
    st.write(f"🚫 已忽略 **{len(st.session_state.ignored_stocks)}** 檔")
    col_restore, col_clear = st.columns([1, 1])
    with col_restore:
        if st.button("♻️ 復原", use_container_width=True):
            st.session_state.ignored_stocks.clear()
            save_data_cache(st.session_state.stock_data, st.session_state.ignored_stocks, st.session_state.all_candidates)
            st.rerun()
    with col_clear:
        if st.button("🗑️ 清空", type="primary", use_container_width=True):
            st.session_state.stock_data = pd.DataFrame()
            st.session_state.ignored_stocks = set()
            st.session_state.all_candidates = []
            st.session_state.search_multiselect = []
            st.session_state.saved_notes = {} 
            save_search_cache([])
            if os.path.exists(DATA_CACHE_FILE): os.remove(DATA_CACHE_FILE)
            st.rerun()
    
    if st.button("🧹 清除手動備註", use_container_width=True):
        st.session_state.saved_notes = {}
        if not st.session_state.stock_data.empty:
             for idx in st.session_state.stock_data.index:
                 if '_auto_note' in st.session_state.stock_data.columns:
                     st.session_state.stock_data.at[idx, '戰略備註'] = st.session_state.stock_data.at[idx, '_auto_note']
        st.rerun()

    st.markdown("---")
    st.link_button("📥 Goodinfo 當日週轉率排行", "https://reurl.cc/Or9e37", use_container_width=True)

# --- 動態 CSS ---
font_px = f"{st.session_state.font_size}px"
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
# 資料功能
# ==========================================
@st.cache_data
def load_local_stock_names():
    code_map, name_map = {}, {}
    if os.path.exists("stock_names.csv"):
        try:
            df = pd.read_csv("stock_names.csv", header=None, names=["code", "name"], dtype=str)
            for _, row in df.iterrows():
                c, n = str(row['code']).strip(), str(row['name']).strip()
                code_map[c] = n
                name_map[n] = c
        except: pass
    return code_map, name_map

@st.cache_data(ttl=86400)
def get_stock_name_online(code):
    code_map, _ = load_local_stock_names()
    return code_map.get(str(code).strip(), str(code).strip())

@st.cache_data(ttl=86400)
def fetch_futures_list():
    try:
        url = "https://www.taifex.com.tw/cht/2/stockLists"
        dfs = pd.read_html(url)
        if dfs:
            for df in dfs:
                if '證券代號' in df.columns: return set(df['證券代號'].astype(str).str.strip().tolist())
                if 'Stock Code' in df.columns: return set(df['Stock Code'].astype(str).str.strip().tolist())
    except: pass
    return set()

def get_live_price(code):
    try:
        realtime_data = twstock.realtime.get(code)
        if realtime_data and realtime_data.get('success'):
            p = realtime_data['realtime'].get('latest_trade_price')
            if p and p != '-' and float(p) > 0: return float(p)
            b = realtime_data['realtime'].get('best_bid_price', [])
            if b and b[0] and b[0] != '-': return float(b[0])
    except: pass
    try:
        t = yf.Ticker(f"{code}.TW")
        p = t.fast_info.get('last_price')
        if p and not math.isnan(p): return float(p)
        t = yf.Ticker(f"{code}.TWO")
        p = t.fast_info.get('last_price')
        if p and not math.isnan(p): return float(p)
    except: pass
    return None

def fetch_yahoo_web_backup(code):
    try:
        url = f"https://tw.stock.yahoo.com/quote/{code}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        r = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(r.text, 'html.parser')
        price_tag = soup.find('span', class_='Fz(32px)')
        if not price_tag: return None
        price = float(price_tag.text.replace(',', ''))
        today = datetime.now().date()
        df = pd.DataFrame({'Open': [price], 'High': [price], 'Low': [price], 'Close': [price], 'Volume': [0]}, index=[pd.to_datetime(today)])
        return df, None
    except: return None, None

def get_tick_size(price):
    try: price = float(price)
    except: return 0.01
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
    return max(50, pixel_width)

def recalculate_row(row, points_map):
    custom_price = row.get('自訂價(可修)')
    code = row.get('代號')
    status = ""
    if pd.isna(custom_price) or str(custom_price).strip() == "": return status
    
    try:
        price = float(custom_price)
        l_up = float(row.get('當日漲停價')) if row.get('當日漲停價') else None
        l_down = float(row.get('當日跌停價')) if row.get('當日跌停價') else None
        
        if l_up and abs(price - l_up) < 0.01: status = "🔴 漲停"
        elif l_down and abs(price - l_down) < 0.01: status = "🟢 跌停"
        else:
            points = points_map.get(code, [])
            hit = False
            for p in points:
                if abs(p['val'] - price) < 0.01:
                    hit = True; break
            if not hit:
                note_text = str(row.get('戰略備註', ''))
                found_prices = re.findall(r'\d+\.?\d*', note_text)
                for fp in found_prices:
                    if abs(float(fp) - price) < 0.01: hit = True; break
            if hit: status = "🟡 命中"
        return status
    except: return status

def fetch_stock_data_raw(code, name_hint="", extra_data=None):
    code = str(code).strip()
    hist = pd.DataFrame()
    source_used = "none"

    def is_valid(df):
        if df is None or df.empty: return False
        if df.iloc[-1]['Close'] <= 0: return False
        if df.iloc[-1]['High'] < df.iloc[-1]['Close']: return False
        return True

    try:
        t = yf.Ticker(f"{code}.TW")
        h = t.history(period="3mo")
        if h.empty or not is_valid(h):
            t = yf.Ticker(f"{code}.TWO")
            h = t.history(period="3mo")
        if not h.empty and is_valid(h): hist, source_used = h, "yf"
    except: pass

    if hist.empty:
        try:
            s = twstock.Stock(code)
            d = s.fetch_31()
            if d:
                df = pd.DataFrame(d)
                df['Date'] = pd.to_datetime(df['date'])
                df = df.set_index('Date')
                df = df.rename(columns={'open':'Open', 'high':'High', 'low':'Low', 'close':'Close', 'capacity':'Volume'})
                if is_valid(df): hist, source_used = df, "tw"
        except: pass

    if hist.empty:
        df, _ = fetch_yahoo_web_backup(code)
        if df is not None: hist, source_used = df, "web"

    if hist.empty: return None

    # High/Low Clean
    hist['High'] = hist[['High', 'Close']].max(axis=1)
    hist['Low'] = hist[['Low', 'Close']].min(axis=1)

    tz = pytz.timezone('Asia/Taipei')
    now = datetime.now(tz)
    is_today_in_hist = (hist.index[-1].date() == now.date())
    is_during_trading = (now.time() < dt_time(13, 30))
    
    hist_strat = hist.copy()
    if is_during_trading and is_today_in_hist: hist_strat = hist_strat.iloc[:-1]
    elif not is_during_trading and not is_today_in_hist and source_used != "web":
        live = get_live_price(code)
        if live:
            nr = pd.DataFrame({'Open': live, 'High': live, 'Low': live, 'Close': live, 'Volume': 0}, index=[pd.to_datetime(now.date())])
            hist_strat = pd.concat([hist_strat, nr])

    if hist_strat.empty: return None

    strategy_base = hist_strat.iloc[-1]['Close']
    prev = hist_strat.iloc[-2]['Close'] if len(hist_strat) >= 2 else strategy_base
    pct_change = ((strategy_base - prev) / prev) * 100 if prev > 0 else 0
    limit_up, limit_down = calculate_limits(strategy_base)
    
    target_p = apply_sr_rules(strategy_base * 1.03, strategy_base)
    stop_p = apply_sr_rules(strategy_base * 0.97, strategy_base)
    
    points = []
    if len(hist_strat) >= 5:
        ma5_val = float((sum(Decimal(str(x)) for x in hist_strat['Close'].tail(5).values) / Decimal("5")).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
        ma5_p = apply_sr_rules(ma5_val, strategy_base)
        tag = "多" if ma5_val < strategy_base else ("空" if ma5_val > strategy_base else "平")
        points.append({"val": ma5_p, "tag": tag, "force": True})

    if len(hist_strat) >= 2:
        lc = hist_strat.iloc[-1]
        p_o = apply_tick_rules(lc['Open'])
        if limit_down <= p_o <= limit_up: points.append({"val": p_o, "tag": ""})
        p_h = apply_tick_rules(lc['High'])
        if limit_down <= p_h <= limit_up: points.append({"val": p_h, "tag": ""})
        p_l = apply_tick_rules(lc['Low'])
        if limit_down <= p_l <= limit_up: points.append({"val": p_l, "tag": ""})

    if len(hist_strat) >= 3:
        ppc = hist_strat.iloc[-2]
        pp_h = apply_tick_rules(ppc['High'])
        if limit_down <= pp_h <= limit_up: points.append({"val": pp_h, "tag": ""})
        pp_l = apply_tick_rules(ppc['Low'])
        if limit_down <= pp_l <= limit_up: points.append({"val": pp_l, "tag": ""})

    show_p3, show_m3 = False, False
    if not hist_strat.empty:
        h90 = apply_tick_rules(hist_strat['High'].max())
        l90_raw = hist_strat['Low'][hist_strat['Low'] > 0].min() if not hist_strat['Low'][hist_strat['Low'] > 0].empty else hist_strat['Low'].min()
        l90 = apply_tick_rules(l90_raw)
        points.append({"val": h90, "tag": "高"})
        points.append({"val": l90, "tag": "低"})

        if len(hist_strat) >= 2:
            prev_c = hist_strat.iloc[-2]['Close']
            l_u_t, l_d_t = calculate_limits(prev_c)
            h_t = hist_strat.iloc[-1]['High']
            l_t = hist_strat.iloc[-1]['Low']
            c_t = hist_strat.iloc[-1]['Close']
            
            if abs(h_t - l_u_t) < 0.01:
                is_new_h = (abs(l_u_t - h90) < 0.05)
                tag = "漲停高" if is_new_h else "漲停"
                if limit_down <= l_u_t <= limit_up: points.append({"val": l_u_t, "tag": tag})
                if c_t >= l_u_t * 0.97: show_p3 = True
            
            if abs(l_t - l_d_t) < 0.01:
                if c_t <= l_d_t * 1.03: show_m3 = True

    if show_p3: points.append({"val": target_p, "tag": ""})
    if show_m3: points.append({"val": stop_p, "tag": ""})

    disp_pts = []
    for p in points:
        if p.get('force') or (limit_down <= p['val'] <= limit_up): disp_pts.append(p)
    disp_pts.sort(key=lambda x: x['val'])

    final_pts = []
    for val, group in itertools.groupby(disp_pts, key=lambda x: round(x['val'], 2)):
        g = list(group)
        tags = [x['tag'] for x in g if x['tag']]
        ftag = ""
        if "漲停高" in tags: ftag = "漲停高"
        elif "跌停低" in tags: ftag = "跌停低"
        elif "漲停" in tags: ftag = "漲停"
        elif "跌停" in tags: ftag = "跌停"
        else:
            if "高" in tags: ftag = "高"
            elif "低" in tags: ftag = "低"
            elif "多" in tags: ftag = "多"
            elif "空" in tags: ftag = "空"
            elif "平" in tags: ftag = "平"
        final_pts.append({"val": val, "tag": ftag})

    note_parts = []
    seen = set()
    for p in final_pts:
        if p['val'] in seen and not p['tag']: continue
        seen.add(p['val'])
        v_s = fmt_price(p['val'])
        t = p['tag']
        item = f"{t}{v_s}" if t in ["漲停","漲停高","跌停","跌停低","高","低"] else (f"{v_s}{t}" if t else v_s)
        note_parts.append(item)

    auto_note = "-".join(note_parts)
    manual_note = st.session_state.saved_notes.get(code, "")
    full_note = f"{auto_note} {manual_note}" if manual_note else auto_note

    fname = name_hint if name_hint else get_stock_name_online(code)
    light = "⚪"
    if "多" in full_note: light = "🔴"
    elif "空" in full_note: light = "🟢"
    
    has_futures = "✅" if code in st.session_state.futures_list else ""

    return {
        "代號": code, "名稱": f"{light} {fname}", "收盤價": round(strategy_base, 2),
        "漲跌幅": pct_change, "期貨": has_futures,
        "當日漲停價": limit_up, "當日跌停價": limit_down,
        "自訂價(可修)": None, "獲利目標": target_p, "防守停損": stop_p,
        "戰略備註": full_note, "_points": final_pts, "狀態": "", "_auto_note": auto_note
    }

# ==========================================
# UI & Logic
# ==========================================
tab1, tab2 = st.tabs(["⚡ 當沖戰略室 ⚡", "💰 當沖損益室 💰"])

with tab1:
    c_search, c_file = st.columns([2, 1])
    with c_search:
        code_map, _ = load_local_stock_names()
        opts = [f"{c} {n}" for c, n in sorted(code_map.items())]
        
        t1, t2 = st.tabs(["📂 本機", "☁️ 雲端"])
        with t1:
            uploaded_file = st.file_uploader("上傳", type=['xlsx','csv','html','xls'], label_visibility="collapsed")
            sheet_idx = 0
            if uploaded_file and not uploaded_file.name.endswith('.csv'):
                try: sheet_idx = st.selectbox("工作表", pd.ExcelFile(uploaded_file).sheet_names, index=0)
                except: pass
        with t2:
            def on_hist(): st.session_state.cloud_url_input = st.session_state.hist_sel
            sel = st.selectbox("歷史", st.session_state.url_history or ["(無)"], key="hist_sel", index=None, on_change=on_hist, label_visibility="collapsed")
            url_in = st.text_input("URL", key="cloud_url_input")

        def update_search(): save_search_cache(st.session_state.search_multiselect)
        search_sel = st.multiselect("🔍 快速查詢", opts, key="search_multiselect", on_change=update_search)

    if st.button("🚀 執行分析"):
        save_search_cache(st.session_state.search_multiselect)
        if not st.session_state.futures_list: st.session_state.futures_list = fetch_futures_list()
        
        targets = []
        df_in = pd.DataFrame()
        
        if url_in and url_in not in st.session_state.url_history:
            st.session_state.url_history.insert(0, url_in)
            save_url_history(st.session_state.url_history)

        try:
            if uploaded_file:
                uploaded_file.seek(0)
                fn = uploaded_file.name.lower()
                if fn.endswith('.csv'): 
                    try: df_in = pd.read_csv(uploaded_file, dtype=str, encoding='cp950')
                    except: 
                        uploaded_file.seek(0)
                        df_in = pd.read_csv(uploaded_file, dtype=str)
                elif fn.endswith(('html','htm','xls')): 
                    dfs = pd.read_html(uploaded_file, encoding='cp950')
                    for d in dfs:
                        if d.apply(lambda r: r.astype(str).str.contains('代號').any(), axis=1).any():
                            df_in = d
                            for i, row in d.iterrows():
                                if "代號" in row.values:
                                    df_in.columns = row; df_in = df_in.iloc[i+1:]; break
                            break
                elif fn.endswith('.xlsx'): df_in = pd.read_excel(uploaded_file, sheet_name=sheet_idx, dtype=str)
            elif url_in:
                u = url_in
                if "docs.google.com" in u: u = u.split("/edit")[0] + "/export?format=csv"
                try: df_in = pd.read_csv(u, dtype=str)
                except: df_in = pd.read_excel(u, dtype=str)
        except Exception as e: st.error(f"Error: {e}")

        if search_sel:
            for s in search_sel:
                p = s.split(' ', 1)
                targets.append((p[0], p[1] if len(p)>1 else "", 'search', 9999))

        if not df_in.empty:
            df_in.columns = df_in.columns.astype(str).str.strip()
            cc = next((c for c in df_in.columns if "代號" in str(c)), None)
            nn = next((c for c in df_in.columns if "名稱" in str(c)), None)
            if cc:
                cnt = 0
                for _, r in df_in.iterrows():
                    c_raw = str(r[cc]).replace('=','').replace('"','').strip()
                    if not c_raw or c_raw.lower()=='nan': continue
                    if hide_non_stock and (c_raw.startswith('00') or len(c_raw)>4): continue
                    if c_raw in st.session_state.ignored_stocks: continue
                    n_raw = str(r[nn]) if nn else ""
                    targets.append((c_raw, n_raw, 'upload', cnt))
                    cnt += 1

        st.session_state.all_candidates = targets
        st.session_state.stock_data = pd.DataFrame()
        
        seen = set()
        bar = st.progress(0)
        tot = len(search_sel) + min(len([t for t in targets if t[2]=='upload']), st.session_state.limit_rows)
        if tot == 0: tot = 1
        fetched = 0
        up_curr = 0
        
        cache = {}
        batch_data = []
        
        for c, n, src, order in targets:
            if src == 'upload':
                if up_curr >= st.session_state.limit_rows: continue
                up_curr += 1
            
            if (c, src) in seen: continue
            seen.add((c, src))
            
            if c not in cache: cache[c] = fetch_stock_data_raw(c, n, order)
            d = cache[c]
            
            if d:
                d['_source'] = src
                d['_order'] = order
                d['_source_rank'] = 1 if src=='upload' else 2
                batch_data.append(d)
                fetched += 1
                bar.progress(min(fetched/tot, 1.0))
        
        if batch_data:
            st.session_state.stock_data = pd.DataFrame(batch_data)
            save_data_cache(st.session_state.stock_data, st.session_state.ignored_stocks, st.session_state.all_candidates)
        bar.empty()

    if not st.session_state.stock_data.empty:
        df_show = st.session_state.stock_data.copy()
        if '_source_rank' in df_show.columns: df_show = df_show.sort_values(by=['_source_rank', '_order'])
        df_show = df_show.reset_index(drop=True)
        
        # [FIXED] 準確計算 Note Width
        note_w = calculate_note_width(df_show['戰略備註'], st.session_state.font_size)
        
        # 建立自動備註對照表
        auto_notes_map = {}
        if '_auto_note' in df_show.columns:
            auto_notes_map = df_show.set_index('代號')['_auto_note'].to_dict()
        
        points_map = {}
        if '_points' in df_show.columns:
            points_map = df_show.set_index('代號')['_points'].to_dict()

        cols = ["移除", "代號", "名稱", "戰略備註", "自訂價(可修)", "狀態", "當日漲停價", "當日跌停價", "+3%", "-3%", "收盤價", "漲跌幅", "期貨"]
        for c in cols: 
            if c not in df_show.columns: df_show[c] = None

        # 格式化數值
        fmt_cols = ["當日漲停價", "當日跌停價", "+3%", "-3%", "自訂價(可修)"]
        for c in fmt_cols: df_show[c] = df_show[c].apply(fmt_price)

        # 顏色處理 (Emoji)
        for i in df_show.index:
            try:
                p, chg = float(df_show.at[i, "收盤價"]), float(df_show.at[i, "漲跌幅"])
                icon = "🔴" if chg > 0 else ("🟢" if chg < 0 else "⚪")
                df_show.at[i, "收盤價"] = f"{icon} {fmt_price(p)}"
                df_show.at[i, "漲跌幅"] = f"{icon} {chg:+.2f}%"
            except: pass

        # data_editor
        edited = st.data_editor(
            df_show[cols],
            column_config={
                "移除": st.column_config.CheckboxColumn("刪", width=40),
                "代號": st.column_config.TextColumn(width=50, disabled=True),
                "名稱": st.column_config.TextColumn(width="small", disabled=True),
                "收盤價": st.column_config.TextColumn(width="small", disabled=True),
                "漲跌幅": st.column_config.TextColumn(width="small", disabled=True),
                "期貨": st.column_config.TextColumn(width=40, disabled=True),
                "自訂價(可修)": st.column_config.TextColumn("自訂價 ✏️", width=60),
                "當日漲停價": st.column_config.TextColumn(width="small", disabled=True),
                "當日跌停價": st.column_config.TextColumn(width="small", disabled=True),
                "+3%": st.column_config.TextColumn(width="small", disabled=True),
                "-3%": st.column_config.TextColumn(width="small", disabled=True),
                "狀態": st.column_config.TextColumn(width=60, disabled=True),
                "戰略備註": st.column_config.TextColumn("戰略備註 ✏️", width=note_w),
            },
            hide_index=True,
            use_container_width=False,
            num_rows="fixed",
            key="main_editor"
        )

        # 刪除邏輯
        if not edited.empty:
            to_del = edited[edited["移除"] == True]
            if not to_del.empty:
                codes = to_del["代號"].unique()
                for c in codes: st.session_state.ignored_stocks.add(str(c))
                st.session_state.stock_data = st.session_state.stock_data[~st.session_state.stock_data["代號"].isin(codes)]
                st.rerun()

        # [CRITICAL FIX] 自動更新與備註邏輯
        if not edited.empty:
            # 將 edited_df 轉為字典以便快速查找，key 為 Index (因為順序可能沒變)
            # 注意：st.data_editor 的 index 對應 df_show 的 index
            # 我們需要更新回 st.session_state.stock_data (它是原始資料)
            
            # 因為 df_show 可能經過排序，我們用 '代號' 作為 Key 來對應
            edit_map = edited.set_index('代號')[['自訂價(可修)', '戰略備註']].to_dict('index')
            
            # 取得「顯示在畫面上的最後一列」的代號
            last_visible_code = edited.iloc[-1]['代號']
            
            should_rerun = False
            
            for i, row in st.session_state.stock_data.iterrows():
                code = row['代號']
                if code in edit_map:
                    new_p = str(edit_map[code]['自訂價(可修)']).strip()
                    new_n = str(edit_map[code]['戰略備註']).strip()
                    
                    old_p = str(row['自訂價(可修)']).strip()
                    old_n = str(row['戰略備註']).strip()
                    
                    # 1. 處理自訂價 (靜默更新)
                    if old_p != new_p:
                        st.session_state.stock_data.at[i, '自訂價(可修)'] = new_p
                        # 只有當「最後一列」變動時，才標記需要重跑
                        if code == last_visible_code and st.session_state.auto_update_last_row:
                            should_rerun = True
                    
                    # 2. 處理備註 (分離儲存，靜默更新)
                    if old_n != new_n:
                        # 寫回原始資料，確保畫面同步
                        st.session_state.stock_data.at[i, '戰略備註'] = new_n
                        
                        # 解析手動部分：移除開頭的自動文字
                        auto_txt = auto_notes_map.get(code, "")
                        manual_part = new_n
                        if auto_txt and new_n.startswith(auto_txt):
                            manual_part = new_n[len(auto_txt):].strip()
                        
                        st.session_state.saved_notes[code] = manual_part

            # 3. 觸發更新條件：是最後一列變動，且值不為空
            if should_rerun:
                last_val = str(edit_map[last_visible_code]['自訂價(可修)']).strip()
                if last_val:
                    time.sleep(1.0) # 強制緩衝，讓使用者感覺到「輸入完成了」
                    
                    # 重算狀態
                    for i, row in st.session_state.stock_data.iterrows():
                        new_s = recalculate_row(row, points_map)
                        st.session_state.stock_data.at[i, '狀態'] = new_s
                    st.rerun()

        st.markdown("---")
        if st.button("⚡ 執行更新", type="primary"):
            for i, row in st.session_state.stock_data.iterrows():
                st.session_state.stock_data.at[i, '狀態'] = recalculate_row(row, points_map)
            st.rerun()

# [Tab 2 損益試算 保持不變]
with tab2:
    st.markdown("#### 💰 當沖損益室 💰")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        cp = st.number_input("基準價", value=float(st.session_state.calc_base_price), step=0.01, format="%.2f")
        if cp != st.session_state.calc_base_price:
            st.session_state.calc_base_price = cp
            st.session_state.calc_view_price = apply_tick_rules(cp)
    with c2: sh = st.number_input("股數", value=1000, step=1000)
    with c3: dis = st.number_input("折扣", value=2.8, step=0.1)
    with c4: mf = st.number_input("低消", value=20)
    with c5: tc = st.number_input("檔數", value=5)
    dire = st.radio("方向", ["多", "空"], horizontal=True)
    lu, ld = calculate_limits(st.session_state.calc_base_price)
    
    b1, b2, _ = st.columns([1,1,6])
    with b1: 
        if st.button("🔼"): 
            st.session_state.calc_view_price = move_tick(st.session_state.calc_view_price, tc)
            st.rerun()
    with b2:
        if st.button("🔽"): 
            st.session_state.calc_view_price = move_tick(st.session_state.calc_view_price, -tc)
            st.rerun()

    tr = range(tc, -(tc+1), -1)
    cd = []
    bp = st.session_state.calc_base_price
    vp = st.session_state.calc_view_price
    is_long = "多" in dire
    
    for i in tr:
        p = move_tick(vp, i)
        if p > lu or p < ld: continue
        if is_long:
            cost = (bp*sh) + max(mf, math.floor(bp*sh*0.001425*dis/10))
            inc = (p*sh) - max(mf, math.floor(p*sh*0.001425*dis/10)) - math.floor(p*sh*0.0015)
            prof = inc - cost
        else:
            inc = (bp*sh) - max(mf, math.floor(bp*sh*0.001425*dis/10)) - math.floor(bp*sh*0.0015)
            cost = (p*sh) + max(mf, math.floor(p*sh*0.001425*dis/10))
            prof = inc - cost
        
        nt = "up" if abs(p-lu)<0.001 else ("down" if abs(p-ld)<0.001 else "")
        ib = (abs(p-bp)<0.001)
        
        cd.append({
            "成交價": fmt_price(p), "漲跌": f"{p-bp:+.2f}", "預估損益": int(prof), 
            "報酬率%": f"{(prof/(bp*sh)*100):+.2f}%" if bp*sh!=0 else "0%",
            "_profit": prof, "_note": nt, "_base": ib
        })
        
    dcf = pd.DataFrame(cd)
    def sty(r):
        if r['_base']: return ['background-color: #ffffcc; color: black; font-weight: bold']*len(r)
        if r['_note']=='up': return ['background-color: #ff4b4b; color: white']*len(r)
        if r['_note']=='down': return ['background-color: #00cc00; color: white']*len(r)
        return ['color: #ff4b4b' if r['_profit']>0 else ('color: #00cc00' if r['_profit']<0 else 'color: gray')]*len(r)

    if not dcf.empty:
        st.dataframe(dcf.style.apply(sty, axis=1), use_container_width=False, hide_index=True, height=(len(dcf)+1)*35, column_config={"_profit":None,"_note":None,"_base":None})
