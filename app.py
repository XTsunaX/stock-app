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

# ==========================================
# 0. 頁面設定與初始化
# ==========================================
st.set_page_config(page_title="當沖戰略室", page_icon="⚡", layout="wide")

# 1. 確保標題在最上方顯示
st.title("⚡ 當沖戰略室 ⚡")

CONFIG_FILE = "config.json"

def load_config():
    """讀取設定檔"""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_config(font_size, limit_rows):
    """儲存設定檔"""
    try:
        config = {"font_size": font_size, "limit_rows": limit_rows}
        with open(CONFIG_FILE, "w") as f:
            json.dump(config, f)
        return True
    except:
        return False

# --- 初始化 Session State ---
if 'stock_data' not in st.session_state:
    st.session_state.stock_data = pd.DataFrame()

# 計算機用的 Session State
if 'calc_base_price' not in st.session_state:
    st.session_state.calc_base_price = 100.0

# 優先從設定檔讀取
saved_config = load_config()

if 'font_size' not in st.session_state:
    st.session_state.font_size = saved_config.get('font_size', 18)

if 'limit_rows' not in st.session_state:
    st.session_state.limit_rows = saved_config.get('limit_rows', 5)

# --- 側邊欄設定 ---
with st.sidebar:
    st.header("⚙️ 設定")
    
    current_font_size = st.slider(
        "字體大小 (表格)", 
        min_value=12, 
        max_value=72, 
        key='font_size'
    )
    
    hide_etf = st.checkbox("隱藏 ETF (00開頭)", value=True)
    st.markdown("---")
    
    current_limit_rows = st.number_input(
        "顯示筆數", 
        min_value=1, 
        key='limit_rows'
    )
    
    if st.button("💾 儲存設定"):
        if save_config(current_font_size, current_limit_rows):
            st.toast("設定已儲存！下次開啟將自動套用。", icon="✅")
        else:
            st.error("設定儲存失敗。")
    
    st.caption("功能說明")
    st.info("🗑️ **如何刪除股票？**\n\n勾選左側框框後按 `Delete` 鍵。")

# --- 動態 CSS ---
font_px = f"{st.session_state.font_size}px"

st.markdown(f"""
    <style>
    .block-container {{ padding-top: 0.5rem; padding-bottom: 1rem; }}
    
    /* 套用到所有 Streamlit 表格相關元素 */
    div[data-testid="stDataFrame"] table,
    div[data-testid="stDataFrame"] td,
    div[data-testid="stDataFrame"] th,
    div[data-testid="stDataFrame"] input,
    div[data-testid="stDataFrame"] div,
    div[data-testid="stDataFrame"] span {{
        font-size: {font_px} !important;
        font-family: 'Microsoft JhengHei', sans-serif !important;
        line-height: 1.5 !important;
    }}
    
    div[data-testid="stDataFrame"] {{
        width: 100%;
    }}
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
# 2. 核心計算邏輯 (含台股 Tick 規則)
# ==========================================

def get_tick_size(price):
    """取得台股價格對應的跳動檔位"""
    if price < 10: return 0.01
    if price < 50: return 0.05
    if price < 100: return 0.1
    if price < 500: return 0.5
    if price < 1000: return 1.0
    return 5.0

def calculate_limits(price):
    """計算漲跌停價 (10%)"""
    try:
        p = float(price)
        raw_up = p * 1.10
        tick_up = get_tick_size(raw_up) 
        limit_up = math.floor(raw_up / tick_up) * tick_up
        
        raw_down = p * 0.90
        tick_down = get_tick_size(raw_down) 
        limit_down = math.ceil(raw_down / tick_down) * tick_down
        
        return float(f"{limit_up:.2f}"), float(f"{limit_down:.2f}")
    except:
        return 0, 0

def apply_tick_rules(price):
    """將任意價格修正為符合台股 Tick 規則的價格"""
    try:
        p = float(price)
        tick = get_tick_size(p)
        rounded_price = round(p / tick) * tick
        return float(f"{rounded_price:.2f}")
    except:
        return price

def move_tick(price, steps):
    """計算價格往上或往下 N 檔後的價格"""
    curr = float(price)
    if steps > 0:
        for _ in range(steps):
            tick = get_tick_size(curr)
            curr = round(curr + tick, 2)
    elif steps < 0:
        for _ in range(abs(steps)):
            tick = get_tick_size(curr - 0.0001) # 往下時取下一檔的 tick
            curr = round(curr - tick, 2)
    return curr

def fetch_stock_data_raw(code, name_hint="", extra_data=None):
    code = str(code).strip()
    try:
        ticker = yf.Ticker(f"{code}.TW")
        hist = ticker.history(period="3mo") 
        if hist.empty:
            ticker = yf.Ticker(f"{code}.TWO")
            hist = ticker.history(period="3mo")
        if hist.empty: return None

        today = hist.iloc[-1]
        current_price = today['Close']
        prev_day = hist.iloc[-2] if len(hist) >= 2 else today
        
        pct_change = ((current_price - prev_day['Close']) / prev_day['Close']) * 100
        
        # 1. 欄位顯示用的數據 (以收盤價為基準)
        target_price = apply_tick_rules(current_price * 1.03)
        stop_price = apply_tick_rules(current_price * 0.97)
        limit_up_col, limit_down_col = calculate_limits(current_price) 

        # 2. 戰略備註用的漲跌停參考 (以昨日收盤為基準)
        limit_up_today, limit_down_today = calculate_limits(prev_day['Close'])

        # 點位收集
        points = []
        ma5 = apply_tick_rules(hist['Close'].tail(5).mean())
        points.append({"val": ma5, "tag": "多" if current_price > ma5 else "空"})
        points.append({"val": apply_tick_rules(today['Open']), "tag": ""})
        points.append({"val": apply_tick_rules(today['High']), "tag": ""})
        points.append({"val": apply_tick_rules(today['Low']), "tag": ""})
        
        past_5 = hist.iloc[-6:-1] if len(hist) >= 6 else hist.iloc[:-1]
        if not past_5.empty:
            points.append({"val": apply_tick_rules(past_5['High'].max()), "tag": ""})
            points.append({"val": apply_tick_rules(past_5['Low'].min()), "tag": ""})
            
        high_90 = apply_tick_rules(hist['High'].max())
        low_90 = apply_tick_rules(hist['Low'].min())
        points.append({"val": high_90, "tag": "高"})
        points.append({"val": low_90, "tag": "低"})

        # 戰略備註整理
        display_candidates = []
        for p in points:
            v = float(f"{p['val']:.2f}")
            
            # 備註過濾邏輯：確保顯示的點位不超過收盤價的 +/- 10% (limit_up_col)
            is_in_range = limit_down_col <= v <= limit_up_col
            is_5ma = "多" in p['tag'] or "空" in p['tag']
            
            if is_in_range or is_5ma:
                display_candidates.append({"val": v, "tag": p['tag']})
        
        # 檢查是否觸及今日漲跌停 (基於昨日收盤價)
        touched_up = today['High'] >= limit_up_today - 0.01
        touched_down = today['Low'] <= limit_down_today + 0.01

        if touched_up:
            display_candidates.append({"val": limit_up_today, "tag": "漲停"})
        if touched_down:
            display_candidates.append({"val": limit_down_today, "tag": "跌停"})
            
        display_candidates.sort(key=lambda x: x['val'])
        
        final_display_points = []
        extra_points = [] 

        for val, group in itertools.groupby(display_candidates, key=lambda x: round(x['val'], 2)):
            g_list = list(group)
            tags = [x['tag'] for x in g_list]
            
            final_tag = ""
            is_limit_up = "漲停" in tags
            is_limit_down = "跌停" in tags
            is_high = "高" in tags
            is_low = "低" in tags
            
            is_close_price = abs(val - current_price) < 0.01
            
            # --- 漲停高/跌停低 + 延伸計算 ---
            if is_limit_up:
                if is_high and is_close_price: 
                    final_tag = "漲停高"
                    ext_val = apply_tick_rules(val * 1.03)
                    extra_points.append({"val": ext_val, "tag": ""})
                else:
                    final_tag = "漲停"
                    
            elif is_limit_down:
                if is_low and is_close_price:
                    final_tag = "跌停低"
                    ext_val = apply_tick_rules(val * 0.97)
                    extra_points.append({"val": ext_val, "tag": ""})
                else:
                    final_tag = "跌停"
            else:
                if is_high: final_tag = "高"
                elif is_low: final_tag = "低"
                elif "多" in tags: final_tag = "多"
                elif "空" in tags: final_tag = "空"
                else: final_tag = ""

            final_display_points.append({"val": val, "tag": final_tag})
        
        if extra_points:
            for ep in extra_points:
                final_display_points.append(ep)
            final_display_points.sort(key=lambda x: x['val'])
            
        note_parts = []
        seen_vals = set() 
        for p in final_display_points:
            if p['val'] in seen_vals and p['tag'] == "": continue
            seen_vals.add(p['val'])
            v_str = f"{p['val']:.0f}" if p['val'].is_integer() else f"{p['val']:.2f}"
            t = p['tag']
            if t in ["漲停", "漲停高", "跌停", "跌停低", "高", "低"]:
                item = f"{t}{v_str}"
            elif t: 
                item = f"{v_str}{t}"
            else: 
                item = v_str
            note_parts.append(item)
        
        strategy_note = "-".join(note_parts)
        final_name = name_hint if name_hint else get_stock_name_online(code)
        
        return {
            "代號": code,
            "名稱": final_name,
            "收盤價": round(current_price, 2),
            "漲跌幅": pct_change, 
            "當日漲停價": limit_up_col,   
            "當日跌停價": limit_down_col,
            "自訂價(可修)": None, 
            "獲利目標": target_price, 
            "防守停損": stop_price,   
            "戰略備註": strategy_note,
            "_points": full_calc_points
        }
    except Exception as e:
        return None

# ==========================================
# 主介面 (Tabs)
# ==========================================

tab1, tab2 = st.tabs(["⚡ 當沖戰略室 ⚡", "💰 當沖損益試算 💰"])

# -------------------------------------------------------
# Tab 1: 當沖戰略室
# -------------------------------------------------------
with tab1:
    col_search, col_file = st.columns([2, 1])
    with col_search:
        search_query = st.text_input("🔍 快速查詢 (中文/代號)", placeholder="鴻海, 2603, 緯創")
    with col_file:
        uploaded_file = st.file_uploader("📂 上傳清單", type=['xlsx', 'csv'])
        selected_sheet = None
        if uploaded_file:
            try:
                if uploaded_file.name.endswith('.csv'):
                    xl = None 
                else:
                    xl = pd.ExcelFile(uploaded_file) 
            except ImportError:
                st.error("❌ 讀取 Excel 失敗：環境缺少 `openpyxl` 套件。")
            except Exception as e:
                st.error(f"❌ 讀取檔案失敗: {e}")

            if xl:
                default_idx = 0
                if "週轉率" in xl.sheet_names: default_idx = xl.sheet_names.index("週轉率")
                selected_sheet = st.selectbox("工作表", xl.sheet_names, index=default_idx)

    if st.button("🚀 執行分析", type="primary"):
        targets = []
        
        # 1. 處理上傳清單
        if uploaded_file:
            try:
                if uploaded_file.name.endswith('.csv'): 
                    df_up = pd.read_csv(uploaded_file)
                else: 
                    df_up = pd.read_excel(uploaded_file, sheet_name=selected_sheet)
                
                c_col = next((c for c in df_up.columns if "代號" in c), None)
                n_col = next((c for c in df_up.columns if "名稱" in c), None)
                
                if c_col:
                    for _, row in df_up.iterrows():
                        c = str(row[c_col]).split('.')[0].strip()
                        if c.isdigit():
                            if len(c) < 4: c = c.zfill(4) 
                            n = str(row[n_col]) if n_col else ""
                            targets.append((c, n, 'upload', {}))
            except Exception as e:
                st.error(f"讀取失敗: {e}")

        # 2. 處理搜尋輸入
        if search_query:
            inputs = [x.strip() for x in search_query.replace('，',',').split(',') if x.strip()]
            for inp in inputs:
                if inp.isdigit(): 
                    targets.append((inp, "", 'search', {}))
                else:
                    with st.spinner(f"搜尋「{inp}」..."):
                        code = search_code_online(inp)
                    if code: 
                        targets.append((code, inp, 'search', {}))
                    else: 
                        st.toast(f"找不到「{inp}」", icon="⚠️")

        results = []
        seen = set()
        bar = st.progress(0)
        total = len(targets)
        
        for i, (code, name, source, extra) in enumerate(targets):
            if code in seen: continue
            if hide_etf and code.startswith("00"): continue
            
            data = fetch_stock_data_raw(code, name, extra)
            if data:
                data['_source'] = source
                results.append(data)
                seen.add(code)
            if total > 0: bar.progress((i+1)/total)
        
        bar.empty()
        if results:
            st.session_state.stock_data = pd.DataFrame(results)

    if not st.session_state.stock_data.empty:
        limit = st.session_state.limit_rows
        df_all = st.session_state.stock_data
        
        if '_source' in df_all.columns:
            df_up = df_all[df_all['_source'] == 'upload'].head(limit)
            df_se = df_all[df_all['_source'] == 'search']
            df_display = pd.concat([df_up, df_se]).reset_index(drop=True)
        else:
            df_display = df_all.head(limit).reset_index(drop=True)
        
        # 3. 欄位排序更新
        input_cols = ["代號", "名稱", "收盤價", "漲跌幅", "戰略備註", "自訂價(可修)", "當日漲停價", "當日跌停價", "獲利目標", "防守停損", "_points"]
        
        for col in input_cols:
            if col not in df_display.columns and col != "_points":
                df_display[col] = None

        edited_df = st.data_editor(
            df_display[input_cols],
            column_config={
                "代號": st.column_config.TextColumn(disabled=True, width="small"),
                "名稱": st.column_config.TextColumn(disabled=True, width="medium"),
                "收盤價": st.column_config.NumberColumn(format="%.2f", disabled=True),
                "漲跌幅": st.column_config.NumberColumn(format="%.2f%%", disabled=True),
                "自訂價(可修)": st.column_config.NumberColumn(
                    "自訂價 ✏️",
                    help="輸入後查看命中結果",
                    format="%.2f",
                    step=0.01,
                    required=False,
                    width="medium" 
                ),
                "當日漲停價": st.column_config.NumberColumn("當日漲停價", format="%.2f", disabled=True),
                "當日跌停價": st.column_config.NumberColumn("當日跌停價", format="%.2f", disabled=True),
                "獲利目標": st.column_config.NumberColumn("+3%", format="%.2f", disabled=True),
                "防守停損": st.column_config.NumberColumn("-3%", format="%.2f", disabled=True),
                "戰略備註": st.column_config.TextColumn(width="large", disabled=True),
                "_points": None 
            },
            hide_index=True, # 隱藏索引
            use_container_width=True,
            num_rows="dynamic",
            key="main_editor"
        )
        
        results_hit = []
        for idx, row in edited_df.iterrows():
            custom_price = row['自訂價(可修)']
            hit_type = 'none'

            if not (pd.isna(custom_price) or custom_price == ""):
                price = float(custom_price)
                points = row['_points']
                limit_up = df_display.at[idx, '當日漲停價']
                limit_down = df_display.at[idx, '當日跌停價']
                
                if abs(price - limit_up) < 0.01:
                    hit_type = 'up' 
                elif abs(price - limit_down) < 0.01:
                    hit_type = 'down'
                else:
                    for p in points:
                        if abs(p['val'] - price) < 0.01:
                            hit_type = 'normal'
                            break
                            
            results_hit.append({"_hit_type": hit_type})
        
        res_df_calced = pd.DataFrame(results_hit, index=edited_df.index)
        final_df = pd.concat([edited_df, res_df_calced], axis=1)

        st.markdown("### 🎯 計算結果 (命中亮色提示)")
        
        mask = final_df['自訂價(可修)'].notna() & (final_df['自訂價(可修)'] != "")
        
        if mask.any():
            display_cols = ["代號", "名稱", "自訂價(可修)", "獲利目標", "防守停損", "戰略備註", "_hit_type"]
            display_df = final_df[mask][display_cols]
            
            def highlight_hit_row(row):
                t = row['_hit_type']
                if t == 'up':
                    return ['background-color: #ff4b4b; color: white; font-weight: bold;'] * len(row)
                elif t == 'down':
                    return ['background-color: #00cc00; color: white; font-weight: bold;'] * len(row)
                elif t == 'normal':
                    return ['background-color: #fff9c4; color: black; font-weight: bold;'] * len(row)
                return [''] * len(row)

            st.dataframe(
                display_df.style.apply(highlight_hit_row, axis=1),
                use_container_width=True,
                hide_index=True, 
                column_config={
                    "自訂價(可修)": st.column_config.NumberColumn("自訂價", format="%.2f"),
                    "獲利目標": st.column_config.NumberColumn("+3%", format="%.2f"),
                    "防守停損": st.column_config.NumberColumn("-3%", format="%.2f"),
                    "_hit_type": None 
                }
            )

# -------------------------------------------------------
# Tab 2: 當沖損益計算
# -------------------------------------------------------
with tab2:
    st.markdown("#### 💰 當沖損益試算 💰")
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        calc_price = st.number_input("基準價格", value=st.session_state.calc_base_price, step=0.1, format="%.2f")
        st.session_state.calc_base_price = calc_price
    with c2:
        shares = st.number_input("股數", value=1000, step=1000)
    with c3:
        discount = st.number_input("手續費折扣 (折)", value=2.8, step=0.1, min_value=0.1, max_value=10.0)
    with c4:
        min_fee = st.number_input("最低手續費 (元)", value=20, step=1)
        
    direction = st.radio("交易方向", ["當沖多 (先買後賣)", "當沖空 (先賣後買)"], horizontal=True)
    
    b1, b2, _ = st.columns([1, 1, 6])
    with b1:
        if st.button("🔼 向上", use_container_width=True):
            st.session_state.calc_base_price = move_tick(st.session_state.calc_base_price, 5)
            st.rerun()
    with b2:
        if st.button("🔽 向下", use_container_width=True):
            st.session_state.calc_base_price = move_tick(st.session_state.calc_base_price, -5)
            st.rerun()
            
    ticks_range = range(10, -11, -1) 
    calc_data = []
    
    base_p = st.session_state.calc_base_price
    is_long = "多" in direction
    
    fee_rate = 0.001425
    tax_rate = 0.0015 
    
    for i in ticks_range:
        p = move_tick(base_p, i)
        
        if is_long:
            buy_price = base_p
            sell_price = p
            
            buy_fee = max(min_fee, math.floor(buy_price * shares * fee_rate * (discount/10)))
            sell_fee = max(min_fee, math.floor(sell_price * shares * fee_rate * (discount/10)))
            tax = math.floor(sell_price * shares * tax_rate)
            
            cost = (buy_price * shares) + buy_fee
            income = (sell_price * shares) - sell_fee - tax
            profit = income - cost
            
            total_fee = buy_fee + sell_fee
            
        else: 
            sell_price = base_p
            buy_price = p
            
            sell_fee = max(min_fee, math.floor(sell_price * shares * fee_rate * (discount/10)))
            buy_fee = max(min_fee, math.floor(buy_price * shares * fee_rate * (discount/10)))
            tax = math.floor(sell_price * shares * tax_rate)
            
            income = (sell_price * shares) - sell_fee - tax
            cost = (buy_price * shares) + buy_fee
            profit = income - cost
            
            total_fee = buy_fee + sell_fee
            
        roi = (profit / (base_p * shares)) * 100
        
        diff = p - base_p
        diff_str = f"{diff:+.2f}" if diff != 0 else "0.00"
        
        calc_data.append({
            "成交價": f"{p:.2f}",
            "漲跌": diff_str,
            "預估損益": int(profit),
            "報酬率%": f"{roi:+.2f}%",
            "手續費": int(total_fee),
            "交易稅": int(tax),
            "_profit": profit
        })
        
    df_calc = pd.DataFrame(calc_data)
    
    def style_calc_row(row):
        prof = row['_profit']
        if prof > 0:
            return ['color: #ff4b4b; font-weight: bold'] * len(row) 
        elif prof < 0:
            return ['color: #00cc00; font-weight: bold'] * len(row) 
        else:
            return ['color: gray'] * len(row)

    st.dataframe(
        df_calc.style.apply(style_calc_row, axis=1),
        use_container_width=True,
        hide_index=True,
        column_config={
            "_profit": None
        }
    )
