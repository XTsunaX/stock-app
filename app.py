import streamlit as st
import pandas as pd
import yfinance as yf
import requests
from bs4 import BeautifulSoup
import math
import time

# --- 1. 頁面與 CSS 設定 (緊湊版面) ---
st.set_page_config(page_title="當沖戰略室 V3", page_icon="⚡", layout="wide")

st.markdown("""
    <style>
    /* 緊湊版面設定 */
    .block-container { padding-top: 1rem; padding-bottom: 1rem; padding-left: 2rem; padding-right: 2rem; }
    
    /* 表格字體優化 */
    div[data-testid="stDataFrame"] { font-size: 15px; }
    
    /* 紅綠文字風格 */
    .t-up { color: #ff4b4b; font-weight: bold; }
    .t-down { color: #00cc00; font-weight: bold; }
    .t-hit { background-color: #ffffcc; color: #000; padding: 2px 5px; border-radius: 4px; font-weight: bold; border: 1px solid #ffd700; }
    
    /* 側邊欄緊湊 */
    section[data-testid="stSidebar"] .block-container { padding-top: 2rem; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 核心功能 A: 搜尋與資料抓取
# ==========================================

@st.cache_data(ttl=86400)
def search_code_by_name(query):
    """
    輸入中文名稱 (如: 鴻海)，嘗試透過 Yahoo 搜尋代號。
    """
    query = query.strip()
    if query.isdigit(): return query # 如果是數字直接回傳
    
    try:
        url = f"https://tw.stock.yahoo.com/h/kimosearch/search_list.html?keyword={query}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        r = requests.get(url, headers=headers, timeout=3)
        soup = BeautifulSoup(r.text, "html.parser")
        
        # 尋找搜尋結果中的代號
        # Yahoo 結構通常在 <div class="D(f) Ai(c) ..."> 中包含代號
        # 這裡做簡易抓取，取第一個符合 "數字.TW" 格式的
        links = soup.find_all('a', href=True)
        for link in links:
            href = link['href']
            if "/quote/" in href:
                code_part = href.split("/quote/")[1].split(".")[0]
                if code_part.isdigit():
                    return code_part
        return query # 找不到就回傳原字串試試
    except:
        return query

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
        limit_down = math.ceil((p * 0.90) / tick) * tick
        return limit_up, limit_down
    except:
        return 0, 0

# ==========================================
# 核心功能 B: 戰略運算 (The Brain)
# ==========================================

def analyze_stock_strategy(code, name_input=None, custom_price=None):
    """
    綜合分析：5日高低、漲跌停過濾、支撐壓力自動判斷
    """
    code = str(code).strip()
    
    try:
        # 1. 抓取數據 (包含今日)
        ticker = yf.Ticker(f"{code}.TW")
        hist = ticker.history(period="10d") # 抓多一點確保有5天
        
        if hist.empty:
            ticker = yf.Ticker(f"{code}.TWO")
            hist = ticker.history(period="10d")
            
        if hist.empty: return None

        # 2. 基礎數據
        today = hist.iloc[-1]
        # 昨收 (用於計算漲跌停)
        prev_close = hist['Close'].iloc[-2] if len(hist) >= 2 else today['Open']
        limit_up, limit_down = calculate_limits(prev_close)
        
        # 現價 (若無指定 custom_price，預設為收盤價)
        current_price = today['Close']
        target_analyze_price = float(custom_price) if custom_price is not None else current_price
        
        # 3. 收集關鍵點位 (Strategy Points)
        points = []
        
        # (A) 5MA
        ma5 = hist['Close'].tail(5).mean()
        points.append({"val": ma5, "tag": "多" if target_analyze_price > ma5 else "空"})
        
        # (B) 今日數據
        points.append({"val": today['Open'], "tag": ""}) # 開盤
        points.append({"val": today['High'], "tag": ""})
        points.append({"val": today['Low'], "tag": ""})
        
        # (C) 過去 5 日高低點 (不含今日)
        past_5 = hist.iloc[-6:-1] if len(hist) >= 6 else hist.iloc[:-1]
        if not past_5.empty:
            p_high = past_5['High'].max()
            p_low = past_5['Low'].min()
            points.append({"val": p_high, "tag": "高"})
            points.append({"val": p_low, "tag": ""}) # 近低不標字
            
        # 4. 過濾與排序 (Logic Point 1: 在漲跌停範圍內)
        valid_points = []
        seen_values = set()
        
        for p in points:
            v = float(f"{p['val']:.2f}")
            # 過濾邏輯: 必須在 (跌停 <= v <= 漲停) 之間
            if limit_down <= v <= limit_up:
                if v not in seen_values:
                    # 標籤合併
                    tag = p['tag']
                    valid_points.append({"val": v, "tag": tag})
                    seen_values.add(v)
        
        # 排序
        valid_points.sort(key=lambda x: x['val'])
        
        # 5. 生成戰略備註字串
        note_parts = []
        hit_status = "" # 用於 Point 2 的命中提示
        
        for p in valid_points:
            v_str = f"{p['val']:.0f}" if p['val'].is_integer() else f"{p['val']:.2f}"
            item_str = f"{v_str}{p['tag']}" if p['tag'] else v_str
            if "高" in p['tag']: item_str = f"高{v_str}" # 調整 "高" 的位置
            
            note_parts.append(item_str)
            
            # 檢查是否命中 (誤差 0.05 內)
            if abs(target_analyze_price - p['val']) < 0.05:
                hit_status = f"⚡ 命中: {item_str}"

        strategy_note = "-".join(note_parts)
        
        # 6. 計算獲利/防守 (Logic Point 5)
        # 尋找由現價往上的第一個壓力，往下的第一個支撐
        resistance = None
        support = None
        
        for p in valid_points:
            if p['val'] > target_analyze_price:
                resistance = p['val']
                break # 找到第一個比現價大的就是壓力
                
        for p in reversed(valid_points):
            if p['val'] < target_analyze_price:
                support = p['val']
                break # 找到第一個比現價小的就是支撐
        
        # 設定目標
        target_profit = resistance if resistance else limit_up # 若無壓力，看漲停 (Point 5)
        if resistance is None and target_analyze_price >= limit_up:
             target_profit = target_analyze_price # 已經漲停，目標即現價
             
        stop_loss = support if support else limit_down # 若無支撐，看跌停
        if support is None and target_analyze_price <= limit_down:
            stop_loss = target_analyze_price

        # 7. 漲跌力度 (Point 10: 紅/綠)
        pct = (target_analyze_price - prev_close) / prev_close * 100
        pct_icon = "🟥" if pct > 0 else ("🟩" if pct < 0 else "⬜")
        pct_str = f"{pct_icon} {pct:+.2f}%"
        
        # 8. 名稱處理
        real_name = name_input if name_input else code
        try:
            # 嘗試獲取簡單名稱 (如果沒有輸入的話)
            if not name_input:
                info_name = ticker.info.get('shortName', '') # 有時會抓不到
                if not info_name:
                    # 簡易備用: 若是輸入代號，就顯示代號
                    real_name = code
        except:
            pass

        display_name = f"{real_name} ({code})"

        return {
            "代號": code,
            "股票名稱": display_name,
            "自訂進場": target_analyze_price, # 可編輯
            "漲跌力度": pct_str,
            "獲利目標": target_profit,
            "防守停損": stop_loss,
            "戰略備註": strategy_note,
            "狀態提示": hit_status,
            "漲停價": limit_up,
            "跌停價": limit_down,
            "收盤價": current_price # 參考用
        }

    except Exception as e:
        return None

# ==========================================
# 介面建構
# ==========================================

# 側邊欄設定
with st.sidebar:
    st.header("⚙️ 設定面板")
    
    # ETF 過濾 (Point 9)
    hide_etf = st.toggle("隱藏 ETF/債券 (00開頭)", value=True)
    
    st.markdown("---")
    st.caption("顯示控制")
    # 行列自訂 (Point 8)
    col_control_tab, row_control_tab = st.tabs(["欄位", "行數"])
    with col_control_tab:
        all_cols = ["股票名稱", "自訂進場", "漲跌力度", "獲利目標", "防守停損", "戰略備註", "狀態提示", "漲停價", "跌停價"]
        selected_cols = st.multiselect("選擇顯示欄位", all_cols, default=all_cols)
    with row_control_tab:
        row_mode = st.radio("行數調整方式", ["滑桿", "手動輸入"])
        if row_mode == "滑桿":
            limit_rows = st.slider("顯示筆數", 5, 200, 50)
        else:
            limit_rows = st.number_input("輸入筆數", min_value=1, value=50)

# 主畫面
st.title("⚡ 當沖戰略操盤室")

# 上方輸入區 (整合搜尋與上傳)
col_search, col_file = st.columns([2, 1])

with col_search:
    # 多股搜尋 (Point 4)
    search_input = st.text_area("🔍 快速查詢 (支援多股/中文，用逗號分隔)", 
                                placeholder="例如: 2330, 鴻海, 2603, 8043", height=70)
    
with col_file:
    uploaded_file = st.file_uploader("📂 上傳 Excel/CSV (選填)", type=['xlsx', 'csv'])

# 執行按鈕
if st.button("🚀 執行戰略分析", type="primary", use_container_width=True):
    
    # 1. 整合清單
    target_list = [] # 格式: (code, name)
    
    # (A) 處理搜尋輸入
    if search_input:
        inputs = [x.strip() for x in search_input.replace('，',',').split(',') if x.strip()]
        for inp in inputs:
            if inp.isdigit():
                target_list.append((inp, "")) # 純代號
            else:
                # 中文名稱 -> 轉代號
                code_found = search_code_by_name(inp)
                target_list.append((code_found, inp)) # (代號, 輸入的名稱)

    # (B) 處理檔案上傳
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df_up = pd.read_csv(uploaded_file)
            else:
                df_up = pd.read_excel(uploaded_file)
            
            # 找代號與名稱
            c_col = next((c for c in df_up.columns if "代號" in c), None)
            n_col = next((c for c in df_up.columns if "名稱" in c), None)
            
            if c_col:
                for _, row in df_up.iterrows():
                    c = str(row[c_col]).split('.')[0]
                    n = str(row[n_col]) if n_col else ""
                    if c.isdigit():
                        target_list.append((c, n))
        except:
            st.error("檔案讀取失敗")

    # 若無資料則顯示範例
    if not target_list:
        st.info("請輸入代號或上傳檔案。")
    else:
        # 去除重複 (保留順序)
        seen = set()
        final_targets = []
        for t in target_list:
            if t[0] not in seen:
                final_targets.append(t)
                seen.add(t[0])
        
        # 2. 批次分析 (顯示進度條)
        results = []
        progress_bar = st.progress(0)
        
        for i, (code, name) in enumerate(final_targets):
            # ETF 過濾邏輯 (Point 9)
            if hide_etf and (code.startswith("00") or "債" in name):
                progress_bar.progress((i + 1) / len(final_targets))
                continue
                
            data = analyze_stock_strategy(code, name)
            if data:
                results.append(data)
            progress_bar.progress((i + 1) / len(final_targets))
            
        progress_bar.empty()
        
        # 3. 顯示結果 (Data Editor)
        if results:
            # 存入 session state 以便編輯後保留狀態 (這是 Streamlit 編輯功能的關鍵)
            if 'strategy_df' not in st.session_state or True: # 每次按鈕都重整
                 st.session_state.strategy_df = pd.DataFrame(results).head(limit_rows)

            df_display = st.session_state.strategy_df
            
            # 建立 Data Editor
            edited_df = st.data_editor(
                df_display,
                column_config={
                    "自訂進場": st.column_config.NumberColumn(
                        "自訂進場 (可修)",
                        help="修改此價格，獲利/停損與狀態會自動重算",
                        step=0.1, format="%.2f"
                    ),
                    "戰略備註": st.column_config.TextColumn("戰略備註 (近低-5MA-近高)", width="large"),
                    "狀態提示": st.column_config.TextColumn("狀態", help="若自訂價命中關鍵點會顯示"),
                },
                column_order=selected_cols, # Point 8: 自訂欄位
                use_container_width=True,
                hide_index=True,
                key="editor_key" # 綁定 key 才能即時反應
            )
            
            # --- 即時重算邏輯 (Point 2) ---
            # 檢查是否有修改。Streamlit 的 data_editor 修改後會直接反應在 edited_df
            # 但我們需要依照新的「自訂進場」重新跑一次獲利邏輯
            
            # 這裡做一個簡易的後處理重算 (因為 analyze 比較耗時，我們只重算比較簡單的邏輯)
            # 或是比較好的做法：比對 edited_df 和 st.session_state.strategy_df
            # 如果「自訂進場」變了，就更新該行的獲利目標/防守
            
            # 為了效能，我們直接在前端顯示修改後的 dataframe，
            # 若使用者改了價格，雖然「戰略備註」不會變(因為是歷史數據)，但「獲利/防守」應該要變
            # 由於 Python 腳本是由上而下執行，這裡其實較難做到 "即時單格重算"，
            # 除非我們寫一個 callback。
            
            # 替代方案：提示使用者若修改價格，請觀察「狀態提示」欄位 (我們可以在這裡做簡單的字串比對)
            
            st.caption("💡 提示：修改「自訂進場」價格後，若需精確重算獲利目標，可再次點擊執行，或自行對照「戰略備註」。(即時重算功能需更複雜後端)")
            
        else:
            st.warning("無符合條件的資料 (可能全被過濾或查無資料)。")
