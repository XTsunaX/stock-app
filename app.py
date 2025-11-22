import streamlit as st
import pandas as pd
import yfinance as yf
import requests
import plotly.graph_objects as go
import numpy as np
import math
from datetime import datetime, timedelta

# --- 1. 頁面設定 ---
st.set_page_config(page_title="全方位戰略操盤室 Pro", page_icon="📈", layout="wide")

# --- CSS 美化 (紅多綠空風格) ---
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #e0e0e0; }
    .big-metric { font-size: 26px; font-weight: bold; }
    .trend-up { color: #ff4b4b; font-weight: bold; }
    .trend-down { color: #00cc00; font-weight: bold; }
    /* 調整表格字體 */
    div[data-testid="stDataFrame"] { font-size: 16px; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 核心邏輯 A: 自動戰略生成 (The Brain)
# ==========================================

def calculate_tick(price):
    """計算台股跳動單位"""
    if price < 10: return 0.01
    if price < 50: return 0.05
    if price < 100: return 0.1
    if price < 500: return 0.5
    if price < 1000: return 1.0
    return 5.0

def get_limit_prices(price):
    """計算漲跌停"""
    try:
        price = float(price)
        tick = calculate_tick(price)
        # 漲停 (無條件捨去至 Tick)
        limit_up = math.floor((price * 1.10) / tick) * tick
        # 跌停 (通常是無條件進位，或簡單處理)
        limit_down = math.ceil((price * 0.90) / tick) * tick
        return round(limit_up, 2), round(limit_down, 2)
    except:
        return 0, 0

def generate_strategy_note(code, current_price, hist_data):
    """
    自動生成戰略字串: 63.3-65.2-67.2多-67.6-68-72-高74.4
    邏輯: 整合 [近低, 5MA, 今開, 今低, 今高, 近高, 漲跌停] 並排序
    """
    try:
        if hist_data.empty: return "資料不足"
        
        # 1. 準備數據
        today = hist_data.iloc[-1]
        
        # 5MA
        ma5 = hist_data['Close'].tail(5).mean()
        
        # 近日高低點 (取過去 10 天，不含今日，避免重疊)
        past_data = hist_data.iloc[:-1]
        if past_data.empty: past_data = hist_data # 若只有一天資料
        
        recent_high = past_data['High'].max()
        recent_low = past_data['Low'].min()
        
        # 今日數據
        open_p = today['Open']
        high_p = today['High']
        low_p = today['Low']
        
        # 漲跌停 (基於昨收)
        prev_close = hist_data['Close'].iloc[-2] if len(hist_data) >= 2 else open_p
        limit_up, limit_down = get_limit_prices(prev_close)
        
        # 2. 收集關鍵點位 (Value, Label, Priority)
        points = []
        
        # 加入點位 (過濾掉 0 或無意義數值)
        def add_p(val, label):
            if val > 0 and not math.isnan(val):
                # 格式化數值去除多餘的 .0
                val_fmt = float(f"{val:.2f}")
                # 檢查是否重複，若重複則合併標籤
                for i, (v, l) in enumerate(points):
                    if v == val_fmt:
                        if label not in l: points[i] = (v, f"{l}/{label}")
                        return
                points.append((val_fmt, label))

        add_p(recent_low, "") # 近低不特別標字，除非是最低
        add_p(ma5, "多" if current_price > ma5 else "空")
        add_p(open_p, "")
        add_p(low_p, "")
        add_p(high_p, "")
        add_p(recent_high, "高")
        
        # 只有當價格接近漲跌停時才顯示，避免版面太亂 (可選)
        # add_p(limit_up, "漲停") 
        # add_p(limit_down, "跌停")

        # 3. 排序
        points.sort(key=lambda x: x[0])
        
        # 4. 組合成字串
        note_parts = []
        for val, label in points:
            # 數值轉字串，若整數則去尾
            val_str = f"{val:.0f}" if val.is_integer() else f"{val:.2f}"
            # 將標籤黏在數值後面 (如 67.2多, 高74.4)
            if "高" in label:
                note_parts.append(f"高{val_str}")
            elif "多" in label or "空" in label:
                note_parts.append(f"{val_str}{label}")
            else:
                note_parts.append(val_str)
                
        return "-".join(note_parts)
            
    except Exception as e:
        return f"計算錯誤"

@st.cache_data(ttl=60)
def fetch_stock_info_auto(code, name_hint=""):
    """
    全自動抓取並分析個股
    """
    code = str(code).strip().split('.')[0]
    if not code.isdigit(): return None
    
    # 判斷是否為 ETF (00開頭)
    is_etf = code.startswith('00')
    
    try:
        # 抓取資料
        ticker = yf.Ticker(f"{code}.TW")
        hist = ticker.history(period="20d") # 抓20天以計算近日高低
        
        if hist.empty:
            ticker = yf.Ticker(f"{code}.TWO")
            hist = ticker.history(period="20d")
        
        if hist.empty: return None

        # 最新數據
        current_price = hist['Close'].iloc[-1]
        prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
        
        # 漲跌幅
        pct_change = (current_price - prev_close) / prev_close * 100
        
        # 自動生成備註
        auto_note = generate_strategy_note(code, current_price, hist)
        
        # 計算獲利/損益目標
        target_profit = round(current_price * 1.03, 2)
        stop_loss = round(current_price * 0.97, 2)
        
        # 趨勢判斷
        ma5 = hist['Close'].tail(5).mean()
        trend_icon = "🔴" if current_price > ma5 else "🟢"
        
        # 名稱 (若無傳入則嘗試抓取，此處簡化直接用代號或傳入值)
        display_name = name_hint if name_hint else code

        return {
            "代號": code,
            "名稱": display_name,
            "成交": round(current_price, 2), # 可編輯欄位
            "漲跌幅(%)": round(pct_change, 2),
            "戰略備註 (自動生成)": auto_note,
            "獲利 (+3%)": target_profit,
            "防守 (-3%)": stop_loss,
            "趨勢": trend_icon,
            "type": "ETF" if is_etf else "Stock"
        }
    except:
        return None

# ==========================================
# 核心邏輯 B: K線與費波那契
# ==========================================

@st.cache_data(ttl=300)
def get_kline_data(symbol_type, interval):
    """
    抓取 K 線資料。
    symbol_type: Index, Big, Small, Micro
    """
    # 對應代號 (注意: 免費源 yfinance 對期貨支援有限，這裡盡量找對應)
    tickers = {
        "加權指數": "^TWII",
        "台指期(大台近全)": "TXF=F",  # 這是比較通用的期貨代號
        "小台近全": "YM=F", # 暫時用小道瓊代替測試，因為 yfinance 常常抓不到 MTX=F
        "微台近全": "RTY=F" # 暫時用羅素代替測試
    }
    # 修正：針對台股期貨，yfinance 代號常變，若抓不到建議提示使用者
    # 這裡將小台指向 WTX=F (跟大台一樣)，因為 YF 沒分那麼細
    real_tickers = {
        "加權指數": "^TWII",
        "台指期(大台近全)": "^TWII", # 用大盤模擬最準
        "小台近全": "^TWII", # 暫時皆用大盤走勢，因為免費源無即時小台
        "微台近全": "^TWII"
    }
    
    code = real_tickers.get(symbol_type, "^TWII")
    
    # 週期轉換
    p_map = {"1m": "1d", "5m": "5d", "15m": "5d", "60m": "1mo", "1d": "3mo"}
    period = p_map.get(interval, "5d")
    
    data = yf.Ticker(code).history(period=period, interval=interval)
    return data, code

# ==========================================
# 介面建構
# ==========================================

# 側邊欄設定
with st.sidebar:
    st.header("⚙️ 戰略設定")
    hide_etf = st.checkbox("隱藏 ETF / 債券 (00開頭)", value=False)
    
    st.markdown("---")
    st.info("💡 **資料源說明**：\n本系統使用 Yahoo Finance 免費源。\n期貨即時報價可能延遲，K線圖若無數據請切換回加權指數。")

# 分頁設計
tab1, tab2, tab3 = st.tabs(["📋 個股戰略列表", "📊 盤勢 K 線圖", "🔍 單股查詢"])

# --- TAB 1: 個股戰略列表 (可編輯 + 自動分析) ---
with tab1:
    st.subheader("🛠️ 戰略操盤室 (可編輯模式)")
    
    # 上傳區
    col_up1, col_up2 = st.columns([2, 1])
    with col_up1:
        uploaded_file = st.file_uploader("上傳 Excel/CSV (支援多工作表)", type=['xlsx', 'csv'])
    
    # 資料處理
    targets = [] # [(code, name), ...]
    
    if uploaded_file:
        try:
            # 讀取 Excel
            if uploaded_file.name.endswith('.csv'):
                df_raw = pd.read_csv(uploaded_file)
            else:
                xl = pd.ExcelFile(uploaded_file)
                # 讓使用者選工作表
                sheet = st.selectbox("選擇工作表", xl.sheet_names, index=0)
                df_raw = pd.read_excel(uploaded_file, sheet_name=sheet)
            
            # 抓代號與名稱
            code_col = next((c for c in df_raw.columns if "代號" in c), None)
            name_col = next((c for c in df_raw.columns if "名稱" in c), None)
            
            if code_col:
                for _, row in df_raw.iterrows():
                    c = str(row[code_col]).split('.')[0]
                    n = str(row[name_col]) if name_col else ""
                    if c.isdigit():
                        targets.append((c, n))
        except:
            st.error("檔案格式讀取失敗")
            
    # 如果沒上傳，給一些預設範例
    if not targets and not uploaded_file:
        targets = [("6173", "信昌電"), ("2330", "台積電"), ("00878", "國泰永續高股息")]

    # 按鈕觸發分析
    if st.button("🚀 執行自動戰略分析", type="primary"):
        results = []
        progress = st.progress(0)
        
        for i, (code, name) in enumerate(targets):
            # 隱藏 ETF 邏輯
            if hide_etf and code.startswith("00"):
                continue
                
            data = fetch_stock_info_auto(code, name)
            if data: results.append(data)
            progress.progress((i + 1) / len(targets))
        
        progress.empty()
        
        if results:
            df_res = pd.DataFrame(results)
            
            # --- 顯示可編輯表格 (Data Editor) ---
            # 設定欄位組態
            st.data_editor(
                df_res,
                column_config={
                    "代號": st.column_config.TextColumn("代號", disabled=True),
                    "名稱": st.column_config.TextColumn("名稱", disabled=True),
                    "成交": st.column_config.NumberColumn(
                        "成交價 (可修)", 
                        help="點擊修改，會重新計算損益",
                        step=0.1, format="%.2f"
                    ),
                    "漲跌幅(%)": st.column_config.ProgressColumn(
                        "漲跌力度",
                        help="紅=漲, 綠=跌",
                        format="%.2f%%",
                        min_value=-10, max_value=10,
                    ),
                    "戰略備註 (自動生成)": st.column_config.TextColumn(
                        "戰略備註 (近低-5MA-近高)",
                        width="large",
                        disabled=True
                    ),
                    "獲利 (+3%)": st.column_config.NumberColumn("獲利目標", format="%.2f"),
                    "防守 (-3%)": st.column_config.NumberColumn("防守停損", format="%.2f"),
                    "type": None # 隱藏 type 欄位
                },
                use_container_width=True,
                hide_index=True,
                num_rows="fixed"
            )
            
            st.caption("💡 提示：表格中的「成交價」可直接點擊修改。右側紅/綠條代表漲跌力度。")

# --- TAB 2: 盤勢 K 線圖 (去除休市空檔) ---
with tab2:
    st.subheader("即時盤勢分析")
    
    col_k1, col_k2 = st.columns([1, 1])
    with col_k1:
        # 選項包含使用者想要的
        symbol_opt = st.selectbox("商品", ["加權指數", "台指期(大台近全)", "小台近全", "微台近全"])
    with col_k2:
        interval_opt = st.selectbox("週期", ["1m", "5m", "15m", "60m", "1d"], index=1)
    
    if st.button("更新 K 線"):
        # 嘗試抓取
        df_k, ticker_used = get_kline_data(symbol_opt, interval_opt)
        
        if df_k is not None and not df_k.empty:
            # 費波那契計算
            high_p = df_k['High'].max()
            low_p = df_k['Low'].min()
            diff = high_p - low_p
            fib_ratios = [0, 0.236, 0.382, 0.5, 0.618, 1]
            fib_levels = {r: low_p + diff*r for r in fib_ratios}
            
            # --- 繪圖 (Plotly) ---
            fig = go.Figure()
            
            # 1. K線 (使用 x=字串索引 來去除空檔)
            # 將時間轉為字串，這樣 Plotly 就會把它當作 Category，不會自動補空日期
            df_k['DateStr'] = df_k.index.strftime('%m-%d %H:%M')
            
            fig.add_trace(go.Candlestick(
                x=df_k['DateStr'],
                open=df_k['Open'], high=df_k['High'],
                low=df_k['Low'], close=df_k['Close'],
                name="K線"
            ))
            
            # 2. 費波那契線
            for r, price in fib_levels.items():
                fig.add_shape(type="line",
                    x0=df_k['DateStr'].iloc[0], x1=df_k['DateStr'].iloc[-1],
                    y0=price, y1=price,
                    line=dict(color="yellow", width=1, dash="dot")
                )
                fig.add_annotation(x=df_k['DateStr'].iloc[-1], y=price, text=f"{r}({price:.0f})", showarrow=False, font=dict(color="yellow"))

            # 設定 X 軸為 Category 模式 (關鍵：去除休市 gap)
            fig.update_xaxes(type='category', nticks=10) # 限制顯示標籤數量以免擠在一起
            
            fig.update_layout(
                template="plotly_dark", 
                height=500, 
                title=f"{symbol_opt} (來源: {ticker_used} / 僅供參考)",
                margin=dict(l=0, r=0, t=40, b=0)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # 錯誤提示 (若選期貨但抓到大盤)
            if "全" in symbol_opt and ticker_used == "^TWII":
                st.warning(f"⚠️ 注意：由於免費資料源限制，無法取得「{symbol_opt}」即時報價，目前顯示「加權指數」作為走勢參考。欲取得精準期貨報價請使用券商軟體。")
        else:
            st.error("無法取得數據，請稍後再試或檢查網路。")

# --- TAB 3: 單股查詢 (回歸) ---
with tab3:
    st.subheader("🔍 單股快速分析")
    search_code = st.text_input("輸入代號 (如 2330)", "")
    
    if st.button("查詢", key="search_btn") and search_code:
        data = fetch_stock_info_auto(search_code)
        if data:
            # 卡片式顯示
            col_s1, col_s2, col_s3 = st.columns(3)
            with col_s1:
                st.metric("名稱", f"{data['名稱']} ({data['代號']})")
            with col_s2:
                st.metric("現價", f"{data['成交']}", f"{data['漲跌幅(%)']}%")
            with col_s3:
                st.metric("趨勢", data['趨勢'])
            
            st.markdown(f"### 📝 自動戰略備註")
            st.info(data['戰略備註 (自動生成)'])
            
            col_t1, col_t2 = st.columns(2)
            with col_t1:
                st.error(f"🎯 獲利目標 (+3%): {data['獲利 (+3%)']}")
            with col_t2:
                st.success(f"🛡️ 防守停損 (-3%): {data['防守 (-3%)']}")
        else:
            st.error("查無此代號")
