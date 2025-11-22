import streamlit as st
import pandas as pd
import yfinance as yf
import requests
import plotly.graph_objects as go
import math
from datetime import datetime

# --- 1. 頁面全螢幕設定 ---
st.set_page_config(page_title="全方位戰略操盤室", page_icon="📈", layout="wide")

# --- CSS 美化設定 ---
st.markdown("""
    <style>
    .stApp { background-color: #1e1e1e; color: #e0e0e0; } /* 深色模式底 */
    .big-font { font-size: 20px !important; font-weight: bold; }
    .profit-text { color: #ff4b4b; font-weight: bold; }
    .loss-text { color: #00cc00; font-weight: bold; }
    .fib-table { width: 100%; text-align: center; border-collapse: collapse; }
    .fib-table td, .fib-table th { border: 1px solid #444; padding: 8px; }
    .fib-highlight { background-color: #333; color: yellow; font-weight: bold; border: 2px solid yellow !important;}
    .note-box { background-color: #2d2d2d; padding: 10px; border-radius: 5px; border-left: 5px solid #3498db; font-size: 0.9em; margin-top: 5px;}
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 區域 A: 輔助函式 (解析與運算)
# ==========================================

def parse_strategy_note(note_str):
    """
    解析戰略備註字串，例如: 63.3-65.2-67.2多-67.6-68-72-高74.4
    轉換為易讀的中文說明。
    """
    if not isinstance(note_str, str): return "無備註資料"
    
    parts = note_str.split('-')
    explanation = []
    
    try:
        # 嘗試依照使用者提供的邏輯進行對應
        # 假設格式相對固定，若長度不同則做通用處理
        for p in parts:
            p = p.strip()
            if "多" in p:
                val = p.replace("多", "")
                explanation.append(f"🔵 **5MA均線**: {val} (收盤 > {val} 為多)")
            elif "空" in p:
                val = p.replace("空", "")
                explanation.append(f"⚪ **5MA均線**: {val} (收盤 < {val} 為空)")
            elif "高" in p and p.startswith("高"): # 處理 '高74.4'
                val = p.replace("高", "")
                explanation.append(f"🛑 **近日高點**: {val}")
            elif "漲停" in p:
                explanation.append(f"🔥 **漲停價**: {p.replace('漲停', '')}")
            else:
                # 純數字部分，根據位置推測 (這部分比較模糊，依範例推測)
                # 範例順序: 近低1 - 近低2 - 5MA - 今低 - 今開 - 今高 - 近高
                # 這裡做簡單標示，避免誤判
                explanation.append(f"📍 **關鍵價位**: {p}")
        
        return "  \n".join(explanation)
    except:
        return f"原始備註: {note_str}"

@st.cache_data(ttl=300)
def get_market_chart_data(symbol, interval="1d"):
    """抓取大盤或期貨資料並計算費波那契"""
    try:
        # 對應代號: 加權指數 ^TWII, 台指期 TXF=F (Yahoo代號，可能延遲)
        ticker_map = {
            "加權指數": "^TWII",
            "台指期(近月)": "TXF=F" # Yahoo Finance 符號
        }
        code = ticker_map.get(symbol, "^TWII")
        
        # 處理週期格式
        period_map = {
            "1m": "1d", "5m": "5d", "15m": "5d", "60m": "1mo", "1d": "3mo"
        }
        p = period_map.get(interval, "1mo")
        
        data = yf.Ticker(code).history(period=p, interval=interval)
        if data.empty: return None, None
        
        # 取得計算基準的高低點 (依據畫面顯示的範圍)
        high_price = data['High'].max()
        low_price = data['Low'].min()
        diff = high_price - low_price
        
        # 費波那契係數
        fib_ratios = [-2.618, -2, -1.618, -1, 0, 0.236, 0.382, 0.5, 0.618, 0.764, 1, 1.618, 2, 2.618]
        fib_levels = {}
        
        # 計算價格 (預設 0=低點, 1=高點，這是順勢波段算法，也可反過來)
        # 這裡採用：0=Low, 1=High
        for r in fib_ratios:
            price = low_price + (diff * r)
            fib_levels[r] = price
            
        return data, fib_levels
    except Exception as e:
        st.error(f"抓取失敗: {e}")
        return None, None

@st.cache_data(ttl=86400)
def get_tw_stock_name(code):
    """抓取股票中文名稱"""
    try:
        url = f"https://tw.stock.yahoo.com/quote/{code}.TW"
        headers = {'User-Agent': 'Mozilla/5.0'}
        r = requests.get(url, headers=headers, timeout=3)
        if "<title>" in r.text:
            title = r.text.split('<title>')[1].split('</title>')[0]
            return title.split('(')[0].strip()
        return str(code)
    except:
        return str(code)

def calculate_prices(price):
    """計算漲跌停與3%"""
    try:
        price = float(price)
        # 簡易 Tick 規則
        tick = 5.0 if price >= 1000 else (1.0 if price >= 500 else (0.5 if price >= 100 else (0.1 if price >= 50 else (0.05 if price >= 10 else 0.01))))
        
        limit_up_raw = price * 1.10
        limit_down_raw = price * 0.90
        
        limit_up = math.floor(limit_up_raw / tick) * tick
        limit_down = math.ceil(limit_down_raw / tick) * tick # 跌停通常無條件進位至Tick避免超跌
        
        return {
            "漲停": round(limit_up, 2),
            "跌停": round(limit_down, 2),
            "+3%": round(price * 1.03, 2),
            "-3%": round(price * 0.97, 2)
        }
    except:
        return {}

# ==========================================
# 區域 B: 主介面邏輯
# ==========================================

# 建立分頁
tab_market, tab_strategy = st.tabs(["📊 盤勢 K 線與費波那契", "📋 個股戰略列表"])

# --- TAB 1: 盤勢 K 線圖 ---
with tab_market:
    st.subheader("即時大盤/期貨 K 線圖 (含費波那契)")
    
    col_m1, col_m2, col_m3 = st.columns([1, 1, 2])
    with col_m1:
        market_symbol = st.selectbox("選擇商品", ["加權指數", "台指期(近月)"])
    with col_m2:
        k_interval = st.selectbox("K線週期", ["1m", "5m", "15m", "60m", "1d"], index=1)
    
    if st.button("更新 K 線圖"):
        with st.spinner("正在計算費波那契數列..."):
            df_k, fibs = get_market_chart_data(market_symbol, k_interval)
            
            if df_k is not None:
                # 1. 繪製 K 線
                fig = go.Figure(data=[go.Candlestick(
                    x=df_k.index,
                    open=df_k['Open'], high=df_k['High'],
                    low=df_k['Low'], close=df_k['Close'],
                    name="K線"
                )])
                
                # 2. 繪製黃色費波那契線
                fib_display_data = []
                current_price = df_k['Close'].iloc[-1]
                
                for ratio, price in fibs.items():
                    # 畫線
                    fig.add_shape(type="line",
                        x0=df_k.index[0], y0=price, x1=df_k.index[-1], y1=price,
                        line=dict(color="yellow", width=1, dash="dash"),
                    )
                    # 標籤
                    fig.add_annotation(
                        x=df_k.index[-1], y=price,
                        text=f"{price:.1f}({ratio})",
                        showarrow=False, xanchor="left", font=dict(color="yellow")
                    )
                    
                    # 準備表格資料 (判斷是否為重要支撐壓力)
                    is_close = abs(current_price - price) / price < 0.005 # 距離 0.5% 內
                    status = "⚡ 測試中" if is_close else ""
                    fib_display_data.append({"比例": ratio, "點位": round(price, 1), "狀態": status})
                
                fig.update_layout(
                    xaxis_rangeslider_visible=False,
                    template="plotly_dark",
                    height=500,
                    margin=dict(l=0, r=50, t=30, b=30)
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # 3. 費波那契數值表
                st.markdown("### 🔢 費波那契關鍵點位表")
                # 轉為 DataFrame 並反轉順序 (高點在上面)
                df_fib = pd.DataFrame(fib_display_data).sort_values(by="比例", ascending=False)
                
                # 使用 HTML 渲染表格以達成高亮效果
                html_table = "<table class='fib-table'><tr><th>比例 (Ratio)</th><th>點位 (Price)</th><th>狀態</th></tr>"
                for _, row in df_fib.iterrows():
                    highlight_class = "fib-highlight" if row['狀態'] else ""
                    html_table += f"<tr class='{highlight_class}'><td>{row['比例']}</td><td>{row['點位']}</td><td>{row['狀態']}</td></tr>"
                html_table += "</table>"
                st.markdown(html_table, unsafe_allow_html=True)
                
            else:
                st.error("無法取得數據，可能是盤後資料源延遲，請稍後再試。")

# --- TAB 2: 個股戰略 ---
with tab_strategy:
    # 側邊欄控制區 (移到這裡讓它只影響這個 Tab 的感覺)
    with st.sidebar:
        st.header("📋 戰略表設定")
        
        # 1. 上傳與工作表選擇
        uploaded_file = st.file_uploader("上傳 Excel/CSV 檔", type=['xlsx', 'csv'])
        
        df_raw = None
        selected_sheet = "週轉率" # 預設
        
        if uploaded_file:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df_raw = pd.read_csv(uploaded_file)
                else:
                    # 讀取所有工作表名稱
                    xl = pd.ExcelFile(uploaded_file)
                    sheet_names = xl.sheet_names
                    
                    # 選單：預設選「週轉率」，若沒有則選第一個
                    default_idx = sheet_names.index("週轉率") if "週轉率" in sheet_names else 0
                    selected_sheet = st.selectbox("選擇工作表", sheet_names, index=default_idx)
                    
                    df_raw = pd.read_excel(uploaded_file, sheet_name=selected_sheet)
            except Exception as e:
                st.error(f"檔案讀取錯誤: {e}")
        
        # 2. 欄位與列數控制
        display_cols = []
        limit_rows = 100
        
        if df_raw is not None:
            all_cols = df_raw.columns.tolist()
            # 預設顯示所有欄位
            display_cols = st.multiselect("選擇要顯示的欄位", all_cols, default=all_cols)
            limit_rows = st.slider("顯示筆數", 5, len(df_raw), min(20, len(df_raw)))

    # 主要內容區
    st.subheader(f"戰略清單 ({selected_sheet})")
    
    if df_raw is not None:
        # 資料處理：擷取前 N 筆與選定欄位
        df_display = df_raw[display_cols].head(limit_rows)
        
        # 嘗試抓出代號與備註，用於生成互動視窗
        code_col = next((c for c in df_raw.columns if "代號" in c), None)
        note_col = next((c for c in df_raw.columns if "撐" in c or "備註" in c), None) # 模糊比對
        
        # 1. 顯示主表格
        st.dataframe(df_display, use_container_width=True, height=400)
        
        # 2. 圖示說明區
        st.info("""
        ℹ️ **表格圖示說明：**
        🔴 **多頭/獲利目標**：股價高於 5MA 或達到 +3% 獲利點。
        🟢 **空頭/防守停損**：股價低於 5MA 或跌破 -3% 防守點。
        ⚡ **黃色高亮 (K線圖)**：股價正處於費波那契關鍵支撐/壓力位。
        """)
        
        st.markdown("---")
        st.subheader("🔍 戰略備註解析 (點擊查看)")
        
        # 3. 互動解析區 (解決手機長按/滑鼠停留的需求)
        # 透過 Selectbox 選擇股票，下方顯示解析後的中文
        if code_col:
            # 製作選單清單: "8043 - 蜜望實"
            name_col = next((c for c in df_raw.columns if "名稱" in c), None)
            
            stock_options = []
            for idx, row in df_raw.iterrows():
                c = str(row[code_col]).split('.')[0]
                n = str(row[name_col]) if name_col else ""
                if c.isdigit():
                    stock_options.append(f"{c} {n}")
            
            selected_stock_str = st.selectbox("選擇股票查看詳細戰略解析", stock_options)
            
            if selected_stock_str:
                code = selected_stock_str.split(' ')[0]
                # 找出對應的那一行資料
                row_data = df_raw[df_raw[code_col].astype(str).str.contains(code)].iloc[0]
                
                col_d1, col_d2 = st.columns([1, 1])
                
                with col_d1:
                    st.markdown(f"### {selected_stock_str}")
                    # 抓即時股價計算
                    realtime_data = yf.Ticker(f"{code}.TW").history(period="1d")
                    if not realtime_data.empty:
                        now_price = realtime_data['Close'].iloc[-1]
                        calcs = calculate_prices(now_price)
                        st.metric("目前參考價", f"{now_price:.2f}")
                        st.write(f"🔥 漲停: **{calcs.get('漲停')}**")
                        st.write(f"📉 跌停: **{calcs.get('跌停')}**")
                        st.write(f"🎯 +3%: **{calcs.get('+3%')}**")
                        st.write(f"🛡️ -3%: **{calcs.get('-3%')}**")
                
                with col_d2:
                    st.markdown("### 📝 戰略備註解讀")
                    if note_col:
                        raw_note = str(row_data[note_col])
                        parsed_note = parse_strategy_note(raw_note)
                        
                        # 使用不同顏色區塊顯示
                        st.markdown(f"""
                        <div class="note-box">
                            <b>原始字串：</b><br>{raw_note}
                        </div>
                        <div style="margin-top:10px;">
                            <b>中文解析：</b><br>
                            {parsed_note}
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.warning("此檔案未包含「備註/撐壓」欄位，無法解析。")

    else:
        st.info("👋 請從左側側邊欄上傳您的 Excel 檔案以開始分析。")
