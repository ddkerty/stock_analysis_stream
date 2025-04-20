# 필요한 파일 구조에 맞춰 피보나치 되돌림 기능을 단기 기술 분석용 모듈로 분리하고,
# app.py에서 호출할 수 있도록 전체 코드를 구성하겠습니다.

# 1. short_term_analysis.py (신규 모듈 생성용 코드)
short_term_analysis_code = '''
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

def download_price_data(ticker: str, period: str = "1mo", interval: str = "1d") -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval=interval)
    df.dropna(inplace=True)
    return df

def calculate_fibonacci_retracement(df: pd.DataFrame):
    max_price = df["High"].max()
    min_price = df["Low"].min()
    diff = max_price - min_price
    levels = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
    retracement_levels = [max_price - diff * level for level in levels]
    return levels, retracement_levels

def plot_fibonacci_chart(df: pd.DataFrame):
    levels, retracements = calculate_fibonacci_retracement(df)
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name="Candle"
    ))

    for i, level in enumerate(levels):
        fig.add_hline(
            y=retracements[i],
            line_dash="dash",
            annotation_text=f"{int(level*100)}%",
            annotation_position="right"
        )

    fig.update_layout(title="Fibonacci Retracement Chart", height=600)
    return fig

def interpret_fibonacci(df, close_value=None):
    """피보나치 레벨 해석"""
    if 'High' not in df.columns or 'Low' not in df.columns or df.empty:
        return "📏 피보나치 해석 불가 (데이터 부족)"

    high_price = df['High'].max()
    low_price = df['Low'].min()
    diff = high_price - low_price

    if diff <= 0:
        return "📏 피보나치 해석 불가 (고점=저점)"

    fib_levels = {
        "0.236": high_price - 0.236 * diff,
        "0.382": high_price - 0.382 * diff,
        "0.5": high_price - 0.5 * diff,
        "0.618": high_price - 0.618 * diff,
    }

    close = close_value if close_value is not None else df['Close'].iloc[-1]
    closest_level, level_value = min(fib_levels.items(), key=lambda x: abs(close - x[1]))

    explanation = {
        "0.236": "약한 되돌림 → 강한 추세 지속 가능성",
        "0.382": "일반 되돌림 → 단기 저항 가능성",
        "0.5": "심리적 중간선 → 방향성 탐색 구간",
        "0.618": "강한 되돌림 → 반등 또는 지지 시도 주시"
    }.get(closest_level, "")

    return f"📏 **현재가 (${close:.2f})는 Fib {closest_level} (${level_value:.2f}) 근처** → {explanation}"
'''

# 2. app.py 수정용 샘플 코드 (Fibonacci 탭 추가)
app_code = '''
import streamlit as st
from short_term_analysis import download_price_data, plot_fibonacci_chart

st.set_page_config(page_title="TechnutStock", layout="wide")

st.title("📊 TechnutStock - 단기 기술 분석")

with st.sidebar:
    st.header("단기 기술 분석 설정")
    ticker = st.text_input("종목 티커 입력", value="AAPL")
    period = st.selectbox("조회 기간", ["5d", "1mo", "3mo"], index=1)
    interval = st.selectbox("인터벌", ["1m", "5m", "15m", "1d"], index=3)

tab1, tab2 = st.tabs(["📈 피보나치 되돌림", "🛠 향후 추가 기능"])

with tab1:
    st.subheader("📉 Fibonacci Retracement 차트")
    if st.button("차트 분석 실행"):
        df = download_price_data(ticker, period=period, interval=interval)
        if df is not None and not df.empty:
            fig = plot_fibonacci_chart(df)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("데이터를 불러오지 못했습니다.")
with tab2:
    st.info("VWAP, Bollinger Bands 등의 추가 기술 지표 분석 기능은 추후 업데이트 예정입니다.")
'''

# 저장
from pathlib import Path

Path("/mnt/data/short_term_analysis.py").write_text(short_term_analysis_code, encoding='utf-8')
Path("/mnt/data/app_fibonacci.py").write_text(app_code, encoding='utf-8')
