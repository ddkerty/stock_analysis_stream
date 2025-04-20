# 코드 실행 환경이 초기화되어 다시 파일을 저장합니다.

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
'''

# 2. app_fibonacci.py (Streamlit 앱용 코드)
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

# 파일 저장
from pathlib import Path

Path("/mnt/data/short_term_analysis.py").write_text(short_term_analysis_code, encoding='utf-8')
Path("/mnt/data/app_fibonacci.py").write_text(app_code, encoding='utf-8')
