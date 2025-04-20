from pathlib import Path

# 수정된 VWAP 및 피보나치 분석 탭 포함한 전체 app.py 코드 생성
corrected_app_code = """
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

# VWAP 계산 함수
def calculate_vwap(df):
    df = df.copy()
    if not all(col in df.columns for col in ['High', 'Low', 'Close', 'Volume']):
        raise ValueError("VWAP 계산에 필요한 컬럼이 없습니다.")
    df['typical_price'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['tp_volume'] = df['typical_price'] * df['Volume']
    df['VWAP'] = df['tp_volume'].cumsum() / df['Volume'].cumsum()
    return df['VWAP']

# 피보나치 차트 함수
def plot_fibonacci_levels(df, ticker):
    fig = go.Figure()

    valid_df = df.dropna(subset=['High', 'Low'])
    if valid_df.empty:
        st.warning("차트를 그릴 유효한 가격 데이터가 없습니다.")
        return fig

    max_price = valid_df['High'].max()
    min_price = valid_df['Low'].min()
    diff = max_price - min_price

    levels = {
        '1.0 (최저)': min_price,
        '0.618': min_price + 0.382 * diff,
        '0.5': min_price + 0.5 * diff,
        '0.382': min_price + 0.618 * diff,
        '0.236': min_price + 0.764 * diff,
        '0.0 (최고)': max_price
    }
    colors = {'0.0 (최고)': 'red', '0.236': 'orange', '0.382': 'gold', '0.5': 'green', '0.618': 'blue', '1.0 (최저)': 'purple'}

    for key, value in levels.items():
        fig.add_hline(y=value, line_dash="dot", line_color=colors.get(key, 'grey'),
                      annotation_text=f"Fib {key}: ${value:.2f}",
                      annotation_position="top right",
                      annotation_font_size=10)

    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name=f"{ticker} 캔들스틱"
    ))

    if 'VWAP' in df.columns and not df['VWAP'].isnull().all():
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df["VWAP"],
            mode="lines",
            line=dict(color="darkorange", width=1.5),
            name="VWAP"
        ))

    fig.update_layout(
        title=f"{ticker} 피보나치 되돌림 + VWAP 차트",
        xaxis_title="날짜 / 시간",
        yaxis_title="가격 ($)",
        xaxis_rangeslider_visible=False,
        legend_title_text="지표",
        hovermode="x unified",
        margin=dict(l=50, r=50, t=50, b=50)
    )
    return fig

# Streamlit 앱 시작
st.set_page_config(layout="wide", page_title="단기 기술 분석 탭")

st.title("📈 단기 기술 분석 - VWAP & 피보나치")

ticker = st.text_input("종목 티커를 입력하세요 (예: AAPL, TSLA)", "AAPL")
start_date = st.date_input("시작일", datetime(2024, 1, 1))
end_date = st.date_input("종료일", datetime.today())
interval = st.selectbox("데이터 간격", options=["1d", "1h", "15m"], index=0)

if st.button("데이터 불러오기 및 분석"):
    df = yf.download(ticker, start=start_date, end=end_date, interval=interval)

    if df.empty:
        st.error("❌ 데이터를 불러오지 못했습니다. 티커나 날짜를 확인해주세요.")
    elif not all(col in df.columns for col in ['High', 'Low', 'Close', 'Volume', 'Open']):
        st.error(f"❌ 필수 컬럼이 없습니다. 받은 컬럼: {df.columns.tolist()}")
        st.dataframe(df.head())
    else:
        try:
            df["VWAP"] = calculate_vwap(df)
            st.subheader("📌 VWAP + 피보나치 차트")
            st.plotly_chart(plot_fibonacci_levels(df, ticker), use_container_width=True)
            st.dataframe(df.tail(10), use_container_width=True)
        except Exception as e:
            st.error(f"차트 생성 또는 VWAP 계산 중 오류 발생: {e}")
"""

Path("/mnt/data/app_vwap_fib_corrected.py").write_text(corrected_app_code)

import pandas as pd
import ace_tools as tools

tools.display_dataframe_to_user(name="Streamlit 통합 앱 (VWAP + 피보나치) 정정 버전", dataframe=pd.DataFrame({
    "파일명": ["app_vwap_fib_corrected.py"],
    "설명": ["VWAP 계산 오류 방지 로직 및 피보나치 차트 포함 전체 앱 코드"]
}))
