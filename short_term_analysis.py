from pathlib import Path

# 완전한 short_term_analysis.py 내용
short_term_analysis_code = '''\
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

# 파일로 저장
#file_path = Path("/mnt/data/short_term_analysis.py")
#file_path.write_text(short_term_analysis_code, encoding='utf-8')
#file_path
