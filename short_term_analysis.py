# short_term_analysis.py
# 📁 현재 기능: 피보나치 해석 interpret_fibonacci() 함수 1개

import pandas as pd
import numpy as np


def interpret_fibonacci(df: pd.DataFrame,
                        close_value: float | None = None,
                        prev_close: float | None = None) -> str | None:
    """
    피보나치 되돌림 수준별 시나리오·전략 제안 포함 해석
    - prev_close: 직전 봉 종가(있으면 ‘돌파/이탈’ 판단)
    """
    if df.empty or close_value is None:
        return None

    try:
        low, high = df['Low'].min(), df['High'].max()
        diff = high - low
        if diff <= 0:
            return "피보나치 분석 불가 (고가·저가 차이 없음)"

        # 레벨 값 계산
        levels = {
            0.0:  high,
            0.236: high - 0.236 * diff,
            0.382: high - 0.382 * diff,
            0.5:   high - 0.5   * diff,
            0.618: high - 0.618 * diff,
            1.0:  low,
        }

        # 가까운 레벨 찾기 (±1.5 % 이내)
        nearest = min(levels.items(),
                      key=lambda kv: abs(close_value - kv[1]))
        ratio, lvl_price = nearest
        if abs(close_value - lvl_price) / diff > 0.015:
            return "현재가는 주요 피보나치 레벨에서 멀리 떨어져 있어요."

        # 레벨별 시나리오/전략
        comments = {
            0.236: ("얕은 되돌림 후 강세 재개 가능성이 커 보입니다.",
                    "전 고점 돌파 시 추세추종 매수 고려"),
            0.382: ("첫 번째 핵심 지지선입니다.",
                    "하향 돌파 시 0.5까지 눌림 가능성을 염두에 두세요."),
            0.5:   ("추세가 중립으로 전환되는 분기점이에요.",
                    "방향 확인 전까지 관망 또는 포지션 축소가 안전합니다."),
            0.618: ("되돌림의 마지막 보루로 평가됩니다.",
                    "반등 캔들 + 거래량 증가 시 진입, 반대로 종가 이탈 시 손절 고려"),
            1.0:   ("저점을 다시 시험 중입니다.",
                    "지지 실패 시 하락 추세 강화, 성공 시 쌍바닥 반등 시도 가능"),
            0.0:   ("고점 부근이며 차익 실현 압력이 커질 수 있어요.",
                    "음봉 전환·거래량 감소 확인 시 익절 분할 매도 고려"),
        }

        # 이전 봉 대비 돌파/이탈 감지
        breach_msg = ""
        if prev_close is not None:
            if prev_close < lvl_price <= close_value:
                breach_msg = "▶ **상향 돌파** 신호가 나왔습니다."
            elif prev_close > lvl_price >= close_value:
                breach_msg = "▶ **하향 이탈** 신호가 나왔습니다."

        headline = (f"🔍 **현재가가 피보나치 {ratio:.3f}"
                    f" 레벨({lvl_price:.2f}$) 근처입니다.**")
        body, strategy = comments.get(ratio, ("", ""))
        msg = f"{headline}\n- {body}\n- {strategy}"
        if breach_msg:
            msg += f"\n{breach_msg}"
        return msg

    except Exception as e:
        return f"⚠️ 피보나치 해석 오류: {e}"

    
# 추가 기능: RSI(Relative Strength Index) 계산
def calculate_rsi(df, period: int = 14) -> pd.DataFrame:
    """RSI(Relative Strength Index) 계산"""
    df = df.copy()
    if 'Close' not in df.columns:
        raise ValueError("RSI 계산을 위해 'Close' 컬럼이 필요합니다.")
    
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    return df
# 추가 기능: MACD(Moving Average Convergence Divergence) 계산
def calculate_macd(df, fast=12, slow=26, signal=9):
    """
    MACD 및 Signal Line을 계산하고 DataFrame에 컬럼으로 추가합니다.
    """
    df = df.copy()
    if 'Close' not in df.columns:
        raise ValueError("MACD 계산에 'Close' 컬럼이 필요합니다.")
    
    df['EMA_fast'] = df['Close'].ewm(span=fast, adjust=False).mean()
    df['EMA_slow'] = df['Close'].ewm(span=slow, adjust=False).mean()
    df['MACD'] = df['EMA_fast'] - df['EMA_slow']
    df['MACD_signal'] = df['MACD'].ewm(span=signal, adjust=False).mean()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']
    
    return df

def calculate_rsi(df, period=14):
    delta = df['Close'].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    avg_gain = pd.Series(gain).rolling(window=period).mean()
    avg_loss = pd.Series(loss).rolling(window=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    df['RSI'] = rsi
    return df

def calculate_macd(df, fast=12, slow=26, signal=9):
    ema_fast = df['Close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - macd_signal

    df['MACD'] = macd
    df['MACD_signal'] = macd_signal
    df['MACD_hist'] = macd_hist
    return df



