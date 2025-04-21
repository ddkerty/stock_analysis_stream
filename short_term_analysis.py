# short_term_analysis.py
# 📁 현재 기능: 피보나치 해석 interpret_fibonacci() 함수 1개

import pandas as pd

def interpret_fibonacci(df: pd.DataFrame, close_value: float = None) -> str | None:
    """
    피보나치 되돌림 수준을 기반으로 현재가가 주요 레벨 근처에 위치하는지 자동 해석합니다.
    :param df: 고가, 저가를 포함한 데이터프레임
    :param close_value: 현재 종가 또는 분석 기준 가격
    :return: 시그널 메시지 (또는 None)
    """
    if df.empty or close_value is None:
        return None

    try:
        min_price = df['Low'].min()
        max_price = df['High'].max()
        diff = max_price - min_price

        if diff <= 0:
            return "피보나치 분석 불가 (고가/저가 차이 없음)"

        levels = {
            '0.0 (High)': max_price,
            '0.236': max_price - 0.236 * diff,
            '0.382': max_price - 0.382 * diff,
            '0.5': max_price - 0.5 * diff,
            '0.618': max_price - 0.618 * diff,
            '1.0 (Low)': min_price
        }

        message = ""
        for key, value in levels.items():
            if abs(close_value - value) / diff < 0.02:  # 2% 이내 근접한 레벨 찾기
                message = f"🔍 **현재가가 피보나치 {key} 레벨 ({value:.2f})에 근접해 있습니다.**\n"
                # 설명 추가
                if key == "0.382":
                    message += "→ 일반적으로 **조정 후 지지 가능성**이 있는 구간입니다."
                elif key == "0.5":
                    message += "→ **중립적 전환점**으로 여겨지며 반등/하락 양쪽 가능성을 염두에 둬야 합니다."
                elif key == "0.618":
                    message += "→ 기술적으로 **되돌림의 마지막 지지선**으로 평가되며, 반등 기대 심리가 커질 수 있습니다."
                elif key == "0.236":
                    message += "→ 아직 **얕은 되돌림**으로 강한 추세가 지속될 수도 있습니다."
                elif key == "1.0":
                    message += "→ **최저점 테스트** 구간으로 하방 돌파 시 하락 추세 강화 우려."
                elif key == "0.0":
                    message += "→ **고점 근처**로 조정 가능성도 염두."
                break

        return message or "피보나치 주요 레벨 근처에 현재가가 위치해 있지 않습니다."

    except Exception as e:
        return f"⚠️ 피보나치 해석 오류: {str(e)}"
    
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



