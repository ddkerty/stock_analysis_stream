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

        for label, level_price in levels.items():
            if abs(close_value - level_price) / level_price < 0.005:
                return (
                    f"📐 **피보나치 되돌림**\n"
                    f"🔎 현재가가 **Fib {label} (${level_price:.2f})** 근처에 위치합니다.\n"
                    f"→ 이는 되돌림 지지/저항선으로 작용할 가능성이 있으며, 기술적 반응이 나타날 수 있습니다."
                )

        return "현재가는 피보나치 주요 레벨에서 의미 있는 거리만큼 떨어져 있습니다."

    except Exception as e:
        return f"❌ 피보나치 해석 오류: {e}"
