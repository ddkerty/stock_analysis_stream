# technical_interpret.py

import numpy as np
import pandas as pd
from short_term_analysis import interpret_fibonacci

def interpret_technical_signals(latest_row):
    """
    VWAP과 Bollinger Band 기준으로 현재가(종가)의 위치를 해석합니다.
    """
    signal_messages = []

# VWAP 해석
    if 'VWAP' in latest_row and pd.notna(latest_row['VWAP']):
        if latest_row['Close'] > latest_row['VWAP']:
            signal_messages.append("📈 **현재가 > VWAP:** 단기 매수세 우위. *익절 고려 구간일 수 있습니다.*\n→ 특히 윗꼬리 음봉 + 거래량 감소 시 매도 검토")
        elif latest_row['Close'] < latest_row['VWAP']:
            signal_messages.append("📉 **현재가 < VWAP:** 단기 매도세 우위. *매수 진입 후보 구간일 수 있습니다.*\n→ 특히 하단 지지 + 거래량 증가 시 반등 기대 가능")
        else:
            signal_messages.append("↔️ **현재가 = VWAP:** 중립 영역. 시장 방향 탐색 중일 수 있습니다.")


    # Bollinger Band 해석
    if 'Upper' in latest_row and 'Lower' in latest_row and pd.notna(latest_row['Upper']) and pd.notna(latest_row['Lower']):
        close = latest_row['Close']
        upper = latest_row['Upper']
        lower = latest_row['Lower']
        band_width = upper - lower

        if close > upper:
            signal_messages.append("🚨 **> 볼린저 상단 돌파:** 단기 과열 가능성 또는 강한 상승 추세 지속")
        elif close < lower:
            signal_messages.append("💡 **< 볼린저 하단 이탈:** 단기 낙폭 과대 또는 하락 추세 지속 가능성")
        elif band_width > 0:
            position_ratio = (close - lower) / band_width
            if position_ratio > 0.75:
                signal_messages.append("🟢 **밴드 상단 근접:** 상승 압력 우세 (단기 강세 가능성)")
            elif position_ratio < 0.25:
                signal_messages.append("🔴 **밴드 하단 근접:** 하락 압력 우세 (단기 약세 가능성)")
            else:
                signal_messages.append("🟡 **밴드 중앙 위치:** 방향성 탐색 중 (중립적 흐름)")
        else:
            signal_messages.append("✅ **밴드 폭 0:** 방향성 부족 또는 박스권 흐름")

    return signal_messages

    # 피보나치 되돌림 해석
    fib_msg = interpret_fibonacci(df_calculated, close_value=latest_row['Close'])
    if fib_msg:
        signal_messages.append(fib_msg)
