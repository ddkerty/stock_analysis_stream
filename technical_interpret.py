# technical_interpret.py

import numpy as np
import pandas as pd
from short_term_analysis import interpret_fibonacci

def interpret_technical_signals(row, df_context=None):
    """
    VWAP, Bollinger Band, RSI, MACD, 피보나치 기준 자동 해석
    row: DataFrame의 한 줄 (latest_row)
    df_context: 전체 df_calculated (피보나치 등 전체 분석 필요 시)
    """
    signals = []

    # 📊 VWAP 해석
    if 'VWAP' in row and pd.notna(row['VWAP']):
        if row['Close'] > row['VWAP']:
            signals.append("📈 **현재가 > VWAP:** 단기 매수세 우위. *익절 고려 구간일 수 있습니다.*\n→ 특히 윗꼬리 음봉 + 거래량 감소 시 매도 검토")
        elif row['Close'] < row['VWAP']:
            signals.append("📉 **현재가 < VWAP:** 단기 매도세 우위. *매수 진입 후보 구간일 수 있습니다.*\n→ 특히 하단 지지 + 거래량 증가 시 반등 기대 가능")
        else:
            signals.append("↔️ **현재가 = VWAP:** 중립 영역. 시장 방향 탐색 중일 수 있습니다.")

    # 📊 Bollinger Band 해석
    if all(col in row for col in ['Upper', 'Lower']) and pd.notna(row['Upper']) and pd.notna(row['Lower']):
        close, upper, lower = row['Close'], row['Upper'], row['Lower']
        band_width = upper - lower
        if close > upper:
            signals.append("🚨 **> 볼린저 상단 돌파:** 단기 과열 가능성 또는 강한 상승 추세 지속")
        elif close < lower:
            signals.append("💡 **< 볼린저 하단 이탈:** 단기 낙폭 과대 또는 하락 추세 지속 가능성")
        elif band_width > 0:
            position_ratio = (close - lower) / band_width
            if position_ratio > 0.75:
                signals.append("🟢 **밴드 상단 근접:** 상승 압력 우세 (단기 강세 가능성)")
            elif position_ratio < 0.25:
                signals.append("🔴 **밴드 하단 근접:** 하락 압력 우세 (단기 약세 가능성)")
            else:
                signals.append("🟡 **밴드 중앙 위치:** 방향성 탐색 중 (중립적 흐름)")
        else:
            signals.append("✅ **밴드 폭 0:** 방향성 부족 또는 박스권 흐름")

    # 📊 RSI 해석
    if 'RSI' in row and pd.notna(row['RSI']):
        rsi = row['RSI']
        if rsi > 70:
            signals.append("📈 **RSI > 70:** 과매수 영역 진입 → 단기 조정 가능성 고려")
        elif rsi < 30:
            signals.append("📉 **RSI < 30:** 과매도 영역 진입 → 단기 반등 가능성 고려")
        elif 50 < rsi <= 70:
            signals.append("🟢 **RSI 중상단(50~70):** 상승 추세 유지 가능성")
        elif 30 <= rsi < 50:
            signals.append("🔴 **RSI 중하단(30~50):** 하락 압력 우세 가능성")

    # 📊 MACD 해석
    if 'MACD' in row and 'MACD_signal' in row and pd.notna(row['MACD']) and pd.notna(row['MACD_signal']):
        macd = row['MACD']
        signal_line = row['MACD_signal']
        hist = row.get('MACD_hist', macd - signal_line)
        if macd > signal_line and hist > 0:
            signals.append("🟢 **MACD 골든크로스:** 상승 모멘텀 형성 (추세 반전 가능성)")
        elif macd < signal_line and hist < 0:
            signals.append("🔴 **MACD 데드크로스:** 하락 모멘텀 증가 (추세 약화 우려)")
        elif abs(hist) < 0.1:
            signals.append("⚪️ **MACD 신호선 수렴:** 방향성 모호, 추세 전환 전 조정 구간일 수 있음")

    # 📊 피보나치 되돌림 해석
    if df_context is not None and len(df_context) >= 2:
        fib_msg = interpret_fibonacci(
            df_context,
            close_value=row['Close'],
            prev_close=df_context['Close'].iloc[-2]  # 직전 봉
        )
        if fib_msg:
            signals.append(fib_msg)


    return signals
