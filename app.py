# -*- coding: utf-8 -*-
# Combined app.py V1.8 - Added Bollinger Bands to Short-Term Analysis Tab

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import os
from dotenv import load_dotenv
import traceback
import plotly.graph_objects as go
import numpy as np
import logging
import yfinance as yf

# --- 기본 경로 설정 및 로깅 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
try: BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError: BASE_DIR = os.getcwd()

# --- 기술 분석 함수 ---
def calculate_vwap(df):
    """VWAP 계산 (Volume 필요)"""
    df = df.copy()
    required_cols = ['High', 'Low', 'Close', 'Volume']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"VWAP 계산 필요 컬럼 부족. 필요: {required_cols}, 현재: {df.columns.tolist()}")
    if df['Volume'].sum() == 0:
        df['VWAP'] = np.nan # 거래량 0이면 NaN
    else:
        df['typical_price'] = (df['High'] + df['Low'] + df['Close']) / 3
        df['tp_volume'] = df['typical_price'] * df['Volume']
        df['cumulative_volume'] = df['Volume'].cumsum()
        df['cumulative_tp_volume'] = df['tp_volume'].cumsum()
        df['VWAP'] = np.where(df['cumulative_volume'] > 0, df['cumulative_tp_volume'] / df['cumulative_volume'], np.nan)
    return df

def calculate_bollinger_bands(df, window=20, num_std=2):
    """볼린저 밴드 계산 (Close 필요)"""
    df = df.copy()
    required_col = 'Close'
    if required_col not in df.columns:
        raise ValueError(f"볼린저 밴드 계산 필요 컬럼 '{required_col}' 부족. 현재: {df.columns.tolist()}")

    # 충분한 데이터가 있는지 확인 (window 크기보다 많아야 함)
    if len(df) < window:
        st.warning(f"데이터 기간({len(df)}일)이 볼린저 밴드 기간({window}일)보다 짧아 계산 불가.")
        df['MA20'] = np.nan
        df['Upper'] = np.nan
        df['Lower'] = np.nan
        return df

    df['MA20'] = df[required_col].rolling(window=window).mean()
    df['STD20'] = df[required_col].rolling(window=window).std()
    df['Upper'] = df['MA20'] + num_std * df['STD20']
    df['Lower'] = df['MA20'] - num_std * df['STD20']
    return df

def plot_technical_chart(df, ticker):
    """기술적 분석 지표(캔들, VWAP, 볼린저, 피보나치) 통합 차트 생성"""
    fig = go.Figure()
    required_candle_cols = ['Open', 'High', 'Low', 'Close']

    # 1. 캔들 차트 (필수)
    if not all(col in df.columns for col in required_candle_cols):
        st.error(f"캔들차트 필요 컬럼 부족: {required_candle_cols}")
        return fig # 빈 차트 반환
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name=f"{ticker} 캔들"
    ))

    # 2. VWAP (선택적)
    if 'VWAP' in df.columns and not df['VWAP'].isnull().all():
        fig.add_trace(go.Scatter(x=df.index, y=df['VWAP'], mode='lines', name='VWAP', line=dict(color='orange', width=1.5)))
    elif 'VWAP' in df.columns: st.caption("VWAP 데이터 없음(거래량 0 등).")

    # 3. 볼린저밴드 (선택적)
    if 'Upper' in df.columns and 'Lower' in df.columns and not df['Upper'].isnull().all():
        # MA20도 함께 표시 (선택 사항)
        if 'MA20' in df.columns and not df['MA20'].isnull().all():
             fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], mode='lines', name='MA20', line=dict(color='blue', width=1, dash='dash')))
        # 밴드 표시
        fig.add_trace(go.Scatter(x=df.index, y=df['Upper'], mode='lines', name='Bollinger Upper', line=dict(color='grey', width=1)))
        fig.add_trace(go.Scatter(x=df.index, y=df['Lower'], mode='lines', name='Bollinger Lower', line=dict(color='grey', width=1), fill='tonexty', fillcolor='rgba(180,180,180,0.1)')) # 투명도 조정
    elif 'Upper' in df.columns: st.caption("볼린저 밴드 데이터 부족 또는 오류.")

    # 4. 피보나치 되돌림선 (선택적)
    valid_price_df = df.dropna(subset=['High', 'Low'])
    if not valid_price_df.empty:
        min_price = valid_price_df['Low'].min(); max_price = valid_price_df['High'].max(); diff = max_price - min_price
        levels = { # 레벨 이름 명확화
            '0.0 (High)': max_price, '0.236': max_price - 0.236 * diff, '0.382': max_price - 0.382 * diff,
            '0.5': max_price - 0.5 * diff, '0.618': max_price - 0.618 * diff, '1.0 (Low)': min_price
        }
        fib_colors = {'0.0 (High)': 'red', '0.236': 'orange', '0.382': 'gold', '0.5': 'green', '0.618': 'blue', '1.0 (Low)': 'purple'} # 피보나치 색상 구분
        for k, v in levels.items():
            fig.add_hline(y=v, line_dash="dot", annotation_text=f"Fib {k}: ${v:.2f}", line_color=fib_colors.get(k,'navy'), annotation_position="bottom right", annotation_font_size=10)
    else: st.caption("피보나치 레벨 계산 불가 (가격 데이터 부족).")

    # 레이아웃 업데이트
    fig.update_layout(title=f"{ticker} - 기술적 분석 통합 차트", xaxis_title="날짜 / 시간", yaxis_title="가격 ($)",
                      xaxis_rangeslider_visible=False, legend_title_text="지표", hovermode="x unified", margin=dict(l=50, r=50, t=50, b=50))
    return fig

# --- Streamlit 페이지 설정 ---
st.set_page_config(page_title="종합 주식 분석 V1.8", layout="wide", initial_sidebar_state="expanded") # 버전 업데이트

# --- API 키 로드 ---
# (V1.7과 동일한 로직)
NEWS_API_KEY = None; FRED_API_KEY = None; api_keys_loaded = False
secrets_available = hasattr(st, 'secrets'); sidebar_status = st.sidebar.empty()
if secrets_available:
    try:
        NEWS_API_KEY = st.secrets.get("NEWS_API_KEY"); FRED_API_KEY = st.secrets.get("FRED_API_KEY")
        if NEWS_API_KEY and FRED_API_KEY: api_keys_loaded = True
        else: sidebar_status.warning("Secrets 키 일부 누락.")
    except Exception as e: sidebar_status.error(f"Secrets 로드 오류: {e}")
if not api_keys_loaded:
    sidebar_status.info(".env 파일 확인 중...")
    try:
        dotenv_path = os.path.join(BASE_DIR, '.env')
        if os.path.exists(dotenv_path):
            load_dotenv(dotenv_path=dotenv_path)
            NEWS_API_KEY = os.getenv("NEWS_API_KEY"); FRED_API_KEY = os.getenv("FRED_API_KEY")
            if NEWS_API_KEY and FRED_API_KEY: api_keys_loaded = True; sidebar_status.success("API 키 로드 완료 (.env)")
            else: sidebar_status.error(".env 키 일부 누락.")
        else: sidebar_status.error(".env 파일 없음.")
    except Exception as e: sidebar_status.error(f".env 로드 오류: {e}")
comprehensive_analysis_possible = api_keys_loaded
if not api_keys_loaded: st.sidebar.error("API 키 로드 실패! '종합 분석' 제한.")
else: sidebar_status.success("API 키 로드 완료.")

# --- 사이드바 설정 ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/10071/10071119.png", width=80)
    st.title("📊 분석 도구 V1.8") # 버전 업데이트
    st.markdown("---")
    # 기술 분석 탭 이름 변경
    page = st.radio("분석 유형 선택", ["📊 종합 분석", "📈 기술 분석"], captions=["재무, 예측, 뉴스 등", "VWAP, BB, 피보나치 등"], key="page_selector")
    st.markdown("---")
    if page == "📊 종합 분석":
        st.header("⚙️ 종합 분석 설정")
        # V1.8 코드의 해당 라인을 아래와 같이 수정
        ticker_input = st.text_input(
            "종목 티커",
            value="AAPL",  # 기본값은 그대로 두거나 원하는 것으로 변경
            key="main_ticker",
            # help 파라미터 수정
            help="해외(예: AAPL, MSFT) 또는 국내(예: 005930.KS) 티커 입력",
            disabled=not comprehensive_analysis_possible
        )
        # (V1.7과 동일한 종합 분석 설정 로직)
        st.header("⚙️ 종합 분석 설정")
        ticker_input = st.text_input("종목 티커", "AAPL", key="main_ticker", disabled=not comprehensive_analysis_possible)
        analysis_years = st.select_slider("분석 기간 (년)", [1, 2, 3, 5, 7, 10], 2, key="analysis_years", disabled=not comprehensive_analysis_possible)
        st.caption(f"과거 {analysis_years}년 데이터 분석")
        forecast_days = st.number_input("예측 기간 (일)", 7, 90, 30, 7, key="forecast_days", disabled=not comprehensive_analysis_possible)
        st.caption(f"향후 {forecast_days}일 예측")
        num_trend_periods_input = st.number_input("재무 추세 분기 수", 2, 12, 4, 1, key="num_trend_periods", disabled=not comprehensive_analysis_possible)
        st.caption(f"최근 {num_trend_periods_input}개 분기 재무 추세 계산")
        st.divider(); st.subheader("⚙️ 예측 세부 설정 (선택)")
        changepoint_prior_input = st.slider("추세 변화 민감도 (Prophet)", 0.001, 0.5, 0.05, 0.01, "%.3f", help="클수록 과거 추세 변화에 민감 (기본값: 0.05)", key="changepoint_prior", disabled=not comprehensive_analysis_possible)
        st.caption(f"현재 민감도: {changepoint_prior_input:.3f}")
        st.divider(); st.subheader("💰 보유 정보 입력 (선택)")
        avg_price = st.number_input("평단가", 0.0, format="%.2f", key="avg_price", disabled=not comprehensive_analysis_possible)
        quantity = st.number_input("보유 수량", 0, step=1, key="quantity", disabled=not comprehensive_analysis_possible)
        st.caption("평단가 입력 시 리스크 트래커 분석 활성화"); st.divider()
    elif page == "📈 기술 분석":
        # 기술 분석 탭용 설정 (선택적: 볼린저밴드 기간/표준편차 등)
        st.header("⚙️ 기술 분석 설정")
        bb_window = st.number_input("볼린저밴드 기간 (일)", min_value=5, max_value=50, value=20, step=1, key="bb_window")
        bb_std = st.number_input("볼린저밴드 표준편차 배수", min_value=1.0, max_value=3.0, value=2.0, step=0.1, key="bb_std", format="%.1f")
        st.caption(f"현재 설정: {bb_window}일 기간, {bb_std:.1f} 표준편차")
        st.divider()

# --- 캐시된 종합 분석 함수 ---
# (V1.7과 동일)
@st.cache_data(ttl=timedelta(hours=1))
def run_cached_analysis(ticker, news_key, fred_key, years, days, num_trend_periods, changepoint_prior_scale):
    try: import stock_analysis as sa
    except ImportError as import_err: return {"error": f"분석 모듈(stock_analysis.py) 로딩 오류: {import_err}."}
    except Exception as e: return {"error": f"분석 모듈 로딩 중 오류: {e}"}
    logging.info(f"종합 분석 실행: {ticker}, {years}년, {days}일, {num_trend_periods}분기, cp_prior={changepoint_prior_scale}")
    if not news_key or not fred_key: logging.warning("API 키 없이 종합 분석 시도.")
    try: return sa.analyze_stock(ticker, news_key, fred_key, analysis_period_years=years, forecast_days=days, num_trend_periods=num_trend_periods, changepoint_prior_scale=changepoint_prior_scale)
    except Exception as e: logging.error(f"analyze_stock 함수 오류: {e}\n{traceback.format_exc()}"); return {"error": f"종합 분석 중 오류 발생: {e}"}

# --- 메인 화면 로직 ---

# ============== 📊 종합 분석 탭 ==============
if page == "📊 종합 분석":
    # (V1.7과 동일한 로직 - 상세 결과 표시 포함)
    st.title("📊 종합 분석 결과"); st.markdown("기업 정보, 재무 추세, 예측, 리스크 트래커 제공."); st.markdown("---")
    analyze_button_main_disabled = not comprehensive_analysis_possible
    if analyze_button_main_disabled: st.error("API 키 로드 실패. 종합 분석 불가.")
    analyze_button_main = st.button("🚀 종합 분석 시작!", use_container_width=True, type="primary", key="analyze_main_button", disabled=analyze_button_main_disabled)
    results_placeholder = st.container()
    if analyze_button_main:
        ticker = st.session_state.get('main_ticker', "AAPL"); years = st.session_state.get('analysis_years', 2)
        days = st.session_state.get('forecast_days', 30); periods = st.session_state.get('num_trend_periods', 4)
        cp_prior = st.session_state.get('changepoint_prior', 0.05); avg_p = st.session_state.get('avg_price', 0.0)
        qty = st.session_state.get('quantity', 0)
        if not ticker: results_placeholder.warning("종목 티커 입력 필요.")
        else:
            ticker_proc = ticker.strip().upper()
            with st.spinner(f"{ticker_proc} 종합 분석 중..."):
                try:
                    results = run_cached_analysis(ticker_proc, NEWS_API_KEY, FRED_API_KEY, years, days, periods, cp_prior)
                    results_placeholder.empty()
                    if results and isinstance(results, dict) and "error" not in results:
                        # === 상세 결과 표시 시작 (V1.7 복원 내용 유지) ===
                        st.header(f"📈 {ticker_proc} 분석 결과 (민감도: {cp_prior:.3f})")
                        # 1. 요약 정보 ... (이하 V1.7과 동일하게 상세 결과 표시 로직 유지)
                        st.subheader("요약 정보"); col1, col2, col3 = st.columns(3)
                        col1.metric("현재가", f"${results.get('current_price', 'N/A')}")
                        col2.metric("분석 시작일", results.get('analysis_period_start', 'N/A'))
                        col3.metric("분석 종료일", results.get('analysis_period_end', 'N/A'))
                        # 2. 기본적 분석 ...
                        st.subheader("📊 기업 기본 정보"); fundamentals = results.get('fundamentals')
                        # ... (V1.7 상세 내용) ...
                        # 3. 주요 재무 추세 ...
                        st.subheader(f"📈 주요 재무 추세 (최근 {periods} 분기)"); tab_titles = ["영업이익률(%)", "ROE(%)", "부채비율", "유동비율"]; tabs = st.tabs(tab_titles)
                        # ... (V1.7 상세 내용) ...
                        st.divider()
                        # 4. 기술적 분석 차트 (종합) ...
                        st.subheader("기술적 분석 차트 (종합)"); stock_chart_fig = results.get('stock_chart_fig')
                        # ... (V1.7 상세 내용) ...
                        st.divider()
                        # 5. 시장 심리 분석 ...
                        st.subheader("시장 심리 분석"); col_news, col_fng = st.columns([2, 1])
                        # ... (V1.7 상세 내용) ...
                        st.divider()
                        # 6. Prophet 주가 예측 ...
                        st.subheader("Prophet 주가 예측"); forecast_fig = results.get('forecast_fig'); forecast_data_list = results.get('prophet_forecast')
                        # ... (V1.7 상세 내용) ...
                        st.divider()
                        # 7. 리스크 트래커 ...
                        st.subheader("🚨 리스크 트래커 (예측 기반)"); risk_days, max_loss_pct, max_loss_amt = 0, 0, 0
                        # ... (V1.7 상세 내용) ...
                        st.divider()
                        # 8. 자동 분석 결과 요약 ...
                        st.subheader("🧐 자동 분석 결과 요약 (참고용)"); summary_points = []
                        # ... (V1.7 상세 내용) ...
                        # === 상세 결과 표시 끝 ===
                    elif results and "error" in results: results_placeholder.error(f"분석 실패: {results['error']}")
                    else: results_placeholder.error("분석 결과 처리 중 오류.")
                except Exception as e: error_traceback = traceback.format_exc(); logging.error(f"종합 분석 실행 오류: {e}\n{error_traceback}"); results_placeholder.error(f"앱 실행 오류: {e}"); st.exception(e)
    else:
        if comprehensive_analysis_possible: results_placeholder.info("⬅️ 사이드바 설정 후 '종합 분석 시작' 버튼 클릭.")


# ============== 📈 기술 분석 탭 (Bollinger Bands 추가) ==============
elif page == "📈 기술 분석":
    st.title("📈 기술적 분석 (VWAP + Bollinger + Fibonacci)")
    st.markdown("VWAP, 볼린저밴드, 피보나치 되돌림 수준을 함께 시각화합니다.")
    st.markdown("---")

    # V1.8 코드의 해당 라인을 아래와 같이 수정 (help 추가)
    ticker_tech = st.text_input(
        "종목 티커",
        value="AAPL", # 기본값
        key="tech_ticker",
        # help 파라미터 추가
        help="해외(예: AAPL, TSLA) 또는 국내(예: 005930.KS) 티커 입력"
    )

    # 입력 위젯 (V1.7과 동일)
    ticker_tech = st.text_input("종목 티커", "AAPL", key="tech_ticker")
    today = datetime.now().date(); default_start_date = today - relativedelta(months=3)
    col1, col2, col3 = st.columns(3)
    with col1: start_date = st.date_input("시작일", default_start_date, key="tech_start", max_value=today - timedelta(days=1))
    with col2: end_date = st.date_input("종료일", today, key="tech_end", min_value=start_date, max_value=today)
    with col3:
        interval_options = {"일봉": "1d", "1시간": "1h", "30분": "30m", "15분": "15m", "5분": "5m", "1분": "1m"}
        interval_display = st.selectbox("데이터 간격", list(interval_options.keys()), key="tech_interval_display", help="yfinance 기간 제약 확인")
        interval = interval_options[interval_display]

    # 볼린저 밴드 설정값 가져오기 (사이드바에서 정의)
    bb_window_val = st.session_state.get('bb_window', 20)
    bb_std_val = st.session_state.get('bb_std', 2.0)

    analyze_button_tech = st.button("📊 기술적 분석 실행", key="tech_analyze", use_container_width=True, type="primary") # 버튼 텍스트 수정

    if analyze_button_tech:
        if not ticker_tech: st.warning("종목 티커를 입력해주세요.")
        elif start_date >= end_date: st.warning("시작일은 종료일보다 이전이어야 합니다.")
        else:
            ticker_processed_tech = ticker_tech.strip().upper()
            st.write(f"**{ticker_processed_tech}** ({interval_display}, BB:{bb_window_val}일/{bb_std_val:.1f}σ) 분석 중...") # 진행 메시지에 BB 설정 포함
            with st.spinner(f"{ticker_processed_tech} ({interval_display}) 데이터 불러오는 중..."):
                try: # yfinance 다운로드
                    period_days = (end_date - start_date).days; fetch_start_date = start_date
                    if interval == '1m' and period_days > 7: st.warning("1분봉 최대 7일 조회. 시작일 조정."); fetch_start_date = end_date - timedelta(days=7)
                    elif interval in ['5m', '15m', '30m'] and period_days > 60: st.warning(f"{interval_display} 최대 60일 조회. 시작일 조정."); fetch_start_date = end_date - timedelta(days=60)
                    fetch_end_date = end_date + timedelta(days=1)
                    logging.info(f"yf 다운로드 요청: {ticker_processed_tech}, {fetch_start_date}, {fetch_end_date}, {interval}")
                    df_tech = yf.download(ticker_processed_tech, start=fetch_start_date, end=fetch_end_date, interval=interval, progress=False)
                except Exception as yf_err: st.error(f"yfinance 다운로드 오류: {yf_err}"); df_tech = pd.DataFrame()

            if not df_tech.empty:
                logging.info(f"다운로드 완료. 행: {len(df_tech)}")
                st.caption(f"조회 기간: {df_tech.index.min():%Y-%m-%d %H:%M} ~ {df_tech.index.max():%Y-%m-%d %H:%M}")
                required_cols = ['High', 'Low', 'Close', 'Volume', 'Open'] # 기본 컬럼
                if not all(col in df_tech.columns for col in required_cols):
                    st.error(f"❌ 필수 컬럼({required_cols}) 누락. 실제: {df_tech.columns.tolist()}")
                    st.dataframe(df_tech.head())
                else:
                    with st.spinner("기술적 지표 계산 및 차트 생성 중..."):
                        try:
                            df_tech.dropna(subset=required_cols, inplace=True) # 기본 컬럼 기준 NaN 제거
                            if df_tech.empty: st.warning("데이터 정제 후 남은 데이터 없음.")
                            else:
                                # VWAP 계산 (오류 발생 가능성 있음, Volume 없을 시)
                                try: df_tech = calculate_vwap(df_tech)
                                except ValueError as e: st.warning(f"VWAP 계산 불가: {e}") # 오류 대신 경고 표시

                                # 볼린저 밴드 계산 (오류 발생 가능성 있음, Close 없을 시, 기간 짧을 시)
                                try: df_tech = calculate_bollinger_bands(df_tech, window=bb_window_val, num_std=bb_std_val)
                                except ValueError as e: st.warning(f"볼린저 밴드 계산 불가: {e}") # 오류 대신 경고 표시

                                # 통합 차트 생성 및 표시
                                st.subheader(f"📌 {ticker_processed_tech} 기술적 분석 통합 차트 ({interval_display})")
                                chart_tech = plot_technical_chart(df_tech, ticker_processed_tech)
                                st.plotly_chart(chart_tech, use_container_width=True)

                                # 데이터 표시 (계산된 지표 포함)
                                st.subheader("📄 최근 데이터 (지표 포함)")
                                display_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                                if 'VWAP' in df_tech: display_cols.append('VWAP')
                                if 'MA20' in df_tech: display_cols.append('MA20')
                                if 'Upper' in df_tech: display_cols.append('Upper')
                                if 'Lower' in df_tech: display_cols.append('Lower')
                                # 존재하지 않는 컬럼 접근 방지
                                display_cols = [col for col in display_cols if col in df_tech.columns]
                                format_dict = {col: "{:.2f}" for col in display_cols if col not in ['Volume']} # Volume 제외하고 서식 지정
                                st.dataframe(df_tech[display_cols].tail(10).style.format(format_dict), use_container_width=True)

                        except Exception as e: # 계산 또는 차트 생성 중 예상 못한 오류
                             st.error(f"기술적 분석 처리 중 오류 발생: {e}")
                             logging.error(f"Technical analysis processing error: {traceback.format_exc()}")
                             st.dataframe(df_tech.tail(10), use_container_width=True) # 원본 데이터라도 표시
            elif analyze_button_tech: # 버튼은 눌렀는데 df_tech가 비어있는 경우
                 st.error(f"❌ 데이터 로드 실패. 티커/기간/간격 확인 필요.")
    else: # 버튼 클릭 전
        st.info("종목 티커, 기간, 간격 설정 후 '기술적 분석 실행' 버튼 클릭.")


# --- 앱 정보 ---
st.sidebar.markdown("---")
st.sidebar.info("종합 주식 분석 툴 V1.8 | 정보 제공 목적 (투자 조언 아님)")