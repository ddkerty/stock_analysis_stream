# -*- coding: utf-8 -*-
# Combined app.py V1.9.1 - Added Debugging for Technical Analysis KeyError

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
    # 함수 내부에서도 방어적으로 체크
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"VWAP 계산 실패: 컬럼 부족 ({missing_cols})")
    if df['Volume'].isnull().all() or df['Volume'].sum() == 0:
        df['VWAP'] = np.nan
        logging.warning(f"Ticker {df.attrs.get('ticker', '')}: VWAP 계산 불가 (거래량 데이터 부족 또는 0)")
    else:
        df['Volume'].fillna(0, inplace=True)
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
    if required_col not in df.columns or df[required_col].isnull().all():
        raise ValueError(f"볼린저 밴드 계산 실패: 컬럼 '{required_col}' 없거나 데이터 없음.")
    valid_close = df.dropna(subset=[required_col])
    if len(valid_close) < window:
        st.warning(f"볼린저 밴드 계산 위한 유효 데이터({len(valid_close)}개)가 기간({window}개)보다 부족.")
        df['MA20'] = np.nan; df['Upper'] = np.nan; df['Lower'] = np.nan
    else:
        df['MA20'] = df[required_col].rolling(window=window, min_periods=window).mean()
        df['STD20'] = df[required_col].rolling(window=window, min_periods=window).std()
        df['Upper'] = df['MA20'] + num_std * df['STD20']
        df['Lower'] = df['MA20'] - num_std * df['STD20']
    return df

def plot_technical_chart(df, ticker):
    """기술적 분석 지표 통합 차트 생성"""
    fig = go.Figure()
    required_candle_cols = ['Open', 'High', 'Low', 'Close']
    # 차트 그리기 전 최종 확인
    if not all(col in df.columns for col in required_candle_cols) or df[required_candle_cols].isnull().all(axis=None):
        st.error(f"캔들차트 필요 컬럼({required_candle_cols}) 없거나 데이터 없음.")
        return fig
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name=f"{ticker} 캔들"))
    # VWAP
    if 'VWAP' in df.columns and df['VWAP'].notna().any():
        fig.add_trace(go.Scatter(x=df.index, y=df['VWAP'], mode='lines', name='VWAP', line=dict(color='orange', width=1.5)))
    elif 'VWAP' in df.columns: st.caption("VWAP 데이터 없음/표시 불가.")
    # Bollinger
    if 'Upper' in df.columns and 'Lower' in df.columns and df['Upper'].notna().any():
        if 'MA20' in df.columns and df['MA20'].notna().any(): fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], mode='lines', name='MA20', line=dict(color='blue', width=1, dash='dash')))
        fig.add_trace(go.Scatter(x=df.index, y=df['Upper'], mode='lines', name='Bollinger Upper', line=dict(color='grey', width=1)))
        fig.add_trace(go.Scatter(x=df.index, y=df['Lower'], mode='lines', name='Bollinger Lower', line=dict(color='grey', width=1), fill='tonexty', fillcolor='rgba(180,180,180,0.1)'))
    elif 'Upper' in df.columns: st.caption("볼린저 밴드 데이터 없음/표시 불가.")
    # Fibonacci
    valid_price_df = df.dropna(subset=['High', 'Low'])
    if not valid_price_df.empty:
        min_price = valid_price_df['Low'].min(); max_price = valid_price_df['High'].max(); diff = max_price - min_price
        if diff > 0:
            levels = {'0.0 (High)': max_price, '0.236': max_price - 0.236 * diff, '0.382': max_price - 0.382 * diff, '0.5': max_price - 0.5 * diff, '0.618': max_price - 0.618 * diff, '1.0 (Low)': min_price}
            fib_colors = {'0.0 (High)': 'red', '0.236': 'orange', '0.382': 'gold', '0.5': 'green', '0.618': 'blue', '1.0 (Low)': 'purple'}
            for k, v in levels.items(): fig.add_hline(y=v, line_dash="dot", annotation_text=f"Fib {k}: ${v:.2f}", line_color=fib_colors.get(k,'navy'), annotation_position="bottom right", annotation_font_size=10)
        else: st.caption("기간 내 가격 변동 없어 피보나치 미표시.")
    else: st.caption("피보나치 레벨 계산 불가.")
    fig.update_layout(title=f"{ticker} - 기술적 분석 통합 차트", xaxis_title="날짜 / 시간", yaxis_title="가격 ($)", xaxis_rangeslider_visible=False, legend_title_text="지표", hovermode="x unified", margin=dict(l=50, r=50, t=50, b=50))
    return fig

# --- Streamlit 페이지 설정 ---
st.set_page_config(page_title="종합 주식 분석 V1.9.1 (Debug)", layout="wide", initial_sidebar_state="expanded") # 버전 업데이트

# --- API 키 로드 ---
NEWS_API_KEY = None; FRED_API_KEY = None; api_keys_loaded = False
secrets_available = hasattr(st, 'secrets'); sidebar_status = st.sidebar.empty()
# ... (API 키 로드 로직은 V1.8과 동일) ...
if secrets_available:
    try: NEWS_API_KEY = st.secrets.get("NEWS_API_KEY"); FRED_API_KEY = st.secrets.get("FRED_API_KEY");
    except Exception as e: sidebar_status.error(f"Secrets 로드 오류: {e}")
if NEWS_API_KEY and FRED_API_KEY: api_keys_loaded = True
else: sidebar_status.warning("Secrets 키 일부 누락.")
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
    st.title("📊 분석 도구 V1.9.1") # 버전 업데이트
    st.markdown("---")
    page = st.radio("분석 유형 선택", ["📊 종합 분석", "📈 기술 분석"], captions=["재무, 예측, 뉴스 등", "VWAP, BB, 피보나치 등"], key="page_selector")
    st.markdown("---")
    if page == "📊 종합 분석":
        # (V1.8과 동일한 종합 분석 설정 로직)
        st.header("⚙️ 종합 분석 설정")
        ticker_input = st.text_input("종목 티커", "AAPL", key="main_ticker", help="해외(예: AAPL) 또는 국내(예: 005930.KS) 티커", disabled=not comprehensive_analysis_possible)
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
        # (V1.8과 동일한 기술 분석 설정 로직)
        st.header("⚙️ 기술 분석 설정")
        bb_window = st.number_input("볼린저밴드 기간 (일)", 5, 50, 20, 1, key="bb_window")
        bb_std = st.number_input("볼린저밴드 표준편차 배수", 1.0, 3.0, 2.0, 0.1, key="bb_std", format="%.1f")
        st.caption(f"현재 설정: {bb_window}일 기간, {bb_std:.1f} 표준편차")
        st.divider()

# --- 캐시된 종합 분석 함수 ---
@st.cache_data(ttl=timedelta(hours=1))
def run_cached_analysis(ticker, news_key, fred_key, years, days, num_trend_periods, changepoint_prior_scale):
    # (V1.8과 동일)
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
    # (V1.8과 동일한 로직 - 상세 결과 표시 포함)
    st.title("📊 종합 분석 결과"); st.markdown("기업 정보, 재무 추세, 예측, 리스크 트래커 제공."); st.markdown("---")
    analyze_button_main_disabled = not comprehensive_analysis_possible
    if analyze_button_main_disabled: st.error("API 키 로드 실패. 종합 분석 불가.")
    analyze_button_main = st.button("🚀 종합 분석 시작!", use_container_width=True, type="primary", key="analyze_main_button", disabled=analyze_button_main_disabled)
    results_placeholder = st.container()
    if analyze_button_main:
        ticker = st.session_state.get('main_ticker', "AAPL"); years = st.session_state.get('analysis_years', 2); days = st.session_state.get('forecast_days', 30); periods = st.session_state.get('num_trend_periods', 4); cp_prior = st.session_state.get('changepoint_prior', 0.05); avg_p = st.session_state.get('avg_price', 0.0); qty = st.session_state.get('quantity', 0)
        if not ticker: results_placeholder.warning("종목 티커 입력 필요.")
        else:
            ticker_proc = ticker.strip().upper()
            with st.spinner(f"{ticker_proc} 종합 분석 중..."):
                try:
                    results = run_cached_analysis(ticker_proc, NEWS_API_KEY, FRED_API_KEY, years, days, periods, cp_prior)
                    results_placeholder.empty()
                    if results and isinstance(results, dict) and "error" not in results:
                        # === 상세 결과 표시 시작 (V1.8 내용) ===
                        st.header(f"📈 {ticker_proc} 분석 결과 (민감도: {cp_prior:.3f})")
                        st.subheader("요약 정보"); col1, col2, col3 = st.columns(3); col1.metric("현재가", f"${results.get('current_price', 'N/A')}"); col2.metric("분석 시작일", results.get('analysis_period_start', 'N/A')); col3.metric("분석 종료일", results.get('analysis_period_end', 'N/A'))
                        st.subheader("📊 기업 기본 정보"); fundamentals = results.get('fundamentals')
                        if fundamentals and isinstance(fundamentals, dict) and fundamentals.get("시가총액", "N/A") != "N/A": colf1, colf2, colf3 = st.columns(3); # ... (metrics) ...; industry = fundamentals.get("산업", "N/A"); summary = fundamentals.get("요약", "N/A"); # ... (markdown/expander) ...
                        else: st.warning("기업 기본 정보 로드 실패.")
                        st.subheader(f"📈 주요 재무 추세 (최근 {periods} 분기)"); tab_titles = ["영업이익률(%)", "ROE(%)", "부채비율", "유동비율"]; tabs = st.tabs(tab_titles); trend_data_map = {"영업이익률(%)": ('operating_margin_trend', 'Op Margin (%)', "{:.2f}%"), "ROE(%)": ('roe_trend', 'ROE (%)', "{:.2f}%"), "부채비율": ('debt_to_equity_trend', 'D/E Ratio', "{:.2f}"), "유동비율": ('current_ratio_trend', 'Current Ratio', "{:.2f}")}
                        for i, title in enumerate(tab_titles): # ... (tab content) ...
                         st.divider(); st.subheader("기술적 분석 차트 (종합)"); stock_chart_fig = results.get('stock_chart_fig'); # ... (plotly_chart) ...; st.divider()
                        st.subheader("시장 심리 분석"); col_news, col_fng = st.columns([2, 1]); # ... (news/fng logic) ...; st.divider()
                        st.subheader("Prophet 주가 예측"); forecast_fig = results.get('forecast_fig'); forecast_data_list = results.get('prophet_forecast'); # ... (chart/data/cv logic) ...; st.divider()
                        st.subheader("🚨 리스크 트래커 (예측 기반)"); risk_days, max_loss_pct, max_loss_amt = 0, 0, 0; # ... (risk logic) ...; st.divider()
                        st.subheader("🧐 자동 분석 결과 요약 (참고용)"); summary_points = []; # ... (summary logic) ...
                        # === 상세 결과 표시 끝 ===
                    elif results and "error" in results: results_placeholder.error(f"분석 실패: {results['error']}")
                    else: results_placeholder.error("분석 결과 처리 중 오류.")
                except Exception as e: error_traceback = traceback.format_exc(); logging.error(f"종합 분석 실행 오류: {e}\n{error_traceback}"); results_placeholder.error(f"앱 실행 오류: {e}"); st.exception(e)
    else: # 버튼 클릭 전
        if comprehensive_analysis_possible: results_placeholder.info("⬅️ 사이드바 설정 후 '종합 분석 시작' 버튼 클릭.")


# ============== 📈 기술 분석 탭 (Debug Code Added) ==============
elif page == "📈 기술 분석":
    st.title("📈 기술적 분석 (VWAP + Bollinger + Fibonacci)")
    st.markdown("VWAP, 볼린저밴드, 피보나치 되돌림 수준을 함께 시각화합니다.")
    st.markdown("---")
    ticker_tech = st.text_input("종목 티커", "AAPL", key="tech_ticker", help="해외(예: AAPL) 또는 국내(예: 005930.KS) 티커")
    today = datetime.now().date(); default_start_date = today - relativedelta(months=3)
    col1, col2, col3 = st.columns(3)
    with col1: start_date = st.date_input("시작일", default_start_date, key="tech_start", max_value=today - timedelta(days=1))
    with col2: end_date = st.date_input("종료일", today, key="tech_end", min_value=start_date, max_value=today)
    with col3:
        interval_options = {"일봉": "1d", "1시간": "1h", "30분": "30m", "15분": "15m", "5분": "5m", "1분": "1m"}
        interval_display = st.selectbox("데이터 간격", list(interval_options.keys()), key="tech_interval_display", help="yfinance 기간 제약 확인")
        interval = interval_options[interval_display]
    bb_window_val = st.session_state.get('bb_window', 20)
    bb_std_val = st.session_state.get('bb_std', 2.0)
    analyze_button_tech = st.button("📊 기술적 분석 실행", key="tech_analyze", use_container_width=True, type="primary")

    if analyze_button_tech:
        if not ticker_tech: st.warning("종목 티커를 입력해주세요.")
        elif start_date >= end_date: st.warning("시작일은 종료일보다 이전이어야 합니다.")
        else:
            ticker_processed_tech = ticker_tech.strip().upper()
            df_tech = pd.DataFrame()
            st.write(f"**{ticker_processed_tech}** ({interval_display}, BB:{bb_window_val}일/{bb_std_val:.1f}σ) 분석 중...")
            with st.spinner(f"{ticker_processed_tech} 데이터 로딩 중..."):
                try: # yfinance 다운로드
                    period_days = (end_date - start_date).days; fetch_start_date = start_date
                    if interval == '1m' and period_days > 7: st.warning("1분봉 최대 7일 조회. 시작일 조정."); fetch_start_date = end_date - timedelta(days=7)
                    elif interval in ['5m', '15m', '30m'] and period_days > 60: st.warning(f"{interval_display} 최대 60일 조회. 시작일 조정."); fetch_start_date = end_date - timedelta(days=60)
                    fetch_end_date = end_date + timedelta(days=1)
                    logging.info(f"yf 다운로드 요청: {ticker_processed_tech}, {fetch_start_date}, {fetch_end_date}, {interval}")
                    df_tech = yf.download(ticker_processed_tech, start=fetch_start_date, end=fetch_end_date, interval=interval, progress=False)
                    if not df_tech.empty: df_tech.attrs['ticker'] = ticker_processed_tech
                except Exception as yf_err: st.error(f"yfinance 다운로드 오류: {yf_err}")

            analysis_successful = False # 분석 성공 여부 플래그 초기화
            if not df_tech.empty:
                logging.info(f"다운로드 완료. 행: {len(df_tech)}")
                st.caption(f"조회 기간: {df_tech.index.min():%Y-%m-%d %H:%M} ~ {df_tech.index.max():%Y-%m-%d %H:%M}")
                required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                missing_cols = [col for col in required_cols if col not in df_tech.columns]

                if missing_cols:
                    st.error(f"❌ 데이터에 필수 컬럼 누락: {missing_cols}. 실제 컬럼: {df_tech.columns.tolist()}")
                    st.info("지수, 환율 등 일부 자산은 필요한 컬럼이 없을 수 있습니다.")
                    st.dataframe(df_tech.head())
                    # 분석 성공 플래그는 False 유지
                else:
                    # 모든 필수 컬럼 존재 확인 -> 분석 성공 플래그 True
                    analysis_successful = True
            elif analyze_button_tech:
                 st.error(f"❌ 데이터 로드 실패. 티커/기간/간격 확인 필요.")

            # === 분석 성공 시에만 계산 및 차트 그리기 ===
            if analysis_successful:
                with st.spinner("기술적 지표 계산 및 차트 생성 중..."):
                    try:
                        # --- !!! 상세 디버깅 코드 추가 시작 !!! ---
                        st.error("--- DEBUG INFO ---")
                        st.write("dropna 호출 직전 df_tech.columns 객체:", df_tech.columns)
                        st.write("dropna 호출 직전 df_tech.columns 리스트:", df_tech.columns.tolist())
                        # repr() 사용하여 숨겨진 문자 확인 시도
                        st.write("dropna 호출 직전 df_tech.columns 리스트 (repr):", [repr(col) for col in df_tech.columns.tolist()])
                        st.write("dropna에 사용될 required_cols:", required_cols)
                        st.write("df_tech.columns 타입:", type(df_tech.columns))
                        st.write("required_cols 타입:", type(required_cols))
                        st.write("--- 컬럼 개별 비교 ---")
                        for req_col in required_cols:
                            match_found = False
                            for actual_col in df_tech.columns:
                                req_col_stripped = req_col.strip(); actual_col_stripped = actual_col.strip()
                                if req_col == actual_col: st.write(f"- '{req_col}' vs '{actual_col}': 정확히 일치!"); match_found = True; break
                                elif req_col_stripped == actual_col_stripped: st.warning(f"- '{req_col}' vs '{actual_col}': 공백 제거 후 일치! (원본: {repr(actual_col)})"); match_found = True; break
                                elif req_col.lower() == actual_col.lower(): st.warning(f"- '{req_col}' vs '{actual_col}': 대소문자 무시 후 일치! (원본: {repr(actual_col)})"); match_found = True; break
                            if not match_found: st.error(f"- '{req_col}': 일치하는 컬럼 없음!")
                        st.error("--- END DEBUG INFO ---")
                        # --- !!! 상세 디버깅 코드 추가 끝 !!! ---

                        # 데이터 정제 (KeyError 발생 지점)
                        df_processed = df_tech.dropna(subset=required_cols).copy() # inplace=False, copy()

                        if df_processed.empty:
                            st.warning("데이터 정제 후 남은 데이터가 없습니다.")
                        else:
                            # --- 개별 지표 계산 ---
                            df_calculated = df_processed # 복사본 사용
                            try: df_calculated = calculate_vwap(df_calculated)
                            except ValueError as ve_vwap: st.warning(f"VWAP 계산 불가: {ve_vwap}")
                            try: df_calculated = calculate_bollinger_bands(df_calculated, window=bb_window_val, num_std=bb_std_val)
                            except ValueError as ve_bb: st.warning(f"볼린저 밴드 계산 불가: {ve_bb}")

                            # --- 차트 생성 및 표시 ---
                            st.subheader(f"📌 {ticker_processed_tech} 기술적 분석 통합 차트 ({interval_display})")
                            chart_tech = plot_technical_chart(df_calculated, ticker_processed_tech)
                            st.plotly_chart(chart_tech, use_container_width=True)

                            # --- 데이터 표시 ---
                            st.subheader("📄 최근 데이터 (지표 포함)")
                            display_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'VWAP', 'MA20', 'Upper', 'Lower']
                            display_cols = [col for col in display_cols if col in df_calculated.columns]
                            format_dict = {col: "{:.2f}" for col in display_cols if col != 'Volume'}
                            st.dataframe(df_calculated[display_cols].tail(10).style.format(format_dict), use_container_width=True)

                    except KeyError as ke: # KeyError를 여기서도 잡아본다
                        st.error(f"!!! KeyError 발생 (디버깅 필요): {ke} !!!")
                        logging.error(f"KeyError during technical analysis processing: {traceback.format_exc()}")
                        st.info("위 DEBUG INFO의 컬럼 정보와 KeyError 메시지를 비교해주세요.")
                        st.dataframe(df_tech.head()) # 원본 데이터 표시
                    except Exception as e:
                        st.error(f"기술적 분석 처리 중 예상치 못한 오류: {type(e).__name__} - {e}")
                        logging.error(f"Technical analysis processing error: {traceback.format_exc()}")
                        st.dataframe(df_tech.head()) # 원본 데이터 표시

    else: # 버튼 클릭 전
        st.info("종목 티커, 기간, 간격 설정 후 '기술적 분석 실행' 버튼 클릭.")

# --- 앱 정보 ---
st.sidebar.markdown("---")
st.sidebar.info("종합 주식 분석 툴 V1.9.1 (Debug) | 정보 제공 목적")