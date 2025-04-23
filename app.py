# -*- coding: utf-8 -*-
# Combined app.py V1.9.6 (Style fix applied based on feedback)

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
from short_term_analysis import interpret_fibonacci
from technical_interpret import interpret_technical_signals
from short_term_analysis import calculate_rsi, calculate_macd


# --- 기본 경로 설정 및 로깅 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
try: BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError: BASE_DIR = os.getcwd()

# --- 기술 분석 함수 ---
def calculate_vwap(df):
    """VWAP 계산 (Volume 필요)"""
    df = df.copy()
    required_cols = ['High', 'Low', 'Close', 'Volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols: raise ValueError(f"VWAP 계산 실패: 컬럼 부족 ({missing_cols})")
    if df['Volume'].isnull().all() or df['Volume'].sum() == 0:
        df['VWAP'] = np.nan
        logging.warning(f"Ticker {df.attrs.get('ticker', '')}: VWAP 계산 불가 (거래량 부족/0)")
    else:
        # --- FutureWarning 수정: inplace=True 대신 재할당 사용 ---
        df['Volume'] = df['Volume'].fillna(0)
        # --- 수정 끝 ---
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
    if required_col not in df.columns or df[required_col].isnull().all(): raise ValueError(f"BB 계산 실패: 컬럼 '{required_col}' 없음/데이터 없음.")
    valid_close = df.dropna(subset=[required_col])
    if len(valid_close) < window:
        st.warning(f"BB 계산 위한 유효 데이터({len(valid_close)}개)가 기간({window}개)보다 부족.")
        df['MA20'] = np.nan
        df['Upper'] = np.nan
        df['Lower'] = np.nan
    else:
        df['MA20'] = df[required_col].rolling(window=window, min_periods=window).mean()
        df['STD20'] = df[required_col].rolling(window=window, min_periods=window).std()
        df['Upper'] = df['MA20'] + num_std * df['STD20']
        df['Lower'] = df['MA20'] - num_std * df['STD20']
    return df

def plot_technical_chart(df, ticker):
    """기술적 분석 지표 통합 차트 생성 (VWAP, Bollinger Band, Fibonacci, RSI, MACD 포함)"""
    fig = go.Figure()
    required_candle_cols = ['Open', 'High', 'Low', 'Close']
    if not all(col in df.columns for col in required_candle_cols) or df[required_candle_cols].isnull().all(axis=None):
        st.error(f"캔들차트 필요 컬럼({required_candle_cols}) 없음/데이터 없음.")
        return fig

    # --- (1) 캔들 차트 ---
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'],
                                 low=df['Low'], close=df['Close'], name=f"{ticker} 캔들"))

    # --- (2) VWAP ---
    if 'VWAP' in df.columns and df['VWAP'].notna().any():
        fig.add_trace(go.Scatter(x=df.index, y=df['VWAP'], mode='lines',
                                 name='VWAP', line=dict(color='orange', width=1.5)))
    elif 'VWAP' in df.columns:
        st.caption("VWAP 데이터 없음/표시 불가.")

    # --- (3) Bollinger Bands ---
    if 'Upper' in df.columns and 'Lower' in df.columns and df['Upper'].notna().any():
        if 'MA20' in df.columns and df['MA20'].notna().any():
            fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], mode='lines', name='MA20',
                                     line=dict(color='blue', width=1, dash='dash')))
        fig.add_trace(go.Scatter(x=df.index, y=df['Upper'], mode='lines', name='Bollinger Upper',
                                 line=dict(color='grey', width=1)))
        fig.add_trace(go.Scatter(x=df.index, y=df['Lower'], mode='lines', name='Bollinger Lower',
                                 line=dict(color='grey', width=1), fill='tonexty',
                                 fillcolor='rgba(180,180,180,0.1)'))
    elif 'Upper' in df.columns:
        st.caption("볼린저 밴드 데이터 없음/표시 불가.")

    # --- (4) Fibonacci Levels ---
    valid_price_df = df.dropna(subset=['High', 'Low'])
    if not valid_price_df.empty:
        min_price = valid_price_df['Low'].min()
        max_price = valid_price_df['High'].max()
        diff = max_price - min_price
        if diff > 0:
            levels = {'0.0 (High)': max_price, '0.236': max_price - 0.236 * diff,
                      '0.382': max_price - 0.382 * diff, '0.5': max_price - 0.5 * diff,
                      '0.618': max_price - 0.618 * diff, '1.0 (Low)': min_price}
            fib_colors = {'0.0 (High)': 'red', '0.236': 'orange', '0.382': 'gold',
                          '0.5': 'green', '0.618': 'blue', '1.0 (Low)': 'purple'}
            for k, v in levels.items():
                fig.add_hline(y=v, line_dash="dot", annotation_text=f"Fib {k}: ${v:.2f}",
                              line_color=fib_colors.get(k, 'navy'), annotation_position="bottom right",
                              annotation_font_size=10)
        else:
            st.caption("기간 내 가격 변동 없어 피보나치 미표시.")
    else:
        st.caption("피보나치 레벨 계산 불가.")

    # --- (5) RSI ---
    if 'RSI' in df.columns and df['RSI'].notna().any():
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], mode='lines', name='RSI (14)',
                                 line=dict(color='purple', width=1), yaxis='y2'))

    # --- (6) MACD ---
    if 'MACD' in df.columns and 'MACD_signal' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD', mode='lines',
                                 line=dict(color='teal'), yaxis='y3'))
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD_signal'], name='MACD Signal', mode='lines',
                                 line=dict(color='orange'), yaxis='y3'))
        if 'MACD_hist' in df.columns:
            fig.add_trace(go.Bar(x=df.index, y=df['MACD_hist'], name='MACD Histogram',
                                 marker_color='lightblue', yaxis='y3'))

    # --- 레이아웃 ---
    fig.update_layout(
        title=f"{ticker} - 기술적 분석 통합 차트",
        xaxis_title="날짜 / 시간",
        yaxis=dict(domain=[0.4, 1], title="가격 ($)"),
        yaxis2=dict(domain=[0.25, 0.4], title="RSI", overlaying='y', side='right'),
        yaxis3=dict(domain=[0.0, 0.25], title="MACD", overlaying='y', side='right'),
        xaxis_rangeslider_visible=False,
        legend_title_text="지표",
        hovermode="x unified",
        margin=dict(l=50, r=50, t=50, b=50)
    )
    return fig


# --- Streamlit 페이지 설정 ---
st.set_page_config(page_title="종합 주식 분석 V1.9.6", layout="wide", initial_sidebar_state="expanded") # 버전 업데이트

# --- API 키 로드 ---
NEWS_API_KEY = None
FRED_API_KEY = None
api_keys_loaded = False
secrets_available = hasattr(st, 'secrets')
sidebar_status = st.sidebar.empty()
# ... (API 키 로드 상세 로직) ...
if secrets_available:
    try:
        NEWS_API_KEY = st.secrets.get("NEWS_API_KEY")
        FRED_API_KEY = st.secrets.get("FRED_API_KEY")
    except Exception as e:
        sidebar_status.error(f"Secrets 로드 오류: {e}")
if NEWS_API_KEY and FRED_API_KEY:
    api_keys_loaded = True
else:
    sidebar_status.warning("Secrets 키 일부 누락.")
if not api_keys_loaded:
    sidebar_status.info(".env 파일 확인 중...")
    try:
        dotenv_path = os.path.join(BASE_DIR, '.env')
        if os.path.exists(dotenv_path):
            load_dotenv(dotenv_path=dotenv_path)
            NEWS_API_KEY = os.getenv("NEWS_API_KEY")
            FRED_API_KEY = os.getenv("FRED_API_KEY")
            if NEWS_API_KEY and FRED_API_KEY:
                api_keys_loaded = True
                sidebar_status.success("API 키 로드 완료 (.env)")
            else:
                sidebar_status.error(".env 키 일부 누락.")
        else:
            sidebar_status.error(".env 파일 없음.")
    except Exception as e:
        sidebar_status.error(f".env 로드 오류: {e}")
comprehensive_analysis_possible = api_keys_loaded
if not api_keys_loaded:
    st.sidebar.error("API 키 로드 실패! '종합 분석' 제한.")
else:
    sidebar_status.success("API 키 로드 완료.")


# --- 사이드바 설정 ---
with st.sidebar:
    #불필요 한 이미지 st.image("https://cdn-icons-png.flaticon.com/512/10071/10071119.png", width=80)
        # 🔗 강조 스타일 블로그 링크
    st.markdown("📘 [분석도구 상세정보](https://technut.tistory.com/3)", unsafe_allow_html=True)
    st.title("설명서 및 업데이트 (업데이트 예정)") # 버전 업데이트
    st.title("📊 주식 분석 도구 V1.9.6") # 버전 업데이트
    st.markdown("---")
    page = st.radio("분석 유형 선택", ["📊 종합 분석", "📈 기술 분석"], captions=["재무, 예측, 뉴스 등", "VWAP, BB, 피보나치 등"], key="page_selector")
    st.markdown("---")
    if page == "📊 종합 분석":
        # (V1.9.5와 동일한 종합 분석 설정 로직)
        st.header("⚙️ 종합 분석 설정")
        ticker_input = st.text_input("종목 티커", "AAPL", key="main_ticker", help="해외(예: AAPL) 또는 국내(예: 005930.KS) 티커", disabled=not comprehensive_analysis_possible)
        analysis_years = st.select_slider("분석 기간 (년)", [1, 2, 3, 5, 7, 10], 2, key="analysis_years", disabled=not comprehensive_analysis_possible)
        st.caption(f"과거 {analysis_years}년 데이터 분석")
        forecast_days = st.number_input("예측 기간 (일)", 7, 90, 30, 7, key="forecast_days", disabled=not comprehensive_analysis_possible)
        st.caption(f"향후 {forecast_days}일 예측")
        num_trend_periods_input = st.number_input("재무 추세 분기 수", 2, 12, 4, 1, key="num_trend_periods", disabled=not comprehensive_analysis_possible)
        st.caption(f"최근 {num_trend_periods_input}개 분기 재무 추세 계산")
        st.divider()
        st.subheader("⚙️ 예측 세부 설정 (선택)")
        changepoint_prior_input = st.slider("추세 변화 민감도 (Prophet)", 0.001, 0.5, 0.05, 0.01, "%.3f", help="클수록 과거 추세 변화에 민감 (기본값: 0.05)", key="changepoint_prior", disabled=not comprehensive_analysis_possible)
        st.caption(f"현재 민감도: {changepoint_prior_input:.3f}")
        st.divider()
        st.subheader("💰 보유 정보 입력 (선택)")
        avg_price = st.number_input("평단가", 0.0, format="%.2f", key="avg_price", disabled=not comprehensive_analysis_possible)
        quantity = st.number_input("보유 수량", 0, step=1, key="quantity", disabled=not comprehensive_analysis_possible)
        st.caption("평단가 입력 시 리스크 트래커 분석 활성화")
        st.divider()
    elif page == "📈 기술 분석":
        # (V1.9.5와 동일한 기술 분석 설정 로직)
        st.header("⚙️ 기술 분석 설정")
        bb_window = st.number_input("볼린저밴드 기간 (일)", 5, 50, 20, 1, key="bb_window")
        bb_std = st.number_input("볼린저밴드 표준편차 배수", 1.0, 3.0, 2.0, 0.1, key="bb_std", format="%.1f")
        st.caption(f"현재 설정: {bb_window}일 기간, {bb_std:.1f} 표준편차")
        st.divider()

# --- 캐시된 종합 분석 함수 ---
@st.cache_data(ttl=timedelta(hours=1))
def run_cached_analysis(ticker, news_key, fred_key, years, days, num_trend_periods, changepoint_prior_scale):
    # (V1.9.5와 동일)
    try: import stock_analysis as sa
    except ImportError as import_err: return {"error": f"분석 모듈(stock_analysis.py) 로딩 오류: {import_err}."}
    except Exception as e: return {"error": f"분석 모듈 로딩 중 오류: {e}"}
    logging.info(f"종합 분석 실행: {ticker}, {years}년, {days}일, {num_trend_periods}분기, cp_prior={changepoint_prior_scale}")
    if not news_key or not fred_key: logging.warning("API 키 없이 종합 분석 시도.")
    try:
        return sa.analyze_stock(ticker, news_key, fred_key, analysis_period_years=years, forecast_days=days, num_trend_periods=num_trend_periods, changepoint_prior_scale=changepoint_prior_scale)
    except Exception as e:
        logging.error(f"analyze_stock 함수 오류: {e}\n{traceback.format_exc()}")
        return {"error": f"종합 분석 중 오류 발생: {e}"}

# --- 메인 화면 로직 ---

# ============== 📊 종합 분석 탭 ==============
if page == "📊 종합 분석":
    # (V1.9.5와 동일한 로직 - 상세 결과 표시 포함)
    st.title("📊 종합 분석 결과")
    st.markdown("기업 정보, 재무 추세, 예측, 리스크 트래커 제공.")
    st.markdown("---")
    analyze_button_main_disabled = not comprehensive_analysis_possible
    if analyze_button_main_disabled: st.error("API 키 로드 실패. 종합 분석 불가.")
    analyze_button_main = st.button("🚀 종합 분석 시작!", use_container_width=True, type="primary", key="analyze_main_button", disabled=analyze_button_main_disabled)
    results_placeholder = st.container()
    if analyze_button_main:
        ticker = st.session_state.get('main_ticker', "AAPL")
        years = st.session_state.get('analysis_years', 2)
        days = st.session_state.get('forecast_days', 30)
        periods = st.session_state.get('num_trend_periods', 4)
        cp_prior = st.session_state.get('changepoint_prior', 0.05)
        avg_p = st.session_state.get('avg_price', 0.0)
        qty = st.session_state.get('quantity', 0)
        if not ticker: results_placeholder.warning("종목 티커 입력 필요.")
        else:
            ticker_proc = ticker.strip().upper()
            with st.spinner(f"{ticker_proc} 종합 분석 중..."):
                try:
                    results = run_cached_analysis(ticker_proc, NEWS_API_KEY, FRED_API_KEY, years, days, periods, cp_prior)
                    results_placeholder.empty()
                    if results and isinstance(results, dict) and "error" not in results:
                        # === 상세 결과 표시 (V1.9.5 내용 유지, 재무추세 부분 가독성 수정) ===
                        st.header(f"📈 {ticker_proc} 분석 결과 (민감도: {cp_prior:.3f})")
                        # 1. 요약 정보
                        st.subheader("요약 정보")
                        col1, col2, col3 = st.columns(3)
                        col1.metric("현재가", f"${results.get('current_price', 'N/A')}")
                        col2.metric("분석 시작일", results.get('analysis_period_start', 'N/A'))
                        col3.metric("분석 종료일", results.get('analysis_period_end', 'N/A'))
                        # 2. 기본적 분석
                        st.subheader("📊 기업 기본 정보")
                        fundamentals = results.get('fundamentals')
                        if fundamentals and isinstance(fundamentals, dict) and fundamentals.get("시가총액", "N/A") != "N/A":
                            colf1, colf2, colf3 = st.columns(3)
                            with colf1:
                                st.metric("시가총액", fundamentals.get("시가총액", "N/A"))
                                st.metric("PER", fundamentals.get("PER", "N/A"))
                            with colf2:
                                st.metric("EPS", fundamentals.get("EPS", "N/A"))
                                st.metric("Beta", fundamentals.get("베타", "N/A"))
                            with colf3:
                                st.metric("배당수익률", fundamentals.get("배당수익률", "N/A"))
                                st.metric("업종", fundamentals.get("업종", "N/A"))
                            industry = fundamentals.get("산업", "N/A")
                            summary = fundamentals.get("요약", "N/A")
                            if industry != "N/A": st.markdown(f"**산업:** {industry}")
                            if summary != "N/A":
                                with st.expander("회사 요약 보기"):
                                    st.write(summary) # SyntaxError 수정 완료
                            st.caption("Data Source: Yahoo Finance")
                        else: st.warning("기업 기본 정보 로드 실패.")
                        # 3. 주요 재무 추세
                        st.subheader(f"📈 주요 재무 추세 (최근 {periods} 분기)")
                        tab_titles = ["영업이익률(%)", "ROE(%)", "부채비율", "유동비율"]
                        tabs = st.tabs(tab_titles)
                        trend_data_map = {
                            "영업이익률(%)": ('operating_margin_trend', 'Op Margin (%)', "{:.2f}%"),
                            "ROE(%)": ('roe_trend', 'ROE (%)', "{:.2f}%"),
                            "부채비율": ('debt_to_equity_trend', 'D/E Ratio', "{:.2f}"),
                            "유동비율": ('current_ratio_trend', 'Current Ratio', "{:.2f}")
                        }
                        for i, title in enumerate(tab_titles):
                                with tabs[i]:
                                    data_key, col_name, style_format = trend_data_map[title]
                                    trend_data = results.get(data_key)
                                    if trend_data and isinstance(trend_data, list) and len(trend_data) > 0:
                                        try:
                                            df_trend = pd.DataFrame(trend_data)
                                            df_trend['Date'] = pd.to_datetime(df_trend['Date'])
                                            df_trend.set_index('Date', inplace=True)
                                            # --- !!! 분석기가 지적한 부분 수정 !!! ---
                                            if col_name in df_trend.columns:
                                                st.line_chart(df_trend[[col_name]]) # 차트 먼저 그리고
                                                with st.expander("데이터 보기"): # expander 생성
                                                    st.dataframe(df_trend[[col_name]].style.format({col_name: style_format}), use_container_width=True) # 그 안에 dataframe 넣기
                                            # --- 수정 완료 ---
                                            else:
                                                st.error(f"'{col_name}' 컬럼 없음.")
                                        except Exception as e:
                                            st.error(f"{title} 표시 오류: {e}")
                                    else:
                                        st.info(f"{title} 추세 데이터 없음.")
                        st.divider()
                        # 4. 기술적 분석 차트 (종합)
                        st.subheader("기술적 분석 차트 (종합)")
                        stock_chart_fig = results.get('stock_chart_fig')
                        if stock_chart_fig:
                            st.plotly_chart(stock_chart_fig, use_container_width=True)
                        else:
                            st.warning("주가 차트 생성 실패 (종합).")
                            st.divider()
                        # 5. 시장 심리 분석
                        st.subheader("시장 심리 분석")
                        col_news, col_fng = st.columns([2, 1])
                        with col_news:
                            st.markdown("**📰 뉴스 감정 분석**")
                            news_sentiment = results.get('news_sentiment', ["정보 없음."])
                            if isinstance(news_sentiment, list) and len(news_sentiment) > 0:
                                st.info(news_sentiment[0])
                                if len(news_sentiment) > 1:
                                    with st.expander("뉴스 목록 보기"): # for 루프 사용 (V1.9.6 변경점)
                                        for line in news_sentiment[1:]:
                                            st.write(f"- {line}")
                            else:
                                st.write(str(news_sentiment))
                        with col_fng:
                            st.markdown("**😨 공포-탐욕 지수**")
                            fng_index = results.get('fear_greed_index', "N/A")
                            if isinstance(fng_index, dict):
                                st.metric("현재 지수", fng_index.get('value', 'N/A'), fng_index.get('classification', ''))
                            else:
                                st.write(fng_index)
                                st.divider() # 위치 조정 가능성 있음 (뉴스 컬럼과 분리)
                        # 6. Prophet 주가 예측
                        st.subheader("Prophet 주가 예측")
                        forecast_fig = results.get('forecast_fig')
                        forecast_data_list = results.get('prophet_forecast')
                        if forecast_fig:
                            st.plotly_chart(forecast_fig, use_container_width=True)
                        elif isinstance(forecast_data_list, str):
                            st.info(forecast_data_list)
                        else:
                            st.warning("예측 차트 생성 실패.")
                        if isinstance(forecast_data_list, list) and len(forecast_data_list) > 0:
                            st.markdown("**📊 예측 데이터 (최근 10일)**")
                            try:
                                df_fcst = pd.DataFrame(forecast_data_list)
                                df_fcst['ds'] = pd.to_datetime(df_fcst['ds']).dt.strftime('%Y-%m-%d')
                                st.dataframe(df_fcst[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(10).style.format({'yhat': "{:.2f}", 'yhat_lower': "{:.2f}", 'yhat_upper': "{:.2f}"}), use_container_width=True)
                            except Exception as e:
                                st.error(f"예측 데이터 표시 오류: {e}")
                        cv_plot_path = results.get('cv_plot_path')
                        if cv_plot_path and os.path.exists(cv_plot_path):
                            st.markdown("**📉 교차 검증 결과 (MAPE)**")
                            st.image(cv_plot_path, caption="MAPE (낮을수록 정확)")
                        elif cv_plot_path is None and isinstance(forecast_data_list, list):
                            st.caption("교차 검증(CV) 결과 없음.")
                            st.divider()
                        # 7. 리스크 트래커
                        st.subheader("🚨 리스크 트래커 (예측 기반)")
                        risk_days, max_loss_pct, max_loss_amt = 0, 0, 0
                        if avg_p > 0 and isinstance(forecast_data_list, list) and len(forecast_data_list) > 0:
                            try:
                                df_pred = pd.DataFrame(forecast_data_list)
                                required_fcst_cols = ['ds', 'yhat_lower']
                                if not all(col in df_pred.columns for col in required_fcst_cols):
                                    st.warning("예측 컬럼 부족.")
                                else:
                                    df_pred['ds'] = pd.to_datetime(df_pred['ds'])
                                    df_pred['yhat_lower'] = pd.to_numeric(df_pred['yhat_lower'], 'coerce')
                                    df_pred.dropna(subset=['yhat_lower'], inplace=True)
                                    if not df_pred.empty:
                                        df_pred['평단가'] = avg_p
                                        df_pred['리스크 여부'] = df_pred['yhat_lower'] < avg_p
                                        df_pred['예상 손실률'] = np.where(df_pred['리스크 여부'], ((df_pred['yhat_lower'] - avg_p) / avg_p) * 100, 0)
                                        if qty > 0:
                                            df_pred['예상 손실액'] = np.where(df_pred['리스크 여부'], (df_pred['yhat_lower'] - avg_p) * qty, 0)
                                        else:
                                            df_pred['예상 손실액'] = 0
                                        risk_days = df_pred['리스크 여부'].sum()
                                        if risk_days > 0:
                                            max_loss_pct = df_pred['예상 손실률'].min()
                                            max_loss_amt = df_pred['예상 손실액'].min() if qty > 0 else 0
                                        st.markdown("##### 리스크 요약")
                                        col_r1, col_r2, col_r3 = st.columns(3)
                                        col_r1.metric("⚠️ < 평단가 일수", f"{risk_days}일 / {days}일")
                                        col_r2.metric("📉 Max 손실률", f"{max_loss_pct:.2f}%")
                                        if qty > 0:
                                            col_r3.metric("💸 Max 손실액", f"${max_loss_amt:,.2f}")
                                        if risk_days > 0:
                                            st.warning(f"{days}일 예측 중 **{risk_days}일** 평단가(${avg_p:.2f}) 하회 가능성.")
                                        else:
                                            st.success(f"{days}일간 평단가(${avg_p:.2f}) 하회 가능성 낮음.")
                                        st.markdown("##### 평단가 vs 예측 구간 비교")
                                        fig_risk = go.Figure()
                                        if 'yhat_upper' in df_pred.columns:
                                            fig_risk.add_trace(go.Scatter(x=df_pred['ds'], y=df_pred['yhat_upper'], mode='lines', line_color='rgba(0,100,80,0.2)', name='Upper'))
                                        fig_risk.add_trace(go.Scatter(x=df_pred['ds'], y=df_pred['yhat_lower'], mode='lines', line_color='rgba(0,100,80,0.2)', name='Lower', fill='tonexty', fillcolor='rgba(0,100,80,0.1)'))
                                        if 'yhat' in df_pred.columns:
                                            fig_risk.add_trace(go.Scatter(x=df_pred['ds'], y=df_pred['yhat'], mode='lines', line=dict(dash='dash', color='rgba(0,100,80,0.6)'), name='Forecast'))
                                        fig_risk.add_hline(y=avg_p, line_dash="dot", line_color="red", annotation_text=f"평단가: ${avg_p:.2f}", annotation_position="bottom right")
                                        df_risk_periods = df_pred[df_pred['리스크 여부']]
                                        if not df_risk_periods.empty:
                                            fig_risk.add_trace(go.Scatter(x=df_risk_periods['ds'], y=df_risk_periods['yhat_lower'], mode='markers', marker_symbol='x', marker_color='red', name='Risk Day'))
                                        fig_risk.update_layout(hovermode="x unified")
                                        st.plotly_chart(fig_risk, use_container_width=True)
                                        if risk_days > 0:
                                            with st.expander(f"리스크 예측일 상세 데이터 ({risk_days}일)"):
                                                df_risk_days_display = df_pred[df_pred['리스크 여부']].copy()
                                                df_risk_days_display['ds'] = df_risk_days_display['ds'].dt.strftime('%Y-%m-%d')
                                                cols_show = ['ds', 'yhat_lower', '평단가', '예상 손실률']
                                                if qty > 0: cols_show.append('예상 손실액')
                                                st.dataframe(df_risk_days_display[cols_show].style.format({"yhat_lower":"{:.2f}", "평단가":"{:.2f}", "예상 손실률":"{:.2f}%", "예상 손실액":"${:,.2f}"}), use_container_width=True)
                                    else:
                                        st.info("예측 하한선 데이터 유효하지 않음.")
                            except Exception as risk_calc_err:
                                st.error(f"리스크 트래커 오류: {risk_calc_err}")
                                logging.error(f"Risk tracker error: {traceback.format_exc()}")
                        elif avg_p <= 0:
                            st.info("⬅️ '평단가' 입력 시 리스크 분석 결과 확인 가능.")
                        else:
                            st.warning("예측 데이터 유효하지 않아 리스크 분석 불가.")
                            st.divider()
                        # 8. 자동 분석 결과 요약
                        st.subheader("🧐 자동 분석 결과 요약 (참고용)")
                        summary_points = []
                        # (V1.9.5 요약 로직)
                        if isinstance(forecast_data_list, list) and len(forecast_data_list) > 0: # 예측
                            try:
                                start_pred_row = forecast_data_list[0]
                                end_pred_row = forecast_data_list[-1]
                                start_pred = pd.to_numeric(start_pred_row.get('yhat'), 'coerce')
                                end_pred = pd.to_numeric(end_pred_row.get('yhat'), 'coerce')
                                lower = pd.to_numeric(end_pred_row.get('yhat_lower'), 'coerce')
                                upper = pd.to_numeric(end_pred_row.get('yhat_upper'), 'coerce')
                            except:
                                start_pred, end_pred, lower, upper = None, None, None, None
                            if pd.notna(start_pred) and pd.notna(end_pred):
                                trend_obs = "상승" if end_pred > start_pred else "하락" if end_pred < start_pred else "횡보"
                                lower_str = f"${lower:.2f}" if pd.notna(lower) else 'N/A'
                                upper_str = f"${upper:.2f}" if pd.notna(upper) else 'N/A'
                                summary_points.append(f"- **예측:** 향후 {days}일간 **{trend_obs}** 추세 ({lower_str}~{upper_str}).")
                            else: summary_points.append("- 예측: 값 유효하지 않음.")
                        news_res = results.get('news_sentiment')
                        fng_res = results.get('fear_greed_index')
                        if isinstance(news_res, list) and len(news_res) > 0 and ":" in news_res[0]: # 뉴스
                            try:
                                score_part = news_res[0].split(":")[-1].strip()
                                avg_score = float(score_part)
                                sentiment_desc = "긍정적" if avg_score > 0.05 else "부정적" if avg_score < -0.05 else "중립적"
                                summary_points.append(f"- **뉴스:** 평균 감성 {avg_score:.2f}, **{sentiment_desc}** 분위기.")
                            except Exception as e:
                                logging.warning(f"뉴스 요약 오류: {e}")
                                summary_points.append("- 뉴스: 요약 오류.")
                        if isinstance(fng_res, dict):
                            summary_points.append(f"- **시장 심리:** 공포-탐욕 {fng_res.get('value', 'N/A')} (**{fng_res.get('classification', 'N/A')}**).")
                        if fundamentals and isinstance(fundamentals, dict): # 기본 정보
                            per = fundamentals.get("PER", "N/A")
                            sector = fundamentals.get("업종", "N/A")
                            parts = []
                            if per != "N/A": parts.append(f"PER {per}")
                            if sector != "N/A": parts.append(f"업종 '{sector}'")
                            if parts: summary_points.append(f"- **기본 정보:** {', '.join(parts)}.")
                        trend_parts = [] # 재무 추세
                        try:
                            op_margin_trend = results.get('operating_margin_trend')
                            roe_trend = results.get('roe_trend')
                            debt_trend = results.get('debt_to_equity_trend')
                            current_trend = results.get('current_ratio_trend')
                            if op_margin_trend and op_margin_trend: trend_parts.append(f"영업익률 {op_margin_trend[-1].get('Op Margin (%)', 'N/A'):.2f}%")
                            if roe_trend and roe_trend: trend_parts.append(f"ROE {roe_trend[-1].get('ROE (%)', 'N/A'):.2f}%")
                            if debt_trend and debt_trend: trend_parts.append(f"부채비율 {debt_trend[-1].get('D/E Ratio', 'N/A'):.2f}")
                            if current_trend and current_trend: trend_parts.append(f"유동비율 {current_trend[-1].get('Current Ratio', 'N/A'):.2f}")
                            if trend_parts: summary_points.append(f"- **최근 재무:** {', '.join(trend_parts)}.")
                        except Exception as e:
                            logging.warning(f"재무 추세 요약 오류: {e}")
                            summary_points.append("- 최근 재무: 요약 오류.")
                        if avg_p > 0 and isinstance(forecast_data_list, list) and len(forecast_data_list) > 0: # 리스크
                            if risk_days > 0:
                                summary_points.append(f"- **리스크:** {days}일 중 **{risk_days}일** 평단가 하회 가능성 (Max 손실률: **{max_loss_pct:.2f}%**).")
                            else:
                                summary_points.append(f"- **리스크:** 평단가(${avg_p:.2f}) 하회 가능성 낮음.")
                        elif avg_p > 0:
                            summary_points.append("- 리스크: 평단가 입력됨, 분석 불가.")
                        if summary_points:
                            st.markdown("\n".join(summary_points))
                            st.caption("⚠️ **주의:** 투자 조언 아님.")
                        else:
                            st.write("분석 요약 생성 불가.")
                        # === 상세 결과 표시 끝 ===
                    elif results and "error" in results:
                        results_placeholder.error(f"분석 실패: {results['error']}")
                    else:
                        results_placeholder.error("분석 결과 처리 중 오류.")
                except Exception as e:
                    error_traceback = traceback.format_exc()
                    logging.error(f"종합 분석 실행 오류: {e}\n{error_traceback}")
                    results_placeholder.error(f"앱 실행 오류: {e}")
                    st.exception(e)
    else: # 버튼 클릭 전
        if comprehensive_analysis_possible:
            results_placeholder.info("⬅️ 사이드바 설정 후 '종합 분석 시작' 버튼 클릭.")


# ============== 📈 기술 분석 탭 (V1.9.6 Fix Applied) ==============
elif page == "📈 기술 분석":
    st.title("📈 기술적 분석 (VWAP + Bollinger + Fibonacci)")
    st.markdown("VWAP, 볼린저밴드, 피보나치 되돌림 수준을 함께 시각화하고 자동 해석을 제공합니다.")
    st.markdown("---")
    ticker_tech = st.text_input("종목 티커", "AAPL", key="tech_ticker", help="해외(예: AAPL) 또는 국내(예: 005930.KS) 티커")
    today = datetime.now().date()
    default_start_date = today - relativedelta(months=3)
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
                    period_days = (end_date - start_date).days
                    fetch_start_date = start_date
                    if interval == '1m' and period_days > 7:
                        st.warning("1분봉 최대 7일 조회. 시작일 조정.")
                        fetch_start_date = end_date - timedelta(days=7)
                    elif interval in ['5m', '15m', '30m'] and period_days > 60:
                        st.warning(f"{interval_display} 최대 60일 조회. 시작일 조정.")
                        fetch_start_date = end_date - timedelta(days=60)
                    fetch_end_date = end_date + timedelta(days=1)
                    logging.info(f"yf 다운로드 요청: {ticker_processed_tech}, {fetch_start_date}, {fetch_end_date}, {interval}")
                    df_tech = yf.download(ticker_processed_tech, start=fetch_start_date, end=fetch_end_date, interval=interval, progress=False)
                    if not df_tech.empty: df_tech.attrs['ticker'] = ticker_processed_tech
                except Exception as yf_err: st.error(f"yfinance 다운로드 오류: {yf_err}")

            analysis_successful = False # 분석 성공 플래그
            df_calculated = pd.DataFrame() # 계산 결과 저장할 DF 초기화

            if not df_tech.empty:
                logging.info(f"다운로드 완료. 행: {len(df_tech)}")
                st.caption(f"조회 기간: {df_tech.index.min():%Y-%m-%d %H:%M} ~ {df_tech.index.max():%Y-%m-%d %H:%M}")
                if isinstance(df_tech.columns, pd.MultiIndex): # 멀티인덱스 처리
                    logging.info("MultiIndex 컬럼 감지됨. Flattening 시도...")
                    original_columns = df_tech.columns
                    df_tech.columns = df_tech.columns.get_level_values(0)
                    logging.info(f"컬럼 변환 완료: {original_columns.tolist()} -> {df_tech.columns.tolist()}")
                    df_tech = df_tech.loc[:,~df_tech.columns.duplicated()]
                    logging.info(f"중복 제거 후 컬럼: {df_tech.columns.tolist()}")
                required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                missing_cols = [col for col in required_cols if col not in df_tech.columns]
                if missing_cols:
                    st.error(f"❌ 데이터에 필수 컬럼 누락: {missing_cols}. 실제 컬럼: {df_tech.columns.tolist()}")
                    st.dataframe(df_tech.head())
                else: analysis_successful = True # 컬럼 있으면 성공
            elif analyze_button_tech: st.error(f"❌ 데이터 로드 실패.")

            if analysis_successful:
                with st.spinner("기술적 지표 계산 및 차트 생성 중..."):
                    try:
                        df_processed = df_tech.dropna(subset=required_cols).copy()
                        if df_processed.empty: st.warning("데이터 정제 후 남은 데이터가 없습니다.")
                        else:
                            df_calculated = df_processed.copy()
                            try: df_calculated = calculate_vwap(df_calculated)
                            except ValueError as ve_vwap: st.warning(f"VWAP 계산 불가: {ve_vwap}")
                            try: df_calculated = calculate_bollinger_bands(df_calculated, window=bb_window_val, num_std=bb_std_val)
                            except ValueError as ve_bb: st.warning(f"볼린저 밴드 계산 불가: {ve_bb}")
                            df_calculated = calculate_rsi(df_calculated)
                            df_calculated = calculate_macd(df_calculated)

                            st.subheader(f"📌 {ticker_processed_tech} 기술적 분석 통합 차트 ({interval_display})")
                            chart_tech = plot_technical_chart(df_calculated, ticker_processed_tech)
                            st.plotly_chart(chart_tech, use_container_width=True)

                            st.subheader("📄 최근 데이터 (지표 포함)")
                            display_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'VWAP', 'MA20', 'Upper', 'Lower']
                            display_cols = [col for col in display_cols if col in df_calculated.columns]
                            format_dict = {col: "{:.2f}" for col in display_cols if col != 'Volume'}
                            st.dataframe(df_calculated[display_cols].tail(10).style.format(format_dict), use_container_width=True)

                            # --- 자동 해석 기능 ---
                            st.divider()
                            st.subheader("🧠 기술적 시그널 해석 (참고용)")

                            if not df_calculated.empty:
                                latest_row = df_calculated.iloc[-1]
                                signal_messages = interpret_technical_signals(latest_row)  # 🔁 VWAP, BB 해석 불러오기

                                # ✅ 피보나치 해석 추가
                                fib_msg = interpret_fibonacci(df_calculated, close_value=latest_row["Close"])
                                if fib_msg:
                                    signal_messages.append(fib_msg)

                                if signal_messages:
                                    for msg in signal_messages:
                                        st.info(msg)
                                else:
                                    st.info("특별한 기술적 시그널은 감지되지 않았습니다.")

                                st.caption("⚠️ **주의:** 자동 해석은 참고용이며, 투자 결정은 종합 판단 하에 신중히 하세요.")
                            else:
                                st.warning("해석할 데이터가 없습니다.")

                    except Exception as e: # 예상 못한 오류
                        st.error(f"기술적 분석 처리 중 오류: {type(e).__name__} - {e}")
                        logging.error(f"Technical analysis error: {traceback.format_exc()}")
                        st.dataframe(df_tech.head()) # 원본 표시

    else: # 버튼 클릭 전
        st.info("종목 티커, 기간, 간격 설정 후 '기술적 분석 실행' 버튼 클릭.")

# --- 앱 정보 ---
st.sidebar.markdown("---")
st.sidebar.info("종합 주식 분석 툴 V1.9.6 | 정보 제공 목적 (투자 조언 아님)") # 버전 정보 최종 업데이트
st.sidebar.markdown("📌 [개발기 보러가기](https://technut.tistory.com/1)", unsafe_allow_html=True)
st.sidebar.caption("👨‍💻 기술 기반 주식 분석 툴 개발기")