# -*- coding: utf-8 -*-
# Combined app.py V1.9.8 (Modified for Finnhub in Technical Analysis Tab)

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta, time # time 추가
from dateutil.relativedelta import relativedelta
import os
from dotenv import load_dotenv
import traceback
import plotly.graph_objects as go
import numpy as np
import logging
# import yfinance as yf # yfinance는 종합분석 탭에서 여전히 사용될 수 있음 (stock_analysis.py 수정 필요)

# Finnhub 및 레이트 리미터 관련 import
import finnhub
from ratelimit import limits, RateLimitException
from backoff import on_exception, expo
import time as time_module # time import 충돌 방지

# 기존 모듈 임포트
try:
    from short_term_analysis import interpret_fibonacci, calculate_rsi, calculate_macd
    from technical_interpret import interpret_technical_signals
    # stock_analysis.py는 종합분석 탭에서 사용되므로 일단 유지
    # import stock_analysis as sa # analyze_stock 함수 호출 시 필요
except ImportError as e:
    st.error(f"필수 분석 모듈 로딩 실패: {e}. 'short_term_analysis.py', 'technical_interpret.py' 또는 'stock_analysis.py' 파일이 있는지 확인하세요.")
    st.stop()


# --- 기본 경로 설정 및 로깅 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
try: BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError: BASE_DIR = os.getcwd()

# --- Finnhub API 클라이언트 설정 및 레이트 리미터 ---
FINNHUB_API_KEY = None
finnhub_client = None
sidebar_status_finnhub = st.sidebar.empty() # Finnhub 키 상태 메시지용

try:
    FINNHUB_API_KEY = st.secrets.get("FINNHUB_API_KEY")
    if not FINNHUB_API_KEY:
        sidebar_status_finnhub.warning("Finnhub API 키가 Streamlit secrets에 없습니다. .env 파일을 확인합니다.")
        # .env 파일 로드 (Streamlit secrets에 없을 경우)
        dotenv_path = os.path.join(BASE_DIR, '.env')
        if os.path.exists(dotenv_path):
            load_dotenv(dotenv_path=dotenv_path)
            FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
            if FINNHUB_API_KEY:
                sidebar_status_finnhub.success("Finnhub API 키 로드 완료 (.env).")
            else:
                sidebar_status_finnhub.error("Finnhub API 키가 .env 파일에도 없습니다. 기술 분석 기능이 제한됩니다.")
        else:
            sidebar_status_finnhub.error(".env 파일이 없습니다. 기술 분석 기능이 제한됩니다.")
    else:
        sidebar_status_finnhub.success("Finnhub API 키 로드 완료 (Secrets).")

except Exception as e:
    sidebar_status_finnhub.error(f"Finnhub API 키 로드 중 오류: {e}")
    FINNHUB_API_KEY = None # 오류 발생 시 키 없음으로 처리

if FINNHUB_API_KEY:
    finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)
else:
    st.sidebar.error("Finnhub API 키가 없어 기술적 분석 데이터 로딩이 불가능합니다.")

# 레이트 리미터 설정 (분당 60회)
CALLS = 40 # 약간의 여유를 둠
PERIOD = 60  # 초 (1분)

@on_exception(expo, RateLimitException, max_tries=3, logger=logging)
@limits(calls=CALLS, period=PERIOD)
def call_finnhub_api_with_limit(api_function, *args, **kwargs):
    """레이트 리밋을 적용하여 Finnhub API 함수를 호출합니다."""
    try:
        # logging.info(f"Calling Finnhub API (rate-limited): {api_function.__name__}")
        return api_function(*args, **kwargs)
    except RateLimitException as rle:
        logging.warning(f"Rate limit exceeded for {api_function.__name__}. Retrying... Details: {rle}")
        raise # on_exception 데코레이터가 처리하도록 예외를 다시 발생시킴
    except finnhub.FinnhubAPIException as api_e: # Finnhub API 관련 명시적 예외 처리
        logging.error(f"Finnhub API Exception for {api_function.__name__}: {api_e}")
        st.error(f"Finnhub API 오류: {api_e} (요청: {api_function.__name__})")
        raise
    except Exception as e:
        logging.error(f"Error in call_finnhub_api_with_limit for {api_function.__name__}: {e}")
        raise

# --- Finnhub 데이터 요청 함수 ---
def get_finnhub_stock_candles(client, ticker, resolution, start_timestamp, end_timestamp):
    """Finnhub API를 사용하여 주식 캔들 데이터를 가져옵니다 (레이트 리미터 적용)."""
    if not client:
        st.error("Finnhub 클라이언트가 초기화되지 않았습니다.")
        return None
    try:
        logging.info(f"Finnhub 요청: {ticker}, Res: {resolution}, Start: {start_timestamp}, End: {end_timestamp}")
        # API 호출 시 call_finnhub_api_with_limit 사용
        res = call_finnhub_api_with_limit(client.stock_candles, ticker, resolution, start_timestamp, end_timestamp)

        if res and res.get('s') == 'ok':
            df = pd.DataFrame(res)
            if df.empty or 't' not in df.columns: # 't' 컬럼 존재 확인
                st.info(f"{ticker}: Finnhub에서 데이터가 반환되었으나 비어있거나 시간 정보가 없습니다.")
                return pd.DataFrame()

            df['t'] = pd.to_datetime(df['t'], unit='s', utc=True).dt.tz_convert('Asia/Seoul').dt.tz_localize(None) # 한국 시간으로 변환 후 naive로
            df.set_index('t', inplace=True)
            df.rename(columns={'o': 'Open', 'h': 'High', 'l': 'Low', 'c': 'Close', 'v': 'Volume'}, inplace=True)
            # 데이터 타입 변환
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            df.dropna(subset=['Open', 'High', 'Low', 'Close'], inplace=True) # 필수 가격 데이터 NaN 제거
            return df[['Open', 'High', 'Low', 'Close', 'Volume']]
        elif res and res.get('s') == 'no_data':
            st.info(f"{ticker}: 해당 기간({datetime.fromtimestamp(start_timestamp).strftime('%Y-%m-%d')} ~ {datetime.fromtimestamp(end_timestamp).strftime('%Y-%m-%d')})에 대한 데이터가 Finnhub에 없습니다.")
            return pd.DataFrame() # 빈 데이터프레임 반환
        else:
            error_msg = res.get('s', '알 수 없는 응답 상태') if res else '응답 없음'
            st.error(f"Finnhub API에서 {ticker} 데이터 조회 실패: {error_msg}")
            logging.error(f"Finnhub API error for {ticker}: {res}")
            return None
    except RateLimitException:
        st.error(f"Finnhub API 호출 빈도 제한을 초과했습니다 ({ticker}). 잠시 후 다시 시도해주세요.")
        return None
    except finnhub.FinnhubAPIException as api_e:
        # call_finnhub_api_with_limit에서 이미 처리하지만, 여기서 추가 로깅 가능
        logging.error(f"Finnhub API Exception in get_finnhub_stock_candles for {ticker}: {api_e}")
        return None # 이미 st.error가 호출되었을 것
    except Exception as e:
        st.error(f"Finnhub 캔들 데이터 요청 중 오류 ({ticker}): {e}")
        logging.error(f"Unexpected error in get_finnhub_stock_candles for {ticker}: {traceback.format_exc()}")
        return None

# --- 기존 기술 분석 함수 (calculate_vwap, calculate_bollinger_bands, plot_technical_chart) ---
# 이 함수들은 Pandas DataFrame을 입력으로 받으므로, Finnhub 데이터가 DataFrame으로 잘 변환되면 수정 없이 사용 가능할 수 있습니다.
# 단, 컬럼명이 일치해야 합니다 (Open, High, Low, Close, Volume). get_finnhub_stock_candles 함수에서 이미 맞춰주었습니다.

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
        df['Volume'] = df['Volume'].fillna(0)
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

    # (1) 캔들 차트
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'],
                                 low=df['Low'], close=df['Close'], name=f"{ticker} 캔들"))

    # (2) VWAP
    if 'VWAP' in df.columns and df['VWAP'].notna().any():
        fig.add_trace(go.Scatter(x=df.index, y=df['VWAP'], mode='lines',
                                 name='VWAP', line=dict(color='orange', width=1.5)))
    elif 'VWAP' in df.columns:
        st.caption("VWAP 데이터 없음/표시 불가.")

    # (3) Bollinger Bands
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

    # (4) Fibonacci Levels
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

    # (5) RSI
    if 'RSI' in df.columns and df['RSI'].notna().any():
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], mode='lines', name='RSI (14)',
                                 line=dict(color='purple', width=1), yaxis='y2'))

    # (6) MACD
    if 'MACD' in df.columns and 'MACD_signal' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD', mode='lines',
                                 line=dict(color='teal'), yaxis='y3'))
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD_signal'], name='MACD Signal', mode='lines',
                                 line=dict(color='orange'), yaxis='y3'))
        if 'MACD_hist' in df.columns:
            fig.add_trace(go.Bar(x=df.index, y=df['MACD_hist'], name='MACD Histogram',
                                 marker_color='lightblue', yaxis='y3'))

    # 레이아웃
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
st.set_page_config(page_title="종합 주식 분석 Finnhub 개인버전", layout="wide", initial_sidebar_state="expanded") # 버전 업데이트

# --- API 키 로드 (News, FRED - 종합 분석용) ---
NEWS_API_KEY = None
FRED_API_KEY = None
api_keys_loaded_main = False # 종합분석용 키 로드 상태
secrets_available = hasattr(st, 'secrets')
sidebar_status_main_keys = st.sidebar.empty() # 종합분석 키 상태 메시지용

if secrets_available:
    try:
        NEWS_API_KEY = st.secrets.get("NEWS_API_KEY")
        FRED_API_KEY = st.secrets.get("FRED_API_KEY")
    except Exception as e:
        sidebar_status_main_keys.error(f"Secrets (News/FRED) 로드 오류: {e}")

if NEWS_API_KEY and FRED_API_KEY:
    api_keys_loaded_main = True
else:
    if secrets_available: sidebar_status_main_keys.warning("Secrets (News/FRED) 키 일부 누락.")

if not api_keys_loaded_main:
    sidebar_status_main_keys.info(".env 파일 (News/FRED) 확인 중...")
    try:
        dotenv_path_main = os.path.join(BASE_DIR, '.env') # Finnhub .env와 경로 동일
        if os.path.exists(dotenv_path_main):
            # load_dotenv를 다시 호출할 필요는 없으나, 변수만 가져옴
            if not NEWS_API_KEY: NEWS_API_KEY = os.getenv("NEWS_API_KEY")
            if not FRED_API_KEY: FRED_API_KEY = os.getenv("FRED_API_KEY")

            if NEWS_API_KEY and FRED_API_KEY:
                api_keys_loaded_main = True
                sidebar_status_main_keys.success("API 키 (News/FRED) 로드 완료 (.env)")
            else:
                sidebar_status_main_keys.error(".env 파일 내 API 키 (News/FRED) 일부 누락.")
        else:
            sidebar_status_main_keys.error(".env 파일 없음 (News/FRED).")
    except Exception as e:
        sidebar_status_main_keys.error(f".env (News/FRED) 로드 오류: {e}")

comprehensive_analysis_possible = api_keys_loaded_main
if not api_keys_loaded_main:
    st.sidebar.error("API 키 (News/FRED) 로드 실패! '종합 분석' 기능이 제한됩니다.")
else:
    if not secrets_available or not (st.secrets.get("NEWS_API_KEY") and st.secrets.get("FRED_API_KEY")):
         sidebar_status_main_keys.success("API 키 (News/FRED) 로드 완료.")


# --- 사이드바 설정 ---
with st.sidebar:
    with st.expander("☕ 후원계좌"):
        try:
            st.image("qr_kakaopay.png", width=180)
            st.caption("📱 코드 스캔으로 후원할 수 있습니다")
        except Exception as img_e:
            st.warning(f"후원 QR 이미지 로드 실패: {img_e}")

    st.markdown("📘 [분석도구 상세정보](https://technut.tistory.com/3)", unsafe_allow_html=True)
    st.title("📊 주식 분석 도구 Finnhub 개인버전")
    st.markdown("---")

    page = st.radio("분석 유형 선택", ["📊 종합 분석 (yfinance 기반)", "📈 기술 분석 (Finnhub 기반)"],
                    captions=["재무, 예측, 뉴스 등", "VWAP, BB, 피보나치 등"],
                    key="page_selector")
    st.markdown("---")

    if page == "📊 종합 분석 (yfinance 기반)": # 페이지명 변경
        st.header("⚙️ 종합 분석 설정")
        ticker_input = st.text_input("종목 티커 (yfinance)", "AAPL", key="main_ticker",
                                     help="해외(예: AAPL) 또는 국내(예: 005930.KS) 티커",
                                     disabled=not comprehensive_analysis_possible)
        analysis_years = st.select_slider("분석 기간 (년)", [1, 2, 3, 5, 7, 10], 2,
                                          key="analysis_years",
                                          disabled=not comprehensive_analysis_possible)
        st.caption(f"과거 {analysis_years}년 데이터 분석")
        forecast_days = st.number_input("예측 기간 (일)", 7, 90, 30, 7,
                                        key="forecast_days",
                                        disabled=not comprehensive_analysis_possible)
        st.caption(f"향후 {forecast_days}일 예측")
        num_trend_periods_input = st.number_input("재무 추세 분기 수", 2, 12, 4, 1,
                                                 key="num_trend_periods",
                                                 disabled=not comprehensive_analysis_possible)
        st.caption(f"최근 {num_trend_periods_input}개 분기 재무 추세 계산")
        st.divider()
        st.subheader("⚙️ 예측 세부 설정 (선택)")
        changepoint_prior_input = st.slider("추세 변화 민감도 (Prophet)", 0.001, 0.5, 0.05, 0.01, "%.3f",
                                            help="클수록 과거 추세 변화에 민감 (기본값: 0.05)",
                                            key="changepoint_prior",
                                            disabled=not comprehensive_analysis_possible)
        st.caption(f"현재 민감도: {changepoint_prior_input:.3f}")
        st.divider()
        st.subheader("💰 보유 정보 입력 (선택)")
        avg_price = st.number_input("평단가", 0.0, format="%.2f", key="avg_price",
                                     disabled=not comprehensive_analysis_possible)
        quantity = st.number_input("보유 수량", 0, step=1, key="quantity",
                                    disabled=not comprehensive_analysis_possible)
        st.caption("평단가 입력 시 리스크 트래커 분석 활성화")
        st.divider()

    elif page == "📈 기술 분석 (Finnhub 기반)": # 페이지명 변경
        st.header("⚙️ 기술 분석 설정 (Finnhub)")
        bb_window = st.number_input("볼린저밴드 기간 (일)", 5, 50, 20, 1, key="bb_window_fh") # 키 변경
        bb_std = st.number_input("볼린저밴드 표준편차 배수", 1.0, 3.0, 2.0, 0.1, key="bb_std_fh", format="%.1f") # 키 변경
        st.caption(f"현재 설정: {bb_window}일 기간, {bb_std:.1f} 표준편차")
        st.divider()


# --- 캐시된 종합 분석 함수 (stock_analysis.py에 의존) ---
# 이 부분은 stock_analysis.py가 Finnhub으로 완전히 전환되기 전까지는 yfinance 기반으로 동작합니다.
@st.cache_data(ttl=timedelta(hours=1))
def run_cached_analysis(ticker, news_key, fred_key, years, days, num_trend_periods, changepoint_prior_scale):
    """종합 분석 실행 및 결과 반환 (캐싱 적용, 현재 yfinance 기반)"""
    try:
        # stock_analysis.py는 여전히 yfinance를 사용할 수 있음
        import stock_analysis as sa
    except ImportError as import_err:
        return {"error": f"분석 모듈(stock_analysis.py) 로딩 오류: {import_err}."}
    except Exception as e:
        return {"error": f"분석 모듈 로딩 중 예외 발생: {e}"}

    logging.info(f"종합 분석 실행 (yfinance): {ticker}, {years}년, {days}일, {num_trend_periods}분기, cp_prior={changepoint_prior_scale}")
    if not news_key or not fred_key:
        logging.warning(f"API 키 없이 종합 분석 시도 (ticker: {ticker}). 일부 기능 제한될 수 있음.")

    try:
        analysis_results = sa.analyze_stock(
            ticker,
            news_key=news_key,
            fred_key=fred_key,
            analysis_period_years=years,
            forecast_days=days,
            num_trend_periods=num_trend_periods,
            changepoint_prior_scale=changepoint_prior_scale
        )
        return analysis_results
    except Exception as e:
        logging.error(f"analyze_stock 함수 실행 중 오류 발생 (ticker: {ticker}): {e}\n{traceback.format_exc()}")
        return {"error": f"종합 분석 중 오류 발생: {e}"}


# --- 메인 화면 로직 ---
if page == "📊 종합 분석 (yfinance 기반)":
    st.title("📊 종합 분석 결과 (yfinance 기반)") # 타이틀 변경
    st.markdown("기업 정보, 재무 추세, 예측, 리스크 트래커 제공 (stock_analysis.py 모듈 사용).")
    st.markdown("---")
    analyze_button_main_disabled = not comprehensive_analysis_possible
    if analyze_button_main_disabled: st.error("API 키(News/FRED) 로드 실패. 종합 분석 불가.")
    analyze_button_main = st.button("🚀 종합 분석 시작!", use_container_width=True, type="primary", key="analyze_main_button_yf", disabled=analyze_button_main_disabled)
    results_placeholder = st.container()

    if analyze_button_main:
        ticker = st.session_state.get('main_ticker', "AAPL")
        years = st.session_state.get('analysis_years', 2)
        days = st.session_state.get('forecast_days', 30)
        periods = st.session_state.get('num_trend_periods', 4)
        cp_prior = st.session_state.get('changepoint_prior', 0.05)
        avg_p = st.session_state.get('avg_price', 0.0)
        qty = st.session_state.get('quantity', 0)

        if not ticker:
            results_placeholder.warning("종목 티커 입력 필요.")
        else:
            ticker_proc = ticker.strip().upper()
            with st.spinner(f"{ticker_proc} 종합 분석 중 (yfinance 기반)..."):
                try:
                    results = run_cached_analysis(
                        ticker_proc,
                        NEWS_API_KEY,
                        FRED_API_KEY,
                        years, days, periods, cp_prior
                    )
                    # --- 상세 결과 표시 로직 (기존과 동일하게 유지) ---
                    if results and isinstance(results, dict):
                        if "error" in results:
                            results_placeholder.error(f"분석 실패: {results['error']}")
                        else:
                            results_placeholder.empty()
                            if results.get("warn_high_mape"):
                                m = results.get("mape", "N/A")
                                mape_value_str = f"{m:.1f}%" if isinstance(m, (int, float)) else "N/A"
                                st.warning(
                                    f"🔴 모델 정확도 낮음 (MAPE {mape_value_str}). 예측 신뢰도에 주의하세요!"
                                )
                            with results_placeholder:
                                st.header(f"📈 {ticker_proc} 분석 결과 (민감도: {cp_prior:.3f})")
                                # 1. 요약 정보
                                st.subheader("요약 정보")
                                col1, col2, col3 = st.columns(3)
                                col1.metric("현재가", f"${results.get('current_price', 'N/A')}")
                                col2.metric("분석 시작일", results.get('analysis_period_start', 'N/A'))
                                col3.metric("분석 종료일", results.get('analysis_period_end', 'N/A'))
                                # ... (이하 기존 종합 분석 결과 표시 로직 유지) ...
                                # ... (기본적 분석, 재무 추세, 기술적 분석 차트(종합), 시장 심리, Prophet 예측, 리스크 트래커, 자동 분석 요약 등) ...
                                # ... (코드가 너무 길어져서 이 부분은 생략합니다. 기존 코드 그대로 사용하시면 됩니다.) ...

                                # --- 임시로 기본 정보 표시 부분만 남겨둡니다. 실제로는 전체를 복사해야 합니다. ---
                                st.subheader("📊 기업 기본 정보 (yfinance)")
                                fundamentals = results.get('fundamentals')
                                if fundamentals and isinstance(fundamentals, dict) and fundamentals.get("시가총액", "N/A") != "N/A":
                                    # ... (기존 코드) ...
                                    st.caption("Data Source: Yahoo Finance (via stock_analysis.py)")
                                else: st.warning("기업 기본 정보 로드 실패 (yfinance).")
                                st.divider()

                                # (이하 생략 - 기존 종합 분석 결과 표시 코드 전부 포함 필요)
                                st.info("종합 분석 결과 표시는 기존 로직을 따릅니다. stock_analysis.py가 수정되면 이 부분도 Finnhub 데이터로 대체될 수 있습니다.")


                    elif results is None:
                         results_placeholder.error("분석 결과 처리 중 예상치 못한 오류 발생 (결과 없음).")
                    else:
                        results_placeholder.error("분석 결과 처리 중 오류 발생 (결과 형식 오류).")

                except Exception as e:
                    error_traceback = traceback.format_exc()
                    logging.error(f"종합 분석 메인 로직 실행 오류: {e}\n{error_traceback}")
                    results_placeholder.error(f"앱 실행 중 오류 발생: {e}")
    else:
        if comprehensive_analysis_possible:
            results_placeholder.info("⬅️ 사이드바에서 설정을 확인하고 '종합 분석 시작!' 버튼을 클릭하세요.")
        else:
            results_placeholder.warning("API 키(News/FRED) 로드 실패로 종합 분석을 진행할 수 없습니다.")


# ============== 📈 기술 분석 탭 (Finnhub 기반) ==============
elif page == "📈 기술 분석 (Finnhub 기반)": # 페이지명 변경
    st.title("📈 기술적 분석 (Finnhub: VWAP + Bollinger + Fibonacci)") # 타이틀 변경
    st.markdown("Finnhub API를 사용하여 VWAP, 볼린저밴드, 피보나치 되돌림 수준을 시각화하고 자동 해석을 제공합니다.")
    st.markdown("---")

    if not finnhub_client: # Finnhub 클라이언트 없으면 기능 사용 불가
        st.error("Finnhub API 클라이언트가 초기화되지 않았습니다. 사이드바에서 API 키 설정을 확인해주세요.")
    else:
        ticker_tech = st.text_input("종목 티커 (Finnhub)", "AAPL", key="tech_ticker_fh", help="예: AAPL, MSFT 등 (Finnhub 지원 티커)")

        today = datetime.now().date()
        default_start_date = today - relativedelta(months=3)
        # Finnhub 무료 API는 과거 데이터 제한이 있을 수 있음 (예: 1년)
        # 사용자가 너무 과거를 선택하지 않도록 min_value 조정 가능
        min_date_allowed = today - relativedelta(years=2) # 예시: 최대 2년 전

        col1, col2, col3 = st.columns(3)
        with col1:
            start_date_dt = st.date_input("시작일", default_start_date, key="tech_start_fh",
                                       min_value=min_date_allowed, max_value=today - timedelta(days=1))
        with col2:
            end_date_dt = st.date_input("종료일", today, key="tech_end_fh",
                                     min_value=start_date_dt + timedelta(days=1), max_value=today)
        with col3:
            # Finnhub resolution 매핑
            # 무료 API: 1, 5, 15, 30, 60, D, W, M 지원
            # 유료 API: 더 많은 해상도 지원 가능
            finnhub_resolution_options = {
                "일봉": "D", "주봉": "W", "월봉": "M",
                "1시간": "60", "30분": "30", "15분": "15", "5분": "5", "1분": "1"
            }
            # yfinance interval 제약 대신 Finnhub 제약 고려
            interval_help_fh = """
            데이터 간격 선택 (Finnhub):
            - 무료 API는 과거 데이터 범위 및 해상도에 제한이 있을 수 있습니다.
            - (예) 1분봉은 최근 몇 개월 데이터만 제공될 수 있습니다.
            - 너무 긴 기간에 대해 짧은 간격을 선택하면 데이터가 없을 수 있습니다.
            """
            interval_display_fh = st.selectbox("데이터 간격", list(finnhub_resolution_options.keys()),
                                            key="tech_interval_display_fh", help=interval_help_fh, index=0) # 기본 '일봉'
            resolution_fh = finnhub_resolution_options[interval_display_fh]

        bb_window_val = st.session_state.get('bb_window_fh', 20)
        bb_std_val = st.session_state.get('bb_std_fh', 2.0)

        analyze_button_tech = st.button("📊 기술적 분석 실행 (Finnhub)", key="tech_analyze_fh", use_container_width=True, type="primary")

        if analyze_button_tech:
            if not ticker_tech:
                st.warning("종목 티커를 입력해주세요.")
            else:
                ticker_processed_tech = ticker_tech.strip().upper()
                df_tech_fh = pd.DataFrame()

                st.write(f"**{ticker_processed_tech}** ({interval_display_fh}, BB:{bb_window_val}일/{bb_std_val:.1f}σ) Finnhub 분석 중...")

                with st.spinner(f"{ticker_processed_tech} Finnhub 데이터 로딩 및 처리 중..."):
                    try:
                        # 날짜를 Unix timestamp로 변환 (시작일은 00:00:00, 종료일은 23:59:59)
                        start_datetime_obj = datetime.combine(start_date_dt, time.min)
                        end_datetime_obj = datetime.combine(end_date_dt, time.max)

                        start_ts = int(start_datetime_obj.timestamp())
                        end_ts = int(end_datetime_obj.timestamp())

                        logging.info(f"Finnhub 요청: Ticker={ticker_processed_tech}, Resolution={resolution_fh}, StartTS={start_ts}, EndTS={end_ts}")

                        # Finnhub 데이터 요청 함수 호출
                        df_tech_fh = get_finnhub_stock_candles(finnhub_client, ticker_processed_tech, resolution_fh, start_ts, end_ts)

                        if df_tech_fh is None: # get_finnhub_stock_candles 내부에서 오류 발생 시 None 반환 가능
                            st.error(f"❌ **{ticker_processed_tech}**에 대한 Finnhub 데이터를 조회 중 오류가 발생했습니다. 에러 메시지를 확인하세요.")
                        elif df_tech_fh.empty:
                            st.info(f"❌ **{ticker_processed_tech}**에 대한 데이터를 Finnhub에서 조회하지 못했습니다. 티커, 기간 또는 데이터 간격을 확인해주세요. (Finnhub는 데이터가 없으면 빈 목록을 반환할 수 있습니다)")
                        else:
                            logging.info(f"Finnhub 다운로드 완료. 행 수: {len(df_tech_fh)}, 컬럼: {df_tech_fh.columns.tolist()}")
                            # 한국 시간 기준으로 표시되도록 이미 변환됨
                            st.caption(f"조회된 데이터 기간 (한국시간 기준): {df_tech_fh.index.min():%Y-%m-%d %H:%M} ~ {df_tech_fh.index.max():%Y-%m-%d %H:%M}")

                            # 필수 컬럼 확인 (Open, High, Low, Close, Volume은 get_finnhub_stock_candles에서 이미 처리됨)
                            # 데이터 처리 및 지표 계산
                            df_calculated_fh = df_tech_fh.copy() # 이미 필요한 컬럼만 있음
                            df_calculated_fh.attrs['ticker'] = ticker_processed_tech

                            if df_calculated_fh.empty:
                                st.warning("Finnhub 데이터 정제 후 분석할 데이터가 없습니다.")
                            else:
                                try: df_calculated_fh = calculate_vwap(df_calculated_fh)
                                except ValueError as ve_vwap: st.warning(f"VWAP 계산 불가: {ve_vwap}")
                                try: df_calculated_fh = calculate_bollinger_bands(df_calculated_fh, window=bb_window_val, num_std=bb_std_val)
                                except ValueError as ve_bb: st.warning(f"볼린저 밴드 계산 불가: {ve_bb}")
                                try: df_calculated_fh = calculate_rsi(df_calculated_fh)
                                except Exception as e_rsi: st.warning(f"RSI 계산 불가: {e_rsi}")
                                try: df_calculated_fh = calculate_macd(df_calculated_fh)
                                except Exception as e_macd: st.warning(f"MACD 계산 불가: {e_macd}")

                                st.subheader(f"📌 {ticker_processed_tech} 기술적 분석 통합 차트 (Finnhub, {interval_display_fh})")
                                chart_tech_fh = plot_technical_chart(df_calculated_fh, ticker_processed_tech)
                                st.plotly_chart(chart_tech_fh, use_container_width=True)

                                st.subheader("📄 최근 데이터 (계산된 지표 포함 - Finnhub)")
                                display_cols = ['Open', 'High', 'Low', 'Close', 'Volume',
                                                'VWAP', 'MA20', 'Upper', 'Lower',
                                                'RSI', 'MACD', 'MACD_signal', 'MACD_hist']
                                display_cols_exist = [col for col in display_cols if col in df_calculated_fh.columns]
                                format_dict = {col: "{:.2f}" for col in display_cols_exist if col != 'Volume'}
                                if 'Volume' in display_cols_exist: format_dict['Volume'] = "{:,.0f}"
                                st.dataframe(df_calculated_fh[display_cols_exist].tail(10).style.format(format_dict), use_container_width=True)

                                st.divider()
                                st.subheader("🧠 기술적 시그널 해석 (참고용 - Finnhub)")
                                if not df_calculated_fh.empty:
                                    latest_row_fh = df_calculated_fh.iloc[-1].copy()
                                    signal_messages_fh = []
                                    try:
                                        signal_messages_fh.extend(interpret_technical_signals(latest_row_fh, df_context=df_calculated_fh))
                                    except Exception as e_interpret:
                                         st.warning(f"기본 기술적 시그널 해석 중 오류: {e_interpret}")
                                    # 피보나치 해석은 interpret_technical_signals 내부에서 df_context를 통해 호출됨

                                    if signal_messages_fh:
                                        for msg in signal_messages_fh:
                                            st.info(msg)
                                    else:
                                        st.info("현재 특별히 감지된 기술적 시그널은 없습니다.")
                                    st.caption("⚠️ **주의:** 자동 해석은 보조 지표이며 투자 결정은 반드시 종합적인 판단 하에 신중하게 내리시기 바랍니다.")
                                else:
                                    st.warning("해석할 데이터가 부족합니다 (Finnhub).")

                    except RateLimitException: # get_finnhub_stock_candles에서 발생한 예외가 여기까지 올 수 있음
                        # 이미 get_finnhub_stock_candles 내부에서 st.error가 호출되었을 것이므로 여기서는 추가 메시지 불필요
                        pass
                    except finnhub.FinnhubAPIException as api_e_main:
                         st.error(f"Finnhub API 처리 중 예외 발생: {api_e_main}")
                         logging.error(f"Finnhub API main processing error for {ticker_processed_tech}: {traceback.format_exc()}")
                    except Exception as e:
                        st.error(f"기술적 분석 (Finnhub) 처리 중 예기치 못한 오류 발생: {type(e).__name__} - {e}")
                        logging.error(f"Technical analysis tab (Finnhub) error: {traceback.format_exc()}")
                        if df_tech_fh is not None and not df_tech_fh.empty:
                            st.dataframe(df_tech_fh.head())
        else:
            st.info("종목 티커, 기간, 데이터 간격 등을 설정한 후 '기술적 분석 실행 (Finnhub)' 버튼을 클릭하세요.")


# --- 앱 정보 ---
st.sidebar.markdown("---")
st.sidebar.info("종합 주식 분석 툴 (Finnhub 연동) V1.9.9 | 정보 제공 목적 (투자 조언 아님)") # 버전 업데이트
st.sidebar.markdown("📌 [개발기 보러가기](https://technut.tistory.com/1)", unsafe_allow_html=True)
st.sidebar.caption("👨‍💻 기술 기반 주식 분석 툴 개발기")