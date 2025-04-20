# -*- coding: utf-8 -*-
# Combined app.py (종합 분석 + 단기 기술 분석) V1.5

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
import yfinance as yf # yfinance 추가

# --- 기본 경로 설정 및 로깅 (코드 1) ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd() # Fallback for environments where __file__ is not defined

# --- 기술 분석 함수 (코드 2) ---
def calculate_vwap(df):
    """주어진 DataFrame에 대해 VWAP(거래량 가중 평균 가격)를 계산합니다."""
    df = df.copy()
    # 데이터 유효성 검사 강화
    if not all(col in df.columns for col in ['High', 'Low', 'Close', 'Volume']):
        raise ValueError("VWAP 계산에 필요한 컬럼(High, Low, Close, Volume)이 없습니다.")
    if df['Volume'].sum() == 0:
        # 거래량이 0이면 VWAP 계산 불가, NaN 반환 또는 0 처리
        # st.warning("선택한 기간의 거래량이 0입니다. VWAP 계산 불가.")
        df['VWAP'] = np.nan
        return df['VWAP']

    df['typical_price'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['tp_volume'] = df['typical_price'] * df['Volume']
    # 누적 계산 시 0으로 나누는 경우 방지
    df['cumulative_volume'] = df['Volume'].cumsum()
    df['cumulative_tp_volume'] = df['tp_volume'].cumsum()
    df['VWAP'] = np.where(df['cumulative_volume'] > 0,
                          df['cumulative_tp_volume'] / df['cumulative_volume'],
                          np.nan) # 거래량 없으면 NaN
    return df['VWAP']

def plot_fibonacci_levels(df, ticker):
    """주어진 DataFrame을 사용하여 피보나치 되돌림 및 VWAP 차트를 생성합니다."""
    fig = go.Figure()

    # NaN 값 제거 후 최대/최소 계산 (차트 기간 내 유효한 가격 기준)
    valid_df = df.dropna(subset=['High', 'Low'])
    if valid_df.empty:
        st.warning("차트를 그릴 유효한 가격 데이터가 없습니다.")
        return fig # 빈 Figure 반환

    max_price = valid_df['High'].max()
    min_price = valid_df['Low'].min()
    diff = max_price - min_price

    # 피보나치 레벨 계산
    levels = {
        '1.0 (최저)': min_price,
        '0.618': min_price + 0.382 * diff, # 1 - 0.618 = 0.382
        '0.5': min_price + 0.5 * diff,
        '0.382': min_price + 0.618 * diff, # 1 - 0.382 = 0.618
        '0.236': min_price + 0.764 * diff, # 1 - 0.236 = 0.764
        '0.0 (최고)': max_price
    }
    # 거꾸로 계산하는 경우(max_price - ratio * diff)와 순서 유의

    # 색상 팔레트 또는 명확한 색상 지정
    colors = {'0.0 (최고)': 'red', '0.236': 'orange', '0.382': 'gold', '0.5': 'green', '0.618': 'blue', '1.0 (최저)': 'purple'}

    for key, value in levels.items():
        fig.add_hline(y=value, line_dash="dot", line_color=colors.get(key, 'grey'),
                      annotation_text=f"Fib {key}: ${value:.2f}",
                      annotation_position="top right",
                      annotation_font_size=10)

    # 캔들스틱 차트 추가
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name=f"{ticker} 캔들스틱"
    ))

    # VWAP 라인 추가 (존재하는 경우)
    if 'VWAP' in df.columns and not df['VWAP'].isnull().all():
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df["VWAP"],
            mode="lines",
            line=dict(color="darkorange", width=1.5),
            name="VWAP"
        ))
    elif 'VWAP' in df.columns:
         st.caption("VWAP 데이터가 없어 차트에 표시되지 않았습니다.")

    # 레이아웃 업데이트
    fig.update_layout(
        title=f"{ticker} 피보나치 되돌림 + VWAP 차트",
        xaxis_title="날짜 / 시간",
        yaxis_title="가격 ($)",
        xaxis_rangeslider_visible=False, # 레인지 슬라이더 비활성화 (선택 사항)
        legend_title_text="지표",
        hovermode="x unified",
        margin=dict(l=50, r=50, t=50, b=50) # 여백 조정
    )
    return fig


# --- Streamlit 페이지 설정 (코드 1 기반) ---
st.set_page_config(page_title="종합 주식 분석 V1.5", layout="wide", initial_sidebar_state="expanded") # 버전 업데이트

# --- API 키 로드 (코드 1) ---
NEWS_API_KEY = None
FRED_API_KEY = None
api_keys_loaded = False
secrets_available = hasattr(st, 'secrets')
sidebar_status = st.sidebar.empty()

# Secrets 시도
if secrets_available:
    try:
        NEWS_API_KEY = st.secrets.get("NEWS_API_KEY")
        FRED_API_KEY = st.secrets.get("FRED_API_KEY")
        if NEWS_API_KEY and FRED_API_KEY:
            api_keys_loaded = True
            # 성공 메시지는 아래에서 한 번만
        else:
            sidebar_status.warning("Secrets에 API 키 값이 일부 비어있습니다.")
    except KeyError:
        sidebar_status.warning("Secrets에 필요한 API 키(NEWS_API_KEY 또는 FRED_API_KEY)가 없습니다.")
    except Exception as e:
        sidebar_status.error(f"Secrets 로드 중 오류 발생: {e}")

# .env 파일 시도 (Secrets 실패 시)
if not api_keys_loaded:
    sidebar_status.info("Secrets 로드 실패 또는 키 부족. 로컬 .env 파일 확인 중...")
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
                 sidebar_status.error(".env 파일에서 유효한 API 키를 모두 찾을 수 없습니다.")
        else:
            sidebar_status.error(f".env 파일을 찾을 수 없습니다: {dotenv_path}")
    except Exception as e:
        sidebar_status.error(f".env 파일 로드 중 오류 발생: {e}")

# 최종 상태 확인 및 기본 분석 가능 여부 플래그
comprehensive_analysis_possible = api_keys_loaded
if not api_keys_loaded:
    st.sidebar.error("API 키 로드 실패! '종합 분석' 기능이 제한됩니다.")
else:
    sidebar_status.success("API 키 로드 완료.") # 최종 성공 메시지


# --- 사이드바 설정 (탭 구조 + 조건부 표시) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/10071/10071119.png", width=80) # 로고 예시
    st.title("📊 분석 도구 V1.5")
    st.markdown("---") # 구분선

    # 탭 선택
    page = st.radio(
        "분석 유형 선택",
        ["📊 종합 분석", "📈 단기 기술 분석"],
        captions=["재무, 예측, 뉴스 등", "VWAP, 피보나치 등"],
        key="page_selector"
    )
    st.markdown("---")

    # --- 종합 분석용 설정 (선택된 경우에만 표시) ---
    if page == "📊 종합 분석":
        st.header("⚙️ 종합 분석 설정")
        ticker_input = st.text_input(
            "종목 티커", value="AAPL",
            help="예: AAPL, MSFT, 005930.KS",
            key="main_ticker",
            disabled=not comprehensive_analysis_possible # API 키 없으면 비활성화
        )
        analysis_years = st.select_slider(
            "분석 기간 (년)",
            options=[1, 2, 3, 5, 7, 10], value=2,
            key="analysis_years",
            disabled=not comprehensive_analysis_possible
        )
        st.caption(f"과거 {analysis_years}년 데이터 분석")

        forecast_days = st.number_input(
            "예측 기간 (일)",
            min_value=7, max_value=90, value=30, step=7,
            key="forecast_days",
            disabled=not comprehensive_analysis_possible
        )
        st.caption(f"향후 {forecast_days}일 예측")

        num_trend_periods_input = st.number_input(
            "재무 추세 분기 수",
            min_value=2, max_value=12, value=4, step=1,
            key="num_trend_periods",
            disabled=not comprehensive_analysis_possible
        )
        st.caption(f"최근 {num_trend_periods_input}개 분기 재무 추세 계산")

        # --- 예측 세부 설정 (종합 분석 하위) ---
        st.divider()
        st.subheader("⚙️ 예측 세부 설정 (선택)")
        changepoint_prior_input = st.slider(
            "추세 변화 민감도 (Prophet)",
            min_value=0.001, max_value=0.5, value=0.05, step=0.01,
            format="%.3f",
            help="값이 클수록 모델이 과거 데이터의 추세 변화에 더 민감하게 반응합니다. (기본값: 0.05)",
            key="changepoint_prior",
            disabled=not comprehensive_analysis_possible
        )
        st.caption(f"현재 설정된 민감도: {changepoint_prior_input:.3f}")

        # --- 보유 정보 입력 (종합 분석 하위) ---
        st.divider()
        st.subheader("💰 보유 정보 입력 (선택)")
        avg_price = st.number_input(
            "평단가", min_value=0.0, value=0.0, format="%.2f",
            key="avg_price",
            disabled=not comprehensive_analysis_possible
        )
        quantity = st.number_input(
            "보유 수량", min_value=0, value=0, step=1,
            key="quantity",
            disabled=not comprehensive_analysis_possible
        )
        st.caption("평단가 입력 시 리스크 트래커 분석 활성화")
        st.divider()

        # 종합 분석 버튼은 메인 영역으로 이동

    # 단기 기술 분석 탭은 별도 설정 없음 (메인 영역에서 입력 받음)


# --- 캐시된 종합 분석 함수 (코드 1 기반, 모듈 import 내부 이동) ---
@st.cache_data(ttl=timedelta(hours=1)) # 1시간 캐시
def run_cached_analysis(ticker, news_key, fred_key, years, days, num_trend_periods, changepoint_prior_scale):
    """캐싱을 위한 종합 분석 함수 래퍼"""
    # --- 분석 모듈 로드 ---
    # 함수 호출 시점에 로드하여, 이 기능이 필요 없을 때는 로드하지 않도록 함
    try:
        import stock_analysis as sa
        logging.info("stock_analysis 모듈 로드 성공.")
    except ImportError as import_err:
        logging.error(f"stock_analysis.py 모듈 로드 실패: {import_err}")
        return {"error": f"분석 모듈(stock_analysis.py) 로딩 오류: {import_err}. 파일 존재 및 환경 설정을 확인하세요."}
    except Exception as e:
        logging.error(f"stock_analysis.py 모듈 로드 중 예상치 못한 오류: {e}")
        return {"error": f"분석 모듈 로딩 중 오류: {e}"}

    logging.info(f"종합 분석 캐시 미스/만료. 분석 실행: {ticker}, {years}년, {days}일, {num_trend_periods}분기, cp_prior={changepoint_prior_scale}")

    # API 키 재확인 (함수 내부)
    if not news_key or not fred_key:
        logging.error("종합 분석 시도 중 API 키 없음 감지.")
        # API 키가 없는 경우, 키 없이 실행 시도 (sa.analyze_stock 내부에서 처리)
        # 또는 여기서 에러 반환 결정 가능
        # return {"error": "유효한 API 키가 없어 종합 분석을 진행할 수 없습니다."}
        logging.warning("API 키 없이 종합 분석 시도. 일부 기능이 제한될 수 있습니다.")

    try:
        # analyze_stock 호출 시 changepoint_prior_scale 전달
        results = sa.analyze_stock(
            ticker, news_key, fred_key,
            analysis_period_years=years,
            forecast_days=days,
            num_trend_periods=num_trend_periods,
            changepoint_prior_scale=changepoint_prior_scale # 전달
        )
        logging.info(f"{ticker} 종합 분석 완료.")
        return results
    except Exception as e:
        error_traceback = traceback.format_exc()
        logging.error(f"종합 분석 함수(analyze_stock) 내에서 오류 발생: {e}\n{error_traceback}")
        return {"error": f"종합 분석 중 오류 발생: {e}"}


# --- 메인 화면 로직 (선택된 탭에 따라 분기) ---

# ============== 📊 종합 분석 탭 ==============
if page == "📊 종합 분석":
    st.title("📊 종합 분석 결과")
    st.markdown("기업 정보, 재무 추세, 예측(Prophet+기술지표), 리스크 트래커를 제공합니다.")
    st.markdown("---")

    # API 키 로드 실패 시 안내 및 버튼 비활성화
    if not comprehensive_analysis_possible:
        st.error("API 키 로드 실패. 이 기능을 사용하려면 API 키를 설정해주세요.")
        st.info("Streamlit Cloud 사용 시 Secrets에 `NEWS_API_KEY`와 `FRED_API_KEY`를 추가하세요.")
        st.info("로컬 실행 시 `.env` 파일에 키를 추가하고 애플리케이션을 재시작하세요.")
        analyze_button_main_disabled = True
    else:
        analyze_button_main_disabled = False

    # 종합 분석 시작 버튼
    analyze_button_main = st.button(
        "🚀 종합 분석 시작!",
        use_container_width=True,
        type="primary",
        key="analyze_main_button",
        disabled=analyze_button_main_disabled
    )

    results_placeholder = st.container() # 결과 표시 영역

    if analyze_button_main:
        # 사이드바에서 값 가져오기 (버튼 클릭 시점에)
        ticker_for_analysis = st.session_state.get('main_ticker', "AAPL") # key 사용
        analysis_years_val = st.session_state.get('analysis_years', 2)
        forecast_days_val = st.session_state.get('forecast_days', 30)
        num_trend_periods_val = st.session_state.get('num_trend_periods', 4)
        changepoint_prior_val = st.session_state.get('changepoint_prior', 0.05)
        avg_price_val = st.session_state.get('avg_price', 0.0) # 리스크 분석용
        quantity_val = st.session_state.get('quantity', 0) # 리스크 분석용

        if not ticker_for_analysis:
            results_placeholder.warning("종목 티커를 입력해주세요.")
        else:
            ticker_processed = ticker_for_analysis.strip().upper()
            with st.spinner(f"{ticker_processed} 종목 종합 분석 중... 잠시 기다려주세요."):
                try:
                    # 캐시 함수 호출
                    results = run_cached_analysis(
                        ticker_processed, NEWS_API_KEY, FRED_API_KEY,
                        analysis_years_val, forecast_days_val, num_trend_periods_val,
                        changepoint_prior_val
                    )
                    results_placeholder.empty() # 이전 결과 지우기

                    # --- 결과 표시 (코드 1의 결과 표시 로직 전체) ---
                    if results and isinstance(results, dict) and "error" not in results:
                        st.header(f"📈 {ticker_processed} 분석 결과 (추세 민감도: {changepoint_prior_val:.3f})")

                        # 1. 요약 정보
                        st.subheader("요약 정보")
                        col1, col2, col3 = st.columns(3)
                        col1.metric("현재가 (최근 종가)", f"${results.get('current_price', 'N/A')}")
                        col2.metric("분석 시작일", results.get('analysis_period_start', 'N/A'))
                        col3.metric("분석 종료일", results.get('analysis_period_end', 'N/A'))

                        # 2. 기본적 분석
                        st.subheader("📊 기업 기본 정보 (Fundamentals)")
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
                                    st.write(summary)
                            st.caption("Data Source: Yahoo Finance")
                        else:
                            st.warning("기업 기본 정보를 가져오지 못했습니다.")

                        # 3. 주요 재무 추세
                        st.subheader(f"📈 주요 재무 추세 (최근 {num_trend_periods_val} 분기)")
                        tab_titles = ["영업이익률(%)", "ROE(%)", "부채비율", "유동비율"]
                        tab1, tab2, tab3, tab4 = st.tabs(tab_titles)
                        # (코드 1의 탭 내용과 동일하게 구현 - 데이터프레임 생성 및 차트 그리기)
                        # 예시: 영업이익률 탭
                        with tab1:
                             margin_trend_data = results.get('operating_margin_trend')
                             if margin_trend_data and isinstance(margin_trend_data, list) and len(margin_trend_data) > 0:
                                 try:
                                     df_margin = pd.DataFrame(margin_trend_data)
                                     df_margin['Date'] = pd.to_datetime(df_margin['Date'])
                                     df_margin.set_index('Date', inplace=True)
                                     if 'Op Margin (%)' in df_margin.columns:
                                         st.line_chart(df_margin[['Op Margin (%)']])
                                         with st.expander("데이터 보기"):
                                             st.dataframe(df_margin[['Op Margin (%)']].style.format("{:.2f}%"), use_container_width=True)
                                     else: st.error("결과에서 'Op Margin (%)' 컬럼을 찾을 수 없습니다.")
                                 except Exception as e: st.error(f"영업이익률 차트/데이터 표시 오류: {e}")
                             else: st.info("영업이익률 추세 데이터 없음.")
                        # ROE, 부채비율, 유동비율 탭도 유사하게 구현 (코드 1 참조)
                        with tab2:
                            roe_trend_data = results.get('roe_trend')
                            if roe_trend_data and isinstance(roe_trend_data, list) and len(roe_trend_data) > 0:
                                try:
                                    df_roe = pd.DataFrame(roe_trend_data)
                                    df_roe['Date'] = pd.to_datetime(df_roe['Date'])
                                    df_roe.set_index('Date', inplace=True)
                                    if 'ROE (%)' in df_roe.columns:
                                        st.line_chart(df_roe[['ROE (%)']])
                                        with st.expander("데이터 보기"):
                                            st.dataframe(df_roe[['ROE (%)']].style.format({"ROE (%)": "{:.2f}%"}), use_container_width=True)
                                    else: st.error("결과에서 'ROE (%)' 컬럼을 찾을 수 없습니다.")
                                except Exception as e: st.error(f"ROE 차트/데이터 표시 오류: {e}")
                            else: st.info("ROE 추세 데이터 없음.")
                        with tab3:
                            debt_trend_data = results.get('debt_to_equity_trend')
                            if debt_trend_data and isinstance(debt_trend_data, list) and len(debt_trend_data) > 0:
                                try:
                                    df_debt = pd.DataFrame(debt_trend_data)
                                    df_debt['Date'] = pd.to_datetime(df_debt['Date'])
                                    df_debt.set_index('Date', inplace=True)
                                    if 'D/E Ratio' in df_debt.columns:
                                        st.line_chart(df_debt[['D/E Ratio']])
                                        with st.expander("데이터 보기"):
                                            st.dataframe(df_debt[['D/E Ratio']].style.format({"D/E Ratio": "{:.2f}"}), use_container_width=True)
                                    else: st.error("결과에서 'D/E Ratio' 컬럼을 찾을 수 없습니다.")
                                except Exception as e: st.error(f"부채비율 차트/데이터 표시 오류: {e}")
                            else: st.info("부채비율 추세 데이터 없음.")
                        with tab4:
                            current_trend_data = results.get('current_ratio_trend')
                            if current_trend_data and isinstance(current_trend_data, list) and len(current_trend_data) > 0:
                                try:
                                    df_current = pd.DataFrame(current_trend_data)
                                    df_current['Date'] = pd.to_datetime(df_current['Date'])
                                    df_current.set_index('Date', inplace=True)
                                    if 'Current Ratio' in df_current.columns:
                                        st.line_chart(df_current[['Current Ratio']])
                                        with st.expander("데이터 보기"):
                                            st.dataframe(df_current[['Current Ratio']].style.format({"Current Ratio": "{:.2f}"}), use_container_width=True)
                                    else: st.error("결과에서 'Current Ratio' 컬럼을 찾을 수 없습니다.")
                                except Exception as e: st.error(f"유동비율 차트/데이터 표시 오류: {e}")
                            else: st.info("유동비율 추세 데이터 없음.")
                        st.divider()

                        # 4. 기술적 분석 (차트)
                        st.subheader("기술적 분석 차트 (종합)")
                        stock_chart_fig = results.get('stock_chart_fig')
                        if stock_chart_fig:
                            st.plotly_chart(stock_chart_fig, use_container_width=True)
                        else:
                            st.warning("주가 차트 생성 실패 (종합 분석).")
                        st.divider()

                        # 5. 시장 심리
                        st.subheader("시장 심리 분석")
                        col_news, col_fng = st.columns([2, 1])
                        with col_news:
                            st.markdown("**📰 뉴스 감정 분석**")
                            news_sentiment = results.get('news_sentiment', ["뉴스 분석 정보 없음."])
                            if isinstance(news_sentiment, list) and len(news_sentiment) > 0:
                                st.info(news_sentiment[0]) # 헤더
                                if len(news_sentiment) > 1:
                                    with st.expander("최근 뉴스 목록 보기", expanded=False):
                                        for line in news_sentiment[1:]: st.write(f"- {line}")
                            else: st.write(str(news_sentiment)) # 에러 메시지 등
                        with col_fng:
                            st.markdown("**😨 공포-탐욕 지수**")
                            fng_index = results.get('fear_greed_index', "N/A")
                            if isinstance(fng_index, dict):
                                st.metric(label="현재 지수", value=fng_index.get('value', 'N/A'), delta=fng_index.get('classification', ''))
                            else: st.write(fng_index)
                        st.divider()

                        # 6. Prophet 예측 분석
                        st.subheader("Prophet 주가 예측")
                        forecast_fig = results.get('forecast_fig')
                        forecast_data_list = results.get('prophet_forecast') # 리스크 트래커에서도 사용

                        if forecast_fig: st.plotly_chart(forecast_fig, use_container_width=True)
                        elif isinstance(forecast_data_list, str): st.info(forecast_data_list) # 예측 불가 메시지
                        else: st.warning("예측 차트 생성 실패.")

                        if isinstance(forecast_data_list, list) and len(forecast_data_list) > 0:
                             st.markdown("**📊 예측 데이터 (최근 10일)**")
                             try:
                                 df_fcst = pd.DataFrame(forecast_data_list)
                                 df_fcst['ds'] = pd.to_datetime(df_fcst['ds']).dt.strftime('%Y-%m-%d')
                                 st.dataframe(
                                     df_fcst[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(10).style.format({'yhat': "{:.2f}", 'yhat_lower': "{:.2f}", 'yhat_upper': "{:.2f}"}),
                                     use_container_width=True
                                 )
                             except Exception as e:
                                 st.error(f"예측 데이터 표시 중 오류: {e}")
                        # 교차 검증 결과
                        cv_plot_path = results.get('cv_plot_path')
                        if cv_plot_path and isinstance(cv_plot_path, str) and os.path.exists(cv_plot_path):
                             st.markdown("**📉 교차 검증 결과 (MAPE)**")
                             st.image(cv_plot_path, caption="MAPE (Mean Absolute Percentage Error, 낮을수록 예측 정확도 높음)")
                        elif cv_plot_path is None and isinstance(forecast_data_list, list):
                             st.caption("교차 검증(CV) 결과 없음 (데이터 기간 부족 또는 오류).")
                        st.divider()

                        # 7. 리스크 트래커 (코드 1 로직 활용)
                        st.subheader("🚨 리스크 트래커 (예측 기반)")
                        risk_days = 0
                        max_loss_pct = 0
                        max_loss_amt = 0
                        if avg_price_val > 0 and isinstance(forecast_data_list, list) and len(forecast_data_list) > 0:
                             try:
                                 df_pred = pd.DataFrame(forecast_data_list)
                                 # 필수 컬럼 및 타입 확인
                                 required_fcst_cols = ['ds', 'yhat', 'yhat_lower', 'yhat_upper']
                                 if not all(col in df_pred.columns for col in required_fcst_cols):
                                     st.warning("예측 데이터에 필수 컬럼이 부족하여 리스크 분석 불가.")
                                 else:
                                     df_pred['ds'] = pd.to_datetime(df_pred['ds'])
                                     df_pred['yhat_lower'] = pd.to_numeric(df_pred['yhat_lower'], errors='coerce')
                                     df_pred.dropna(subset=['yhat_lower'], inplace=True) # 하한선 없으면 리스크 계산 불가

                                     if not df_pred.empty:
                                         df_pred['평단가'] = avg_price_val
                                         df_pred['리스크 여부'] = df_pred['yhat_lower'] < df_pred['평단가']
                                         df_pred['예상 손실률'] = np.where(df_pred['리스크 여부'], ((df_pred['yhat_lower'] - df_pred['평단가']) / df_pred['평단가']) * 100, 0)

                                         if quantity_val > 0:
                                             df_pred['예상 손실액'] = np.where(df_pred['리스크 여부'], (df_pred['yhat_lower'] - df_pred['평단가']) * quantity_val, 0)
                                         else:
                                             df_pred['예상 손실액'] = 0

                                         risk_days = df_pred['리스크 여부'].sum()
                                         max_loss_pct = df_pred['예상 손실률'].min() if risk_days > 0 else 0
                                         max_loss_amt = df_pred['예상 손실액'].min() if risk_days > 0 and quantity_val > 0 else 0

                                         st.markdown("##### 리스크 요약")
                                         col_r1, col_r2, col_r3 = st.columns(3)
                                         col_r1.metric("⚠️ < 평단가 예측 일수", f"{risk_days}일 / {forecast_days_val}일")
                                         col_r2.metric("📉 Max 예측 손실률", f"{max_loss_pct:.2f}%")
                                         if quantity_val > 0: col_r3.metric("💸 Max 예측 손실액", f"${max_loss_amt:,.2f}")

                                         if risk_days > 0: st.warning(f"향후 {forecast_days_val}일 예측 기간 중 **{risk_days}일** 동안 예측 하한선이 평단가(${avg_price_val:.2f})보다 낮을 수 있습니다. (예상 최대 손실률: **{max_loss_pct:.2f}%**).")
                                         else: st.success(f"향후 {forecast_days_val}일 동안 예측 하한선이 평단가(${avg_price_val:.2f})보다 낮아질 가능성은 현재 예측되지 않았습니다.")

                                         # 리스크 비교 차트 (코드 1 로직 활용)
                                         st.markdown("##### 평단가 vs 예측 구간 비교 차트")
                                         fig_risk = go.Figure()
                                         fig_risk.add_trace(go.Scatter(x=df_pred['ds'], y=df_pred['yhat_upper'], mode='lines', line_color='rgba(0,100,80,0.2)', name='Upper Bound'))
                                         fig_risk.add_trace(go.Scatter(x=df_pred['ds'], y=df_pred['yhat_lower'], mode='lines', line_color='rgba(0,100,80,0.2)', name='Lower Bound', fill='tonexty', fillcolor='rgba(0,100,80,0.1)'))
                                         if 'yhat' in df_pred.columns: fig_risk.add_trace(go.Scatter(x=df_pred['ds'], y=df_pred['yhat'], mode='lines', line=dict(dash='dash', color='rgba(0,100,80,0.6)'), name='Forecast (yhat)'))
                                         fig_risk.add_hline(y=avg_price_val, line_dash="dot", line_color="red", annotation_text=f"평단가: ${avg_price_val:.2f}", annotation_position="bottom right")
                                         df_risk_periods = df_pred[df_pred['리스크 여부']]
                                         if not df_risk_periods.empty: fig_risk.add_trace(go.Scatter(x=df_risk_periods['ds'], y=df_risk_periods['yhat_lower'], mode='markers', marker_symbol='x', marker_color='red', name='Risk Day (Lower < Avg Price)'))
                                         fig_risk.update_layout(title=f"{ticker_processed} 예측 구간 vs 평단가 비교", xaxis_title="날짜", yaxis_title="가격", hovermode="x unified", margin=dict(l=20, r=20, t=40, b=20))
                                         st.plotly_chart(fig_risk, use_container_width=True)

                                         # 리스크 상세 데이터
                                         if risk_days > 0:
                                             with st.expander(f"리스크 예측일 상세 데이터 보기 ({risk_days}일)"):
                                                 df_risk_days_display = df_pred[df_pred['리스크 여부']].copy()
                                                 df_risk_days_display['ds'] = df_risk_days_display['ds'].dt.strftime('%Y-%m-%d')
                                                 cols_to_show = ['ds', 'yhat_lower', '평단가', '예상 손실률']
                                                 if quantity_val > 0: cols_to_show.append('예상 손실액')
                                                 st.dataframe(
                                                     df_risk_days_display[cols_to_show].style.format({"yhat_lower":"{:.2f}", "평단가":"{:.2f}", "예상 손실률":"{:.2f}%", "예상 손실액":"${:,.2f}"}),
                                                     use_container_width=True
                                                 )
                                     else: st.info("예측 하한선 데이터가 유효하지 않아 리스크 분석을 수행할 수 없습니다.")
                             except Exception as risk_calc_err:
                                 st.error(f"리스크 트래커 계산/표시 중 오류 발생: {risk_calc_err}")
                                 logging.error(f"Risk tracker error: {traceback.format_exc()}")
                        elif avg_price_val <= 0:
                             st.info("⬅️ 사이드바에서 '평단가'를 0보다 큰 값으로 입력하시면 리스크 분석 결과를 볼 수 있습니다.")
                        else: # forecast_data_list가 유효하지 않은 경우
                             st.warning("Prophet 예측 데이터가 유효하지 않아 리스크 분석을 수행할 수 없습니다.")
                        st.divider()

                        # 8. 자동 분석 결과 요약 (코드 1 로직 활용)
                        st.subheader("🧐 자동 분석 결과 요약 (참고용)")
                        summary_points = []
                        # (코드 1의 요약 생성 로직과 동일하게 구현)
                        # 예측 요약
                        if isinstance(forecast_data_list, list) and len(forecast_data_list) > 0:
                            try:
                                start_pred_row = forecast_data_list[0]
                                end_pred_row = forecast_data_list[-1]
                                start_pred = pd.to_numeric(start_pred_row.get('yhat'), errors='coerce')
                                end_pred = pd.to_numeric(end_pred_row.get('yhat'), errors='coerce')
                                lower = pd.to_numeric(end_pred_row.get('yhat_lower'), errors='coerce')
                                upper = pd.to_numeric(end_pred_row.get('yhat_upper'), errors='coerce')

                                if pd.notna(start_pred) and pd.notna(end_pred):
                                    trend_obs = "상승" if end_pred > start_pred else "하락" if end_pred < start_pred else "횡보"
                                    lower_str = f"{lower:.2f}" if pd.notna(lower) else 'N/A'
                                    upper_str = f"{upper:.2f}" if pd.notna(upper) else 'N/A'
                                    summary_points.append(f"- **예측:** 향후 {forecast_days_val}일간 **{trend_obs}** 추세 예상 (최종일 예측 범위: ${lower_str} ~ ${upper_str}).")
                                else: summary_points.append("- 예측: 최종 예측값 유효하지 않음.")
                            except Exception as e:
                                logging.warning(f"예측 요약 생성 오류: {e}")
                                summary_points.append("- 예측: 요약 생성 중 오류 발생.")

                        # 뉴스 요약
                        if isinstance(news_sentiment, list) and len(news_sentiment) > 0 and ":" in news_sentiment[0]:
                            try:
                                score_part = news_sentiment[0].split(":")[-1].strip()
                                avg_score = float(score_part)
                                sentiment_desc = "긍정적" if avg_score > 0.05 else "부정적" if avg_score < -0.05 else "중립적"
                                summary_points.append(f"- **뉴스:** 최근 뉴스 평균 감성 점수 {avg_score:.2f}, 전반적으로 **{sentiment_desc}**인 분위기.")
                            except Exception as e:
                                logging.warning(f"뉴스 요약 생성 오류: {e}")
                                summary_points.append("- 뉴스: 요약 생성 중 오류 발생.")

                        # F&G 지수 요약
                        if isinstance(fng_index, dict):
                               summary_points.append(f"- **시장 심리:** 공포-탐욕 지수 {fng_index.get('value', 'N/A')} (**{fng_index.get('classification', 'N/A')}**).")

                        # 기본 정보 요약
                        if fundamentals and isinstance(fundamentals, dict):
                            per_val = fundamentals.get("PER", "N/A")
                            sector_val = fundamentals.get("업종", "N/A")
                            fund_summary_parts = []
                            if per_val != "N/A": fund_summary_parts.append(f"PER {per_val}")
                            if sector_val != "N/A": fund_summary_parts.append(f"업종 '{sector_val}'")
                            if fund_summary_parts: summary_points.append(f"- **기본 정보:** {', '.join(fund_summary_parts)}.")

                        # 재무 추세 요약 (결과 dict 구조 확인 필요)
                        trend_summary_parts = []
                        try:
                            if results.get('operating_margin_trend') and results['operating_margin_trend']: trend_summary_parts.append(f"최근 영업이익률 {results['operating_margin_trend'][-1].get('Op Margin (%)', 'N/A'):.2f}%")
                            if results.get('roe_trend') and results['roe_trend']: trend_summary_parts.append(f"ROE {results['roe_trend'][-1].get('ROE (%)', 'N/A'):.2f}%")
                            if results.get('debt_to_equity_trend') and results['debt_to_equity_trend']: trend_summary_parts.append(f"부채비율 {results['debt_to_equity_trend'][-1].get('D/E Ratio', 'N/A'):.2f}")
                            if results.get('current_ratio_trend') and results['current_ratio_trend']: trend_summary_parts.append(f"유동비율 {results['current_ratio_trend'][-1].get('Current Ratio', 'N/A'):.2f}")
                            if trend_summary_parts: summary_points.append(f"- **최근 재무:** {', '.join(trend_summary_parts)}.")
                        except (KeyError, IndexError, TypeError) as e:
                             logging.warning(f"재무 추세 요약 생성 오류: {e}")
                             summary_points.append("- 최근 재무: 요약 생성 중 오류 발생.")

                        # 리스크 요약
                        if avg_price_val > 0 and isinstance(forecast_data_list, list) and len(forecast_data_list) > 0:
                            if risk_days > 0:
                                summary_points.append(f"- **리스크:** 예측상 향후 {forecast_days_val}일 중 **{risk_days}일**은 평단가 하회 가능성 있음 (최대 **{max_loss_pct:.2f}%** 손실률 예상).")
                            else:
                                summary_points.append(f"- **리스크:** 예측상 향후 {forecast_days_val}일간 평단가(${avg_price_val:.2f}) 하회 가능성은 낮아 보임.")
                        elif avg_price_val > 0:
                               summary_points.append("- 리스크: 평단가는 입력되었으나, 예측 데이터 부족/오류로 분석 불가.")

                        # 최종 요약 출력
                        if summary_points:
                            st.markdown("\n".join(summary_points))
                            st.caption("⚠️ **주의:** 이 요약은 자동 생성된 정보이며 투자 조언이 아닙니다. 모든 투자 결정과 책임은 사용자 본인에게 있습니다.")
                        else:
                            st.write("분석 요약을 생성할 수 없습니다.")

                    # 오류 처리
                    elif results and "error" in results:
                        results_placeholder.error(f"분석 실패: {results['error']}")
                        if "stock_analysis.py" in results['error']:
                            st.info("`stock_analysis.py` 파일이 `app.py`와 동일한 디렉토리에 있는지, 필요한 라이브러리(Prophet 등)가 설치되었는지 확인하세요.")
                    else:
                        results_placeholder.error("분석 결과를 처리하는 중 예기치 않은 오류가 발생했습니다.")

                except Exception as e:
                    # 앱 실행 자체의 예외 처리
                    error_traceback = traceback.format_exc()
                    logging.error(f"종합 분석 실행 중 예기치 않은 오류 발생: {e}\n{error_traceback}")
                    results_placeholder.error(f"앱 실행 중 오류 발생: {e}")
                    st.exception(e) # Streamlit의 내장 예외 표시
    else:
        # 버튼 클릭 전 안내 메시지 (API 키 있을 때만)
        if comprehensive_analysis_possible:
             results_placeholder.info("⬅️ 왼쪽 사이드바에서 분석 설정을 확인/수정한 후 '종합 분석 시작' 버튼을 클릭하세요.")
        # API 키 없을 때의 안내는 버튼 비활성화 메시지로 대체됨

# ============== 📈 단기 기술 분석 탭 ==============
elif page == "📈 단기 기술 분석":
    st.title("📈 단기 기술 분석 - VWAP & 피보나치")
    st.markdown("일봉 또는 분봉 데이터를 기준으로 VWAP과 피보나치 되돌림 수준을 시각화합니다.")
    st.markdown("---")

    # 이 탭의 입력 위젯들
    ticker_tech = st.text_input("종목 티커를 입력하세요 (예: AAPL, TSLA)", "AAPL", key="tech_ticker")

    # 날짜 및 간격 선택 레이아웃 개선
    today = datetime.now().date()
    default_start_date = today - relativedelta(months=3) # 기본 3개월

    col1_tech, col2_tech, col3_tech = st.columns(3)
    with col1_tech:
        start_date = st.date_input("시작일", default_start_date, key="tech_start", max_value=today - timedelta(days=1))
    with col2_tech:
        end_date = st.date_input("종료일", today, key="tech_end", min_value=start_date, max_value=today)
    with col3_tech:
        # 간격 선택 도움말 추가
        interval_options = {
            "일봉": "1d", "1시간": "1h", "30분": "30m", "15분": "15m", "5분": "5m", "1분": "1m"
        }
        interval_display = st.selectbox(
            "데이터 간격",
            options=list(interval_options.keys()),
            key="tech_interval_display",
            help="""
            - 일봉(1d): 최대 730일 조회 가능
            - 1시간(1h): 최대 730일 조회 가능
            - 30분/15분/5분/1분: 최대 60일 조회 가능 (주의: 데이터 양이 많으면 느려질 수 있음)
            """
        )
        interval = interval_options[interval_display] # 실제 API 파라미터

    # 데이터 불러오기 및 분석 버튼
    analyze_button_tech = st.button("📥 데이터 불러오기 및 분석", key="tech_analyze", use_container_width=True, type="primary")

    if analyze_button_tech:
        if not ticker_tech:
            st.warning("종목 티커를 입력해주세요.")
        elif start_date >= end_date:
             st.warning("시작일은 종료일보다 이전이어야 합니다.")
        else:
            ticker_processed_tech = ticker_tech.strip().upper()
            st.write(f"**{ticker_processed_tech}** ({interval_display}) 데이터 로딩 및 분석 중...")
            with st.spinner(f"{ticker_processed_tech} ({interval_display}) 데이터 불러오는 중..."):
                try:
                    # yfinance는 종료일을 포함하지 않을 수 있으므로 하루 추가
                    # 하지만 date_input의 max_value가 today이므로, end_date가 today면 +1하면 미래가 됨.
                    # yfinance는 end date를 exclusive하게 처리할 수 있음. 문서를 확인하거나 그냥 end_date 사용.
                    # end_date가 today면 today 데이터 포함하려면 end_date=None 사용 가능성도 고려.
                    # 명시적으로 종료일 다음날까지 요청하는 것이 안전할 수 있음.
                    fetch_end_date = end_date + timedelta(days=1)

                    # 데이터 기간 제한 (yfinance 제약 조건)
                    period_days = (end_date - start_date).days
                    valid_period = True
                    if interval in ['1m'] and period_days > 7:
                        st.warning("1분봉 데이터는 최대 7일까지만 조회 가능합니다. 시작일을 조정합니다.")
                        start_date = end_date - timedelta(days=7)
                    elif interval in ['5m', '15m', '30m'] and period_days > 60:
                         st.warning(f"{interval_display} 데이터는 최대 60일까지만 조회 가능합니다. 시작일을 조정합니다.")
                         start_date = end_date - timedelta(days=60)
                    elif interval in ['1h', '1d'] and period_days > 730:
                         st.warning(f"{interval_display} 데이터는 최대 730일까지만 조회 가능합니다. 시작일을 조정합니다.")
                         start_date = end_date - timedelta(days=730)

                    # 재조정된 시작일로 fetch_end_date 다시 설정
                    fetch_end_date = end_date + timedelta(days=1)

                    logging.info(f"yfinance 다운로드 요청: Ticker={ticker_processed_tech}, Start={start_date}, End={fetch_end_date}, Interval={interval}")
                    df_tech = yf.download(ticker_processed_tech, start=start_date, end=fetch_end_date, interval=interval, progress=False) # progress=False 추가

                    if df_tech.empty:
                        st.error(f"❌ 데이터를 불러오지 못했습니다. 티커('{ticker_processed_tech}'), 기간({start_date}~{end_date}), 간격('{interval_display}')을 확인하세요.")
                        logging.warning(f"yf.download 결과 비어 있음: {ticker_processed_tech}, {start_date}, {fetch_end_date}, {interval}")
                    else:
                        logging.info(f"다운로드 완료. 데이터 행: {len(df_tech)}")
                        # 실제 데이터의 마지막 날짜 확인
                        actual_end_date = df_tech.index.max().date() if isinstance(df_tech.index, pd.DatetimeIndex) else None
                        st.caption(f"실제 조회된 데이터 기간: {df_tech.index.min().strftime('%Y-%m-%d %H:%M')} ~ {df_tech.index.max().strftime('%Y-%m-%d %H:%M')}")

                        # 필수 컬럼 존재 확인
                        required_cols = ['High', 'Low', 'Close', 'Volume', 'Open']
                        if not all(col in df_tech.columns for col in required_cols):
                            st.error(f"필수 컬럼({required_cols})이 다운로드된 데이터에 없습니다. API 응답 형식이 다르거나 데이터가 없을 수 있습니다.")
                            logging.error(f"필수 컬럼 누락. 다운로드된 컬럼: {df_tech.columns.tolist()}")
                        else:
                             with st.spinner("VWAP 계산 및 차트 생성 중..."):
                                try:
                                     # NaN 처리 (특히 분봉 데이터)
                                     # df_tech = df_tech.ffill().bfill() # 주의: 가격 데이터 왜곡 가능성 있음. 필요한 경우에만 사용.
                                     df_tech.dropna(subset=required_cols, inplace=True) # 필수 컬럼 NaN인 행 제거

                                     if df_tech.empty:
                                         st.warning("데이터 정제 후 남은 데이터가 없습니다.")
                                     elif df_tech['Volume'].sum() == 0:
                                         st.warning("선택한 기간/간격의 거래량이 0입니다. VWAP를 계산할 수 없습니다.")
                                         st.subheader(f"📄 {ticker_processed_tech} 최근 데이터 (VWAP 없음)")
                                         st.dataframe(df_tech.tail(10), use_container_width=True)
                                     else:
                                         # VWAP 계산
                                         df_tech['VWAP'] = calculate_vwap(df_tech)

                                         # 차트 생성 및 표시
                                         st.subheader(f"📌 {ticker_processed_tech} 피보나치 + VWAP 기술적 분석 차트 ({interval_display})")
                                         chart_tech = plot_fibonacci_levels(df_tech, ticker_processed_tech)
                                         st.plotly_chart(chart_tech, use_container_width=True)

                                         # 최근 데이터 표시
                                         st.subheader(f"📄 {ticker_processed_tech} 최근 데이터 (VWAP 포함)")
                                         st.dataframe(df_tech.tail(10).style.format({"Open": "{:.2f}", "High": "{:.2f}", "Low": "{:.2f}", "Close": "{:.2f}", "Adj Close": "{:.2f}", "VWAP": "{:.2f}"}), use_container_width=True)

                                except ValueError as ve: # calculate_vwap 등에서 발생 가능
                                    st.error(f"데이터 처리 오류: {ve}")
                                    logging.error(f"Data processing error: {ve}")
                                except Exception as plot_err:
                                    st.error(f"차트 생성 또는 VWAP 계산 중 예상치 못한 오류 발생: {plot_err}")
                                    logging.error(f"Plotting/VWAP error: {traceback.format_exc()}")
                                    st.dataframe(df_tech.tail(10), use_container_width=True) # 오류 시에도 데이터는 표시

                except Exception as yf_err:
                     st.error(f"yfinance 데이터 다운로드 중 오류 발생: {yf_err}")
                     logging.error(f"yfinance download error: {traceback.format_exc()}")
    else:
        st.info("종목 티커, 기간, 간격을 설정한 후 '데이터 불러오기 및 분석' 버튼을 클릭하세요.")

# --- 앱 정보 ---
st.sidebar.markdown("---")
st.sidebar.info(
    """
    **종합 주식 분석 툴 V1.5**
    - 기본 분석 (재무, 예측 등)
    - 단기 기술 분석 (VWAP, 피보나치)

    **주의:** 본 도구는 정보 제공 목적으로, 투자 조언이 아닙니다.
    투자 결정과 책임은 사용자 본인에게 있습니다.
    """
)