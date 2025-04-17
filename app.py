# app.py (changepoint_prior_scale 슬라이더 추가 - 오류 수정 최종본)

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

# --- 기본 경로 설정 및 로깅 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()

# --- 분석 로직 가져오기 ---
try:
    import stock_analysis as sa
except Exception as import_err: # 모든 Import 관련 오류 포괄
    st.error(f"분석 로직(stock_analysis.py) 로딩 오류: {import_err}")
    st.info("파일 존재 여부, 내부 문법 오류, 라이브러리 설치 상태를 확인하세요.")
    st.stop() # 오류 시 앱 실행 중지

# --- Streamlit 페이지 설정 ---
st.set_page_config(page_title="주식 분석 V1.4", layout="wide", initial_sidebar_state="expanded") # 버전 업데이트

st.title("📊 주식 분석 및 예측 도구 v1.4") # 버전 업데이트
st.markdown("기업 정보, 재무 추세, 예측(Prophet+기술지표), 리스크 트래커를 제공합니다.") # 설명 업데이트

# --- API 키 로드 ---
NEWS_API_KEY = None
FRED_API_KEY = None
api_keys_loaded = False
secrets_available = hasattr(st, 'secrets') # secrets 존재 여부 확인
sidebar_status = st.sidebar.empty() # 사이드바 상태 메시지 표시용 placeholder

if secrets_available:
    try:
        NEWS_API_KEY = st.secrets["NEWS_API_KEY"]
        FRED_API_KEY = st.secrets["FRED_API_KEY"]
        if NEWS_API_KEY and FRED_API_KEY:
            api_keys_loaded = True
            # 성공 메시지는 아래에서 한 번만 표시
        else:
            sidebar_status.warning("Secrets에 API 키 값이 비어있습니다.") # 오류 대신 경고
    except KeyError:
        sidebar_status.warning("Secrets에 필요한 API 키(NEWS_API_KEY 또는 FRED_API_KEY)가 없습니다.")
    except Exception as e:
        sidebar_status.error(f"Secrets 로드 중 오류 발생: {e}")

# Secrets 로드 실패 시 .env 파일 시도
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
                sidebar_status.success("API 키 로드 완료 (.env)") # .env 로드 성공 시 메시지
            else:
                sidebar_status.error(".env 파일에서 유효한 API 키를 찾을 수 없습니다.")
        else:
            sidebar_status.error(f".env 파일을 찾을 수 없습니다: {dotenv_path}")
    except Exception as e:
        sidebar_status.error(f".env 파일 로드 중 오류 발생: {e}")

# 최종 로드 상태 확인
if not api_keys_loaded:
    st.sidebar.error("API 키 로드에 최종 실패했습니다! 일부 기능이 제한됩니다.")


# --- 사이드바 설정 ---
with st.sidebar:
    st.header("⚙️ 분석 설정")
    ticker_input = st.text_input("종목 티커", value="AAPL", help="예: AAPL, MSFT, 005930.KS")
    analysis_years = st.select_slider("분석 기간 (년)", options=[1, 2, 3, 5, 7, 10], value=2)
    st.caption(f"과거 {analysis_years}년 데이터 분석")
    forecast_days = st.number_input("예측 기간 (일)", min_value=7, max_value=90, value=30, step=7)
    st.caption(f"향후 {forecast_days}일 예측")
    num_trend_periods_input = st.number_input("재무 추세 분기 수", min_value=2, max_value=12, value=4, step=1)
    st.caption(f"최근 {num_trend_periods_input}개 분기 재무 추세 계산")

    # --- ⭐ 예측 세부 설정 추가 ---
    st.divider()
    st.header("⚙️ 예측 세부 설정 (선택)")
    changepoint_prior_input = st.slider( # 변수명은 input으로 구분
        "추세 변화 민감도 (Changepoint Prior Scale)", # 라벨에 파라미터명 명시
        min_value=0.001, max_value=0.5, value=0.05, step=0.01, # 기본값 0.05 유지
        format="%.3f", # 소수점 3자리
        help="값이 클수록 모델이 과거 데이터의 추세 변화에 더 민감하게 반응합니다. (기본값: 0.05)"
    )
    st.caption(f"현재 설정된 민감도: {changepoint_prior_input:.3f}") # 선택된 값 표시
    # --------------------------

    st.divider()
    st.header("💰 보유 정보 입력 (선택)")
    avg_price = st.number_input("평단가", min_value=0.0, value=0.0, format="%.2f")
    quantity = st.number_input("보유 수량", min_value=0, value=0, step=1)
    st.caption("평단가 입력 시 리스크 트래커 분석 활성화")
    st.divider()
    analyze_button = st.button("🚀 분석 시작!", use_container_width=True, type="primary") # 버튼 강조

# --- 메인 화면 ---
results_placeholder = st.container() # 결과 표시 영역

# 캐시된 분석 함수 정의 (changepoint_prior_scale 인자 추가)
@st.cache_data(ttl=timedelta(hours=1)) # 1시간 캐시
def run_cached_analysis(ticker, news_key, fred_key, years, days, num_trend_periods, changepoint_prior_scale): # 추가된 인자 받음
    """캐싱을 위한 분석 함수 래퍼"""
    logging.info(f"캐시 미스/만료. 분석 실행: {ticker}, {years}년, {days}일, {num_trend_periods}분기, cp_prior={changepoint_prior_scale}")
    # API 키 유효성 재확인 (캐시 함수 내부에서도)
    if not news_key or not fred_key:
        logging.error("분석 시도 중 API 키 없음 감지.")
        return {"error": "유효한 API 키가 로드되지 않았습니다."}
    try:
        # analyze_stock 호출 시 changepoint_prior_scale 전달
        return sa.analyze_stock(
            ticker, news_key, fred_key,
            analysis_period_years=years,
            forecast_days=days,
            num_trend_periods=num_trend_periods,
            changepoint_prior_scale=changepoint_prior_scale # 전달
        )
    except Exception as e:
        error_traceback = traceback.format_exc()
        logging.error(f"분석 함수(analyze_stock) 내에서 오류 발생: {e}\n{error_traceback}")
        return {"error": f"분석 중 오류 발생: {e}"}

# 분석 시작 버튼 로직
if analyze_button:
    if not ticker_input:
        results_placeholder.warning("종목 티커를 입력해주세요.")
    elif not api_keys_loaded:
        # API 키 로드 실패 시 경고 (에러는 위에서 이미 표시됨)
        results_placeholder.warning("API 키가 로드되지 않아 일부 기능이 제한될 수 있습니다.")
        # 키 없이 분석 시도 ( analyze_stock 내부에서 처리 예상 )
        ticker_processed = ticker_input.strip().upper()
        with st.spinner(f"{ticker_processed} 종목 분석 중 (제한된 기능)..."):
             results = run_cached_analysis(
                 ticker_processed, NEWS_API_KEY, FRED_API_KEY, # None이 전달될 수 있음
                 analysis_years, forecast_days, num_trend_periods_input,
                 changepoint_prior_input # 슬라이더 값 전달
             )
    else:
        ticker_processed = ticker_input.strip().upper()
        with st.spinner(f"{ticker_processed} 종목 분석 중..."):
            try:
                # 캐시 함수 호출 시 changepoint_prior_input 전달
                results = run_cached_analysis(
                    ticker_processed, NEWS_API_KEY, FRED_API_KEY,
                    analysis_years, forecast_days, num_trend_periods_input,
                    changepoint_prior_input # 슬라이더 값 전달
                )
                results_placeholder.empty() # 이전 결과 지우기

                # --- 결과 표시 ---
                if results and isinstance(results, dict) and "error" not in results:
                    st.header(f"📈 {ticker_processed} 분석 결과 (추세 민감도: {changepoint_prior_input:.3f})") # 설정값 표시

                    # 1. 기본 정보
                    st.subheader("요약 정보")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("현재가 (최근 종가)", f"${results.get('current_price', 'N/A')}")
                    col2.metric("분석 시작일", results.get('analysis_period_start', 'N/A'))
                    col3.metric("분석 종료일", results.get('analysis_period_end', 'N/A'))

                    # 2. 기본적 분석
                    st.subheader("📊 기업 기본 정보 (Fundamentals)")
                    fundamentals = results.get('fundamentals') # 세미콜론 제거
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
                        if industry != "N/A":
                            st.markdown(f"**산업:** {industry}")
                        if summary != "N/A":
                            with st.expander("회사 요약 보기"):
                                st.write(summary)
                        st.caption("Data Source: Yahoo Finance")
                    else:
                        st.warning("기업 기본 정보를 가져오지 못했습니다.")

                    # 3. 주요 재무 추세
                    st.subheader(f"📈 주요 재무 추세 (최근 {num_trend_periods_input} 분기)")
                    tab_titles = ["영업이익률(%)", "ROE(%)", "부채비율", "유동비율"]
                    tab1, tab2, tab3, tab4 = st.tabs(tab_titles)
                    with tab1:
                        margin_trend_data = results.get('operating_margin_trend')
                        if margin_trend_data and isinstance(margin_trend_data, list):
                            try:
                                df_margin = pd.DataFrame(margin_trend_data)
                                df_margin['Date'] = pd.to_datetime(df_margin['Date'])
                                df_margin.set_index('Date', inplace=True)
                                # 컬럼명 'Op Margin (%)' 확인 (stock_analysis.py 반환값 기준)
                                if 'Op Margin (%)' in df_margin.columns:
                                    st.line_chart(df_margin[['Op Margin (%)']])
                                    with st.expander("데이터 보기"):
                                        st.dataframe(df_margin[['Op Margin (%)']].style.format("{:.2f}%"), use_container_width=True)
                                else:
                                    st.error("결과에서 'Op Margin (%)' 컬럼을 찾을 수 없습니다.")
                            except Exception as e:
                                st.error(f"영업이익률 차트/데이터 표시 오류: {e}")
                        else:
                            st.info("영업이익률 추세 데이터 없음.")
                    with tab2:
                        roe_trend_data = results.get('roe_trend')
                        if roe_trend_data and isinstance(roe_trend_data, list):
                            try:
                                df_roe = pd.DataFrame(roe_trend_data)
                                df_roe['Date'] = pd.to_datetime(df_roe['Date'])
                                df_roe.set_index('Date', inplace=True)
                                st.line_chart(df_roe[['ROE (%)']])
                                with st.expander("데이터 보기"):
                                    st.dataframe(df_roe[['ROE (%)']].style.format({"ROE (%)": "{:.2f}%"}), use_container_width=True)
                            except Exception as e:
                                st.error(f"ROE 차트/데이터 표시 오류: {e}")
                        else:
                            st.info("ROE 추세 데이터 없음.")
                    with tab3:
                        debt_trend_data = results.get('debt_to_equity_trend')
                        if debt_trend_data and isinstance(debt_trend_data, list):
                            try:
                                df_debt = pd.DataFrame(debt_trend_data)
                                df_debt['Date'] = pd.to_datetime(df_debt['Date'])
                                df_debt.set_index('Date', inplace=True)
                                st.line_chart(df_debt[['D/E Ratio']])
                                with st.expander("데이터 보기"):
                                    st.dataframe(df_debt[['D/E Ratio']].style.format({"D/E Ratio": "{:.2f}"}), use_container_width=True)
                            except Exception as e:
                                st.error(f"부채비율 차트/데이터 표시 오류: {e}")
                        else:
                            st.info("부채비율 추세 데이터 없음.")
                    with tab4:
                        current_trend_data = results.get('current_ratio_trend')
                        if current_trend_data and isinstance(current_trend_data, list):
                            try:
                                df_current = pd.DataFrame(current_trend_data)
                                df_current['Date'] = pd.to_datetime(df_current['Date'])
                                df_current.set_index('Date', inplace=True)
                                st.line_chart(df_current[['Current Ratio']])
                                with st.expander("데이터 보기"):
                                    st.dataframe(df_current[['Current Ratio']].style.format({"Current Ratio": "{:.2f}"}), use_container_width=True)
                            except Exception as e:
                                st.error(f"유동비율 차트/데이터 표시 오류: {e}")
                        else:
                            st.info("유동비율 추세 데이터 없음.")
                    st.divider()

                    # 4. 기술적 분석 (차트)
                    st.subheader("기술적 분석 차트")
                    stock_chart_fig = results.get('stock_chart_fig')
                    if stock_chart_fig:
                        st.plotly_chart(stock_chart_fig, use_container_width=True)
                    else:
                        st.warning("주가 차트 생성 실패.")
                    st.divider()

                    # 5. 시장 심리
                    st.subheader("시장 심리 분석")
                    col_news, col_fng = st.columns([2, 1]) # 뉴스 영역 더 넓게
                    with col_news:
                        st.markdown("**📰 뉴스 감정 분석**")
                        news_sentiment = results.get('news_sentiment', ["뉴스 분석 정보 없음."])
                        if isinstance(news_sentiment, list) and len(news_sentiment) > 0:
                            st.info(news_sentiment[0]) # 헤더 (평균 또는 에러 메시지)
                            if len(news_sentiment) > 1: # 실제 뉴스 내용이 있을 때
                                with st.expander("최근 뉴스 목록 보기", expanded=False):
                                    for line in news_sentiment[1:]: # 수정: 표준 for loop 사용
                                        st.write(f"- {line}")
                        else:
                             # 문자열 에러 메시지 등 처리
                             st.write(str(news_sentiment)) # 안전하게 문자열로 변환
                    with col_fng:
                        st.markdown("**😨 공포-탐욕 지수**")
                        fng_index = results.get('fear_greed_index', "N/A")
                        if isinstance(fng_index, dict):
                            st.metric(label="현재 지수", value=fng_index.get('value', 'N/A'), delta=fng_index.get('classification', ''))
                        else:
                            st.write(fng_index) # "N/A" 등 문자열 출력
                    st.divider()

                    # 6. Prophet 예측 분석
                    st.subheader("Prophet 주가 예측")
                    forecast_fig = results.get('forecast_fig')
                    forecast_data_list = results.get('prophet_forecast')
                    if forecast_fig:
                        st.plotly_chart(forecast_fig, use_container_width=True)
                    elif isinstance(forecast_data_list, str): # 예측 불가 메시지 등
                        st.info(forecast_data_list)
                    else:
                        st.warning("예측 차트 생성 실패.")

                    if isinstance(forecast_data_list, list):
                        st.markdown("**📊 예측 데이터 (최근 10일)**")
                        df_fcst = pd.DataFrame(forecast_data_list)
                        df_fcst['ds'] = pd.to_datetime(df_fcst['ds']).dt.strftime('%Y-%m-%d')
                        st.dataframe(
                            df_fcst[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(10).style.format({'yhat': "{:.2f}", 'yhat_lower': "{:.2f}", 'yhat_upper': "{:.2f}"}),
                            use_container_width=True
                        ) # 세미콜론 제거

                    cv_plot_path = results.get('cv_plot_path')
                    if cv_plot_path and isinstance(cv_plot_path, str) and os.path.exists(cv_plot_path):
                        st.markdown("**📉 교차 검증 결과 (MAPE)**")
                        st.image(cv_plot_path, caption="MAPE (Mean Absolute Percentage Error, 낮을수록 예측 정확도 높음)")
                    elif cv_plot_path is None and isinstance(forecast_data_list, list): # 예측 성공했지만 CV 경로 없을 때
                        st.caption("교차 검증(CV) 결과 없음 (데이터 기간 부족 또는 오류).")
                    st.divider()

                    # 7. 리스크 트래커
                    st.subheader("🚨 리스크 트래커 (예측 기반)")
                    risk_days = 0 # 요약에서 사용하기 위해 초기화
                    max_loss_pct = 0
                    max_loss_amt = 0
                    if avg_price > 0 and isinstance(forecast_data_list, list):
                        df_pred = pd.DataFrame(forecast_data_list)
                        try:
                            # 필요한 컬럼 타입 변환 및 NaN 처리
                            df_pred['ds'] = pd.to_datetime(df_pred['ds'])
                            df_pred['yhat'] = pd.to_numeric(df_pred['yhat'], errors='coerce')
                            df_pred['yhat_lower'] = pd.to_numeric(df_pred['yhat_lower'], errors='coerce')
                            df_pred['yhat_upper'] = pd.to_numeric(df_pred['yhat_upper'], errors='coerce')
                            df_pred.dropna(subset=['yhat_lower'], inplace=True) # 하한선 없으면 리스크 계산 불가

                            if not df_pred.empty:
                                df_pred['평단가'] = avg_price
                                df_pred['리스크 여부'] = df_pred['yhat_lower'] < df_pred['평단가']
                                df_pred['예상 손실률'] = np.where(df_pred['리스크 여부'], ((df_pred['yhat_lower'] - df_pred['평단가']) / df_pred['평단가']) * 100, 0)

                                if quantity > 0:
                                    df_pred['예상 손실액'] = np.where(df_pred['리스크 여부'], (df_pred['yhat_lower'] - df_pred['평단가']) * quantity, 0)
                                else:
                                    df_pred['예상 손실액'] = 0

                                risk_days = df_pred['리스크 여부'].sum()
                                max_loss_pct = df_pred['예상 손실률'].min() if risk_days > 0 else 0
                                max_loss_amt = df_pred['예상 손실액'].min() if risk_days > 0 and quantity > 0 else 0

                                st.markdown("##### 리스크 요약")
                                col_r1, col_r2, col_r3 = st.columns(3)
                                col_r1.metric("⚠️ < 평단가 일수", f"{risk_days}일 / {forecast_days}일")
                                col_r2.metric("📉 Max 예측 손실률", f"{max_loss_pct:.2f}%")
                                if quantity > 0:
                                    col_r3.metric("💸 Max 예측 손실액", f"${max_loss_amt:,.2f}")

                                if risk_days > 0:
                                    st.warning(f"향후 {forecast_days}일 예측 기간 중 **{risk_days}일** 동안 예측 하한선이 평단가(${avg_price:.2f})보다 낮을 수 있습니다. (예상 최대 손실률: **{max_loss_pct:.2f}%**).")
                                else:
                                    st.success(f"향후 {forecast_days}일 동안 예측 하한선이 평단가(${avg_price:.2f})보다 낮아질 가능성은 현재 예측되지 않았습니다.")

                                # 리스크 비교 차트
                                st.markdown("##### 평단가 vs 예측 구간 비교 차트")
                                fig_risk = go.Figure()
                                # 예측 구간 (상한선/하한선)
                                fig_risk.add_trace(go.Scatter(x=df_pred['ds'], y=df_pred['yhat_upper'], mode='lines', line_color='rgba(0,100,80,0.2)', name='Upper Bound'))
                                fig_risk.add_trace(go.Scatter(x=df_pred['ds'], y=df_pred['yhat_lower'], mode='lines', line_color='rgba(0,100,80,0.2)', name='Lower Bound', fill='tonexty', fillcolor='rgba(0,100,80,0.1)')) # 구간 채우기
                                # 예측 중앙값 (점선)
                                fig_risk.add_trace(go.Scatter(x=df_pred['ds'], y=df_pred['yhat'], mode='lines', line=dict(dash='dash', color='rgba(0,100,80,0.6)'), name='Forecast (yhat)'))
                                # 평단가 라인
                                fig_risk.add_hline(y=avg_price, line_dash="dot", line_color="red", annotation_text=f"평단가: ${avg_price:.2f}", annotation_position="bottom right")
                                # 리스크 발생일 표시 (마커)
                                df_risk_periods = df_pred[df_pred['리스크 여부']]
                                if not df_risk_periods.empty:
                                    fig_risk.add_trace(go.Scatter(x=df_risk_periods['ds'], y=df_risk_periods['yhat_lower'], mode='markers', marker_symbol='x', marker_color='red', name='Risk Day (Lower < Avg Price)'))

                                fig_risk.update_layout(title=f"{ticker_processed} 예측 구간 vs 평단가 비교", xaxis_title="날짜", yaxis_title="가격", hovermode="x unified", margin=dict(l=20, r=20, t=40, b=20))
                                st.plotly_chart(fig_risk, use_container_width=True)

                                # 리스크 상세 데이터
                                if risk_days > 0:
                                    with st.expander(f"리스크 예측일 상세 데이터 보기 ({risk_days}일)"):
                                        df_risk_days_display = df_pred[df_pred['리스크 여부']].copy()
                                        df_risk_days_display['ds'] = df_risk_days_display['ds'].dt.strftime('%Y-%m-%d')
                                        cols_to_show = ['ds', 'yhat_lower', '평단가', '예상 손실률']
                                        if quantity > 0:
                                            cols_to_show.append('예상 손실액')
                                        st.dataframe(
                                            df_risk_days_display[cols_to_show].style.format({"yhat_lower":"{:.2f}", "평단가":"{:.2f}", "예상 손실률":"{:.2f}%", "예상 손실액":"${:,.2f}"}),
                                            use_container_width=True
                                        )
                            else:
                                st.info("예측 하한선 데이터가 유효하지 않아 리스크 분석을 수행할 수 없습니다.")

                        except Exception as risk_calc_err:
                            st.error(f"리스크 트래커 계산/표시 중 오류 발생: {risk_calc_err}")
                            logging.error(f"Risk tracker error: {traceback.format_exc()}") # 상세 오류 로깅
                    elif avg_price <= 0:
                        st.info("⬅️ 사이드바에서 '평단가'를 0보다 큰 값으로 입력하시면 리스크 분석 결과를 볼 수 있습니다.")
                    else: # forecast_data_list가 list가 아닐 경우
                        st.warning("Prophet 예측 데이터가 유효하지 않아 리스크 분석을 수행할 수 없습니다.")
                    st.divider()

                    # 8. 자동 분석 결과 요약
                    st.subheader("🧐 자동 분석 결과 요약 (참고용)")
                    summary_points = [] # 변수명 변경

                    # 예측 요약
                    if isinstance(forecast_data_list, list) and len(forecast_data_list) > 0:
                        try:
                            start_pred = forecast_data_list[0]['yhat']
                            end_pred = forecast_data_list[-1]['yhat']
                            # 예측값 타입 체크 및 변환
                            start_pred = float(start_pred) if pd.notna(start_pred) else None
                            end_pred = float(end_pred) if pd.notna(end_pred) else None

                            if start_pred is not None and end_pred is not None:
                                trend_obs = "상승" if end_pred > start_pred else "하락" if end_pred < start_pred else "횡보"
                                last_forecast = forecast_data_list[-1]
                                lower = float(last_forecast['yhat_lower']) if pd.notna(last_forecast['yhat_lower']) else 'N/A'
                                upper = float(last_forecast['yhat_upper']) if pd.notna(last_forecast['yhat_upper']) else 'N/A'
                                lower_str = f"{lower:.2f}" if isinstance(lower, float) else 'N/A'
                                upper_str = f"{upper:.2f}" if isinstance(upper, float) else 'N/A'
                                summary_points.append(f"- **예측:** 향후 {forecast_days}일간 **{trend_obs}** 추세 예상 (최종일 예측 범위: {lower_str} ~ {upper_str}).")
                            else:
                                summary_points.append("- 예측: 최종 예측값 유효하지 않음.")
                        except Exception as e:
                             logging.warning(f"예측 요약 생성 오류: {e}")
                             summary_points.append("- 예측: 요약 생성 중 오류 발생.")

                    # 뉴스 요약
                    if isinstance(news_sentiment, list) and len(news_sentiment) > 0:
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

                    # 재무 추세 요약
                    trend_summary_parts = []
                    # 컬럼명 재확인 및 마지막 값 접근 방식 수정
                    if results.get('operating_margin_trend') and results['operating_margin_trend']: trend_summary_parts.append(f"최근 영업이익률 {results['operating_margin_trend'][-1].get('Op Margin (%)', 'N/A'):.2f}%")
                    if results.get('roe_trend') and results['roe_trend']: trend_summary_parts.append(f"ROE {results['roe_trend'][-1].get('ROE (%)', 'N/A'):.2f}%")
                    if results.get('debt_to_equity_trend') and results['debt_to_equity_trend']: trend_summary_parts.append(f"부채비율 {results['debt_to_equity_trend'][-1].get('D/E Ratio', 'N/A'):.2f}")
                    if results.get('current_ratio_trend') and results['current_ratio_trend']: trend_summary_parts.append(f"유동비율 {results['current_ratio_trend'][-1].get('Current Ratio', 'N/A'):.2f}")
                    if trend_summary_parts: summary_points.append(f"- **최근 재무:** {', '.join(trend_summary_parts)}.")

                    # 리스크 요약 (위에서 계산된 변수 사용)
                    if avg_price > 0 and isinstance(forecast_data_list, list):
                        if risk_days > 0:
                            summary_points.append(f"- **리스크:** 예측상 향후 {forecast_days}일 중 **{risk_days}일**은 평단가 하회 가능성 있음 (최대 **{max_loss_pct:.2f}%** 손실률 예상).")
                        else:
                            summary_points.append(f"- **리스크:** 예측상 향후 {forecast_days}일간 평단가(${avg_price:.2f}) 하회 가능성은 낮아 보임.")
                    elif avg_price > 0:
                         summary_points.append("- 리스크: 평단가는 입력되었으나, 예측 데이터 부족/오류로 분석 불가.")


                    # 최종 요약 출력
                    if summary_points:
                        st.markdown("\n".join(summary_points))
                        st.caption("⚠️ **주의:** 이 요약은 자동 생성된 정보이며 투자 조언이 아닙니다. 모든 투자 결정과 책임은 사용자 본인에게 있습니다.")
                    else:
                        st.write("분석 요약을 생성할 수 없습니다.")

                elif results is None or ("error" in results and results["error"]): # 오류 처리 강화
                    error_msg = results.get("error", "알 수 없는 오류") if isinstance(results, dict) else "분석 결과 없음"
                    results_placeholder.error(f"분석 실패: {error_msg}")
                else: # results가 예상치 못한 형태일 경우
                    results_placeholder.error("분석 결과를 처리하는 중 예기치 않은 오류가 발생했습니다.")
            except Exception as e:
                # 앱 실행 자체의 예외 처리
                error_traceback = traceback.format_exc()
                logging.error(f"앱 실행 중 예기치 않은 오류 발생: {e}\n{error_traceback}")
                results_placeholder.error(f"앱 실행 중 오류 발생: {e}")
                st.exception(e) # Streamlit의 내장 예외 표시 기능 사용
else:
    # 앱 초기 상태 메시지
    results_placeholder.info("⬅️ 왼쪽 사이드바에서 분석 설정을 완료한 후 '분석 시작' 버튼을 클릭하세요.")