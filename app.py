# app.py (최종 버전)

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import os
from dotenv import load_dotenv
import traceback # 상세 오류 로깅 위해 추가
import plotly.graph_objects as go # 리스크 차트 생성 위해 추가
import numpy as np # 리스크 계산 시 np.where 사용 위해 추가

# --- 기본 경로 설정 ---
try:
    # 스크립트 실행 시 __file__ 변수 사용
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # 대화형 환경 등 __file__ 없을 시 현재 작업 디렉토리 사용
    BASE_DIR = os.getcwd()

# --- 분석 로직 가져오기 ---
try:
    import stock_analysis as sa
except ModuleNotFoundError:
    st.error(f"오류: stock_analysis.py 파일을 찾을 수 없습니다. app.py와 같은 폴더({BASE_DIR})에 저장해주세요.")
    st.stop() # 앱 실행 중지 (파일 없으면 진행 불가)
except ImportError as ie:
     st.error(f"stock_analysis.py를 import하는 중 오류 발생: {ie}")
     st.info("필요한 라이브러리가 모두 설치되었는지 확인하세요. (예: pip install -r requirements.txt)")
     st.stop() # 라이브러리 문제 시 진행 불가
except Exception as import_err: # 그 외 import 관련 예외 처리
     st.error(f"stock_analysis.py import 중 예외 발생: {import_err}")
     st.info("stock_analysis.py 파일 내부에 오류가 없는지 확인해보세요.")
     st.stop()

# --- Streamlit 페이지 설정 ---
st.set_page_config(page_title="주식 분석 및 리스크 트래커", layout="wide", initial_sidebar_state="expanded")

st.title("📊 주식 분석 및 예측 도구 v1.1") # 버전 표시 추가 (선택 사항)
st.markdown("과거 데이터 분석, 미래 예측과 함께 기업 기본 정보 및 보유 종목 리스크를 추적합니다.")

# --- API 키 로드 (Secrets 우선, .env 차선) ---
NEWS_API_KEY = None
FRED_API_KEY = None
api_keys_loaded = False
secrets_available = hasattr(st, 'secrets') # Streamlit Cloud 환경 확인

# 사이드바 상단에 로드 상태 표시용 컨테이너
sidebar_status = st.sidebar.empty()

if secrets_available:
    try:
        NEWS_API_KEY = st.secrets["NEWS_API_KEY"]
        FRED_API_KEY = st.secrets["FRED_API_KEY"]
        # sidebar_status.success("API 키 로드 완료 (Secrets)") # 성공 시 메시지 숨기기 가능
        api_keys_loaded = True
    except KeyError:
         sidebar_status.error("Secrets에 API 키 없음. Cloud 설정 확인.")
    except Exception as e:
        sidebar_status.error(f"Secrets 로드 오류: {e}")
else:
    sidebar_status.info("로컬 환경 감지. .env 파일 확인...")
    try:
        dotenv_path = os.path.join(BASE_DIR, '.env')
        if os.path.exists(dotenv_path):
            load_dotenv(dotenv_path=dotenv_path)
            NEWS_API_KEY = os.getenv("NEWS_API_KEY")
            FRED_API_KEY = os.getenv("FRED_API_KEY")
            if NEWS_API_KEY and FRED_API_KEY:
                # sidebar_status.success("API 키 로드 완료 (.env)") # 성공 시 메시지 숨기기 가능
                api_keys_loaded = True
            else:
                sidebar_status.error(".env 파일에서 API 키 찾을 수 없음.")
        else:
            sidebar_status.error(f".env 파일 없음: {dotenv_path}")
    except Exception as e:
        sidebar_status.error(f".env 파일 로드 오류: {e}")

if not api_keys_loaded:
    st.sidebar.error("API 키 로드 실패! 일부 기능 제한됨.") # 최종 실패 시 에러 표시


# --- 사이드바 설정 ---
with st.sidebar:
    st.header("⚙️ 분석 설정")
    ticker_input = st.text_input("종목 티커 입력 (예: AAPL, 005930.KS)", value="AAPL")
    analysis_years = st.select_slider("분석 기간 선택", options=[1, 2, 3, 5, 7, 10], value=2)
    st.caption(f"과거 {analysis_years}년 데이터를 분석합니다.")
    forecast_days = st.number_input("예측 기간 (일)", min_value=7, max_value=90, value=30, step=7)
    st.caption(f"향후 {forecast_days}일 후까지 예측합니다.")
    num_trend_periods_input = st.number_input("수익성 추세 분기 수", min_value=2, max_value=12, value=4, step=1)
    st.caption(f"최근 {num_trend_periods_input}개 분기의 수익성 추세를 계산합니다.")
    st.divider()
    st.header("💰 보유 정보 입력 (선택)")
    avg_price = st.number_input("평단가 (Average Price)", min_value=0.0, value=0.0, format="%.2f")
    quantity = st.number_input("보유 수량 (Quantity)", min_value=0, value=0, step=1)
    st.caption("평단가를 입력하면 예측 기반 리스크 분석이 활성화됩니다.")
    st.divider()
    analyze_button = st.button("🚀 분석 시작!")

# --- 메인 화면 ---
results_placeholder = st.container() # 결과 표시 영역 선언

# 캐시된 분석 함수 정의
@st.cache_data(ttl=timedelta(hours=1)) # 1시간 캐시
def run_cached_analysis(ticker, news_key, fred_key, years, days, num_trend_periods):
    """캐싱을 위한 분석 함수 래퍼"""
    # API 키 유효성 검사는 함수 호출 전에 하는 것이 더 나을 수 있음
    if not news_key or not fred_key:
         # 캐싱된 함수 내에서 st 요소를 직접 사용하는 것은 권장되지 않음
         # print("에러: API 키 미유효 (캐시 함수)") # 로그 기록 방식으로 변경 권장
         logging.error("API 키가 유효하지 않아 분석을 진행할 수 없습니다 (캐시 함수).")
         return {"error": "API 키 미유효"} # 오류 상태 반환
    try:
        return sa.analyze_stock(ticker, news_key, fred_key,
                                analysis_period_years=years,
                                forecast_days=days,
                                num_trend_periods=num_trend_periods)
    except Exception as e:
        logging.error(f"분석 함수(stock_analysis.py) 실행 중 오류 발생: {e}")
        logging.error(traceback.format_exc())
        return {"error": f"분석 중 오류 발생: {e}"} # 오류 상태 반환

# 분석 시작 버튼 로직
if analyze_button:
    if not ticker_input:
        results_placeholder.warning("종목 티커를 입력해주세요.")
    elif not api_keys_loaded:
        results_placeholder.error("API 키가 로드되지 않아 분석을 실행할 수 없습니다. Secrets 또는 .env 파일을 확인하세요.")
    else:
        ticker_processed = ticker_input.strip().upper() # 입력값 정리
        with st.spinner(f"{ticker_processed} 종목 분석 중... 잠시만 기다려주세요..."):
            try:
                # 캐시된 함수 호출 (API 키 전달)
                results = run_cached_analysis(ticker_processed, NEWS_API_KEY, FRED_API_KEY,
                                              analysis_years, forecast_days, num_trend_periods_input)

                results_placeholder.empty() # 이전 결과 표시 영역 비우기

                # 결과가 정상이면 표시, 오류가 포함된 딕셔너리이면 에러 표시
                if results and isinstance(results, dict) and "error" not in results:
                    st.header(f"📈 {ticker_processed} 분석 결과")

                    # 1. 기본 정보
                    st.subheader("요약 정보")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("현재가 (최근 종가)", f"${results.get('current_price', 'N/A')}")
                    col2.metric("분석 시작일", results.get('analysis_period_start', 'N/A'))
                    col3.metric("분석 종료일", results.get('analysis_period_end', 'N/A'))

                    # 2. 기본적 분석(Fundamental) 데이터 표시
                    st.subheader("📊 기업 기본 정보 (Fundamentals)")
                    fundamentals = results.get('fundamentals')
                    if fundamentals and isinstance(fundamentals, dict) and fundamentals.get("시가총액", "N/A") != "N/A": # 데이터 유효성 체크 강화
                        colf1, colf2, colf3 = st.columns(3)
                        with colf1: st.metric("시가총액", fundamentals.get("시가총액", "N/A")); st.metric("PER", fundamentals.get("PER", "N/A"))
                        with colf2: st.metric("EPS (주당순이익)", fundamentals.get("EPS", "N/A")); st.metric("Beta (베타)", fundamentals.get("베타", "N/A"))
                        with colf3: st.metric("배당수익률", fundamentals.get("배당수익률", "N/A")); st.metric("업종", fundamentals.get("업종", "N/A"))
                        industry = fundamentals.get("산업", "N/A"); summary = fundamentals.get("요약", "N/A")
                        if industry != "N/A": st.markdown(f"**산업:** {industry}")
                        if summary != "N/A":
                            with st.expander("회사 요약 보기"): st.write(summary)
                        st.caption("데이터 출처: Yahoo Finance")
                    else:
                        st.warning("기업 기본 정보(펀더멘탈)를 가져오지 못했습니다. (티커 유효성 또는 데이터 제공 여부 확인 필요)")

                    # 3. 수익성 추세 (영업이익률 & ROE)
                    st.subheader(f"📈 수익성 추세 (최근 {num_trend_periods_input} 분기)")
                    # 영업이익률
                    st.markdown("##### 영업이익률 (Operating Margin %)")
                    margin_trend_data = results.get('operating_margin_trend')
                    if margin_trend_data and isinstance(margin_trend_data, list):
                        try:
                            df_margin = pd.DataFrame(margin_trend_data); df_margin['Date'] = pd.to_datetime(df_margin['Date']); df_margin.set_index('Date', inplace=True)
                            st.line_chart(df_margin[['Operating Margin (%)']])
                            with st.expander("영업이익률 데이터 보기"): st.dataframe(df_margin.style.format({"Operating Margin (%)": "{:.2f}%"}), use_container_width=True)
                        except Exception as margin_err: st.error(f"영업이익률 추세 처리/표시 오류: {margin_err}")
                    else: st.info("영업이익률 추세 데이터 없음.")
                    # ROE
                    st.markdown("##### ROE (자기자본이익률 %)")
                    roe_trend_data = results.get('roe_trend')
                    if roe_trend_data and isinstance(roe_trend_data, list):
                        try:
                            df_roe = pd.DataFrame(roe_trend_data); df_roe['Date'] = pd.to_datetime(df_roe['Date']); df_roe.set_index('Date', inplace=True)
                            st.line_chart(df_roe[['ROE (%)']])
                            with st.expander("ROE 데이터 보기"): st.dataframe(df_roe.style.format({"ROE (%)": "{:.2f}%"}), use_container_width=True)
                        except Exception as roe_err: st.error(f"ROE 추세 처리/표시 오류: {roe_err}")
                    else: st.info("ROE 추세 데이터 없음.")
                    st.divider()

                    # 4. 기술적 분석 (차트)
                    st.subheader("기술적 분석 차트")
                    stock_chart_fig = results.get('stock_chart_fig')
                    if stock_chart_fig: st.plotly_chart(stock_chart_fig, use_container_width=True)
                    else: st.warning("주가 차트 생성 실패.")
                    st.divider()

                    # 5. 시장 심리
                    st.subheader("시장 심리 분석")
                    col_news, col_fng = st.columns([2, 1])
                    with col_news:
                        st.markdown("**📰 뉴스 감정 분석**")
                        news_sentiment = results.get('news_sentiment', ["정보 없음"])
                        if isinstance(news_sentiment, list) and len(news_sentiment) > 0:
                            st.info(news_sentiment[0])
                            with st.expander("최근 뉴스 목록 보기", expanded=False):
                                for line in news_sentiment[1:]: # 수정된 for 반복문
                                    st.write(f"- {line}")
                        else: st.write(news_sentiment)
                    with col_fng:
                        st.markdown("**😨 공포-탐욕 지수**")
                        fng_index = results.get('fear_greed_index', "N/A")
                        if isinstance(fng_index, dict): st.metric(label="현재 지수", value=fng_index.get('value', 'N/A'), delta=fng_index.get('classification', ''))
                        else: st.write(fng_index)
                    st.divider()

                    # 6. Prophet 예측 분석
                    st.subheader("Prophet 주가 예측")
                    forecast_fig = results.get('forecast_fig'); forecast_data_list = results.get('prophet_forecast')
                    if forecast_fig: st.plotly_chart(forecast_fig, use_container_width=True)
                    elif isinstance(forecast_data_list, str): st.info(forecast_data_list) # "예측 실패" 등 메시지
                    else: st.warning("예측 차트 생성 실패.")
                    if isinstance(forecast_data_list, list): # 예측 성공 시 테이블 표시
                        st.markdown("**📊 예측 데이터 (최근 10일)**")
                        df_fcst = pd.DataFrame(forecast_data_list); df_fcst['ds'] = pd.to_datetime(df_fcst['ds']).dt.strftime('%Y-%m-%d');
                        st.dataframe(df_fcst[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(10).style.format({'yhat': "{:.2f}", 'yhat_lower': "{:.2f}", 'yhat_upper': "{:.2f}"}), use_container_width=True)
                    cv_plot_path = results.get('cv_plot_path')
                    if cv_plot_path and os.path.exists(cv_plot_path): st.markdown("**📉 교차 검증 결과 (MAPE)**"); st.image(cv_plot_path, caption="MAPE (낮을수록 좋음)")
                    st.divider()

                    # 7. 리스크 트래커
                    st.subheader("🚨 리스크 트래커 (예측 기반)")
                    # ... (이전과 동일한 리스크 트래커 로직) ...
                    if avg_price > 0 and isinstance(forecast_data_list, list):
                        df_pred = pd.DataFrame(forecast_data_list);
                        try:
                            df_pred['ds'] = pd.to_datetime(df_pred['ds']); df_pred['yhat_lower'] = pd.to_numeric(df_pred['yhat_lower'], errors='coerce'); df_pred.dropna(subset=['yhat_lower'], inplace=True)
                            df_pred['평단가'] = avg_price; df_pred['리스크 여부'] = df_pred['yhat_lower'] < df_pred['평단가']; df_pred['예상 손실률'] = np.where(df_pred['리스크 여부'], ((df_pred['yhat_lower'] - df_pred['평단가']) / df_pred['평단가']) * 100, 0)
                            if quantity > 0: df_pred['예상 손실액'] = np.where(df_pred['리스크 여부'], (df_pred['yhat_lower'] - df_pred['평단가']) * quantity, 0)
                            else: df_pred['예상 손실액'] = 0
                            risk_days = df_pred['리스크 여부'].sum(); max_loss_pct = df_pred['예상 손실률'].min() if risk_days > 0 else 0; max_loss_amt = df_pred['예상 손실액'].min() if risk_days > 0 and quantity > 0 else 0
                            st.markdown("##### 리스크 요약"); col_r1, col_r2, col_r3 = st.columns(3); col_r1.metric("⚠️ < 평단가 일수", f"{risk_days}일/{forecast_days}일"); col_r2.metric("📉 Max 손실률", f"{max_loss_pct:.2f}%");
                            if quantity > 0: col_r3.metric("💸 Max 손실액", f"${max_loss_amt:,.2f}")
                            if risk_days > 0: st.warning(f"향후 {forecast_days}일 중 **{risk_days}일** 동안 예측 하한선 < 평단가(${avg_price:.2f}) (최대 **{max_loss_pct:.2f}%** 손실률).")
                            else: st.success(f"향후 {forecast_days}일 동안 예측 하한선이 평단가(${avg_price:.2f})보다 낮아질 가능성은 예측되지 않음.")
                            st.markdown("##### 평단가 vs 예측 구간 비교 차트"); fig_risk = go.Figure(); fig_risk.add_trace(go.Scatter(x=df_pred['ds'], y=pd.to_numeric(df_pred['yhat_upper'], errors='coerce'), mode='lines', line_color='rgba(0,100,80,0.2)', name='Upper')); fig_risk.add_trace(go.Scatter(x=df_pred['ds'], y=df_pred['yhat_lower'], mode='lines', line_color='rgba(0,100,80,0.2)', name='Lower', fill='tonexty', fillcolor='rgba(0,100,80,0.1)')); fig_risk.add_trace(go.Scatter(x=df_pred['ds'], y=pd.to_numeric(df_pred['yhat'], errors='coerce'), mode='lines', line=dict(dash='dash', color='rgba(0,100,80,0.6)'), name='Forecast')); fig_risk.add_hline(y=avg_price, line_dash="dot", line_color="red", annotation_text=f"평단가: ${avg_price:.2f}", annotation_position="bottom right"); df_risk_periods = df_pred[df_pred['리스크 여부']]
                            if not df_risk_periods.empty: fig_risk.add_trace(go.Scatter(x=df_risk_periods['ds'], y=df_risk_periods['yhat_lower'], mode='markers', marker_symbol='x', marker_color='red', name='Risk Day'))
                            fig_risk.update_layout(title=f"{ticker_processed} 예측 구간 vs 평단가", xaxis_title="날짜", yaxis_title="가격", hovermode="x unified", margin=dict(l=20, r=20, t=40, b=20)); st.plotly_chart(fig_risk, use_container_width=True)
                            if risk_days > 0:
                                 with st.expander(f"리스크 예측일 상세 보기 ({risk_days}일)"):
                                     df_risk_days_display = df_pred[df_pred['리스크 여부']].copy(); df_risk_days_display['ds'] = df_risk_days_display['ds'].dt.strftime('%Y-%m-%d'); cols_to_show = ['ds', 'yhat_lower', '평단가', '예상 손실률'];
                                     if quantity > 0: cols_to_show.append('예상 손실액')
                                     st.dataframe(df_risk_days_display[cols_to_show].style.format({"yhat_lower":"{:.2f}", "평단가":"{:.2f}", "예상 손실률":"{:.2f}%", "예상 손실액":"${:,.2f}"}), use_container_width=True)
                        except Exception as risk_calc_err: st.error(f"리스크 트래커 계산/표시 오류: {risk_calc_err}")
                    elif avg_price <= 0: st.info("⬅️ 사이드바에서 '평단가'를 0보다 큰 값으로 입력하시면 리스크 분석을 볼 수 있습니다.")
                    else: st.warning("Prophet 예측 데이터 유효하지 않아 리스크 분석 불가.")
                    st.divider()


                    # 8. 자동 분석 결과 요약
                    # ... (ROE 요약 추가됨) ...
                    st.subheader("🧐 자동 분석 결과 요약 (참고용)"); summary = []
                    if isinstance(forecast_data_list, list) and len(forecast_data_list) > 0:
                        try: start_pred = forecast_data_list[0]['yhat']; end_pred = forecast_data_list[-1]['yhat']; trend_obs = "상승" if end_pred > start_pred else "하락" if end_pred < start_pred else "횡보"; summary.append(f"- Prophet 예측: 향후 {forecast_days}일간 **{trend_obs}** 추세 예상 (${forecast_data_list[-1]['yhat_lower']:.2f} ~ ${forecast_data_list[-1]['yhat_upper']:.2f}).")
                        except: summary.append("- 예측 요약 오류.")
                    if isinstance(news_sentiment, list) and len(news_sentiment) > 0:
                         try: score_part = news_sentiment[0].split(":")[-1].strip(); avg_score = float(score_part); sentiment_desc = "긍정적" if avg_score > 0.05 else "부정적" if avg_score < -0.05 else "중립적"; summary.append(f"- 뉴스 감성: 평균 {avg_score:.2f}, **{sentiment_desc}** 분위기.")
                         except: summary.append("- 뉴스 감정 요약 오류.")
                    if isinstance(fng_index, dict): summary.append(f"- 시장 심리: F&G {fng_index.get('value', 'N/A')} (**{fng_index.get('classification', 'N/A')}**).")
                    if fundamentals and isinstance(fundamentals, dict): per_val = fundamentals.get("PER", "N/A"); sector_val = fundamentals.get("업종", "N/A"); fund_summary = [];
                    if per_val != "N/A": fund_summary.append(f"PER {per_val}");
                    if sector_val != "N/A": fund_summary.append(f"'{sector_val}' 업종");
                    if fund_summary: summary.append(f"- 기업 정보: {', '.join(fund_summary)}.")
                    # ROE 추세 요약 추가
                    roe_trend_data = results.get('roe_trend') # 요약 위해 다시 가져옴
                    if roe_trend_data and isinstance(roe_trend_data, list) and len(roe_trend_data) >= 2:
                        try:
                            first_roe = pd.to_numeric(roe_trend_data[0]['ROE (%)'], errors='coerce'); last_roe = pd.to_numeric(roe_trend_data[-1]['ROE (%)'], errors='coerce')
                            if pd.notna(first_roe) and pd.notna(last_roe):
                                roe_trend_desc = "개선" if last_roe > first_roe else "악화" if last_roe < first_roe else "유지"
                                summary.append(f"- 수익성(ROE): 최근 {len(roe_trend_data)}분기 **{roe_trend_desc}** 추세 (최근 {last_roe:.2f}%).")
                        except Exception as e: logging.warning(f"ROE 요약 생성 중 오류: {e}")

                    if avg_price > 0 and isinstance(forecast_data_list, list) and 'risk_days' in locals(): # risk_days 변수 존재 확인 추가
                        if risk_days > 0: summary.append(f"- 리스크: 예측 하한선 기준, **{risk_days}일** 평단가 하회 가능성 (최대 손실률 {max_loss_pct:.2f}%).")
                        else: summary.append(f"- 리스크: 예측 하한선 기준, 평단가 하회 리스크 없음.")
                    if summary: st.markdown("\n".join(summary)); st.caption("⚠️ **주의:** 투자 조언 아님. 모든 결정은 본인 책임.")
                    else: st.write("분석 요약 생성 불가.")

                elif results is None or "error" in results: # 캐시 함수에서 오류 반환 시
                    error_msg = results.get("error", "알 수 없는 분석 오류") if isinstance(results, dict) else "알 수 없는 분석 오류"
                    results_placeholder.error(f"분석 실패: {error_msg}")
                else:
                    results_placeholder.error("분석 결과 처리 실패.") # 예기치 않은 결과 타입
            except Exception as e:
                results_placeholder.error(f"앱 실행 중 예기치 않은 오류 발생: {e}")
                st.exception(e) # Streamlit에서 오류 상세 정보 표시
else:
    results_placeholder.info("⬅️ 왼쪽 사이드바에서 분석 설정을 완료한 후 '분석 시작' 버튼을 클릭하세요.")