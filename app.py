# app.py (리스크 트래커 기능 추가)

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import os
from dotenv import load_dotenv
import traceback # 상세 오류 로깅 위해 추가
import plotly.graph_objects as go # 리스크 차트 생성 위해 추가

# --- 기본 경로 설정 ---
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()

# --- 분석 로직 가져오기 ---
try:
    import stock_analysis as sa
except ModuleNotFoundError:
    st.error(f"오류: stock_analysis.py 파일을 찾을 수 없습니다. app.py와 같은 폴더({BASE_DIR})에 저장해주세요.")
    st.stop()
except ImportError as ie:
     st.error(f"stock_analysis.py를 import하는 중 오류 발생: {ie}")
     st.info("필요한 라이브러리가 모두 설치되었는지 확인하세요. (예: pip install -r requirements.txt)")
     st.stop()

# --- Streamlit 페이지 설정 ---
st.set_page_config(page_title="주식 분석 및 리스크 트래커", layout="wide", initial_sidebar_state="expanded")

st.title("📊 주식 분석 및 리스크 트래커")
st.markdown("과거 데이터 분석, 미래 예측과 함께 보유 종목의 예측 기반 리스크를 추적합니다.")

# --- API 키 로드 (Secrets 우선, .env 차선) ---
NEWS_API_KEY = None
FRED_API_KEY = None
api_keys_loaded = False
secrets_available = hasattr(st, 'secrets') # Streamlit Cloud 환경인지 간접 확인

if secrets_available:
    try:
        NEWS_API_KEY = st.secrets["NEWS_API_KEY"]
        FRED_API_KEY = st.secrets["FRED_API_KEY"]
        # st.sidebar.success("API 키 로드 완료 (Secrets)") # 성공 메시지는 생략 가능
        api_keys_loaded = True
    except KeyError:
        # Secrets는 있지만 키가 없는 경우
         st.sidebar.error("Secrets에 API 키가 없습니다. Streamlit Cloud 설정을 확인하세요.")
    except Exception as e:
        # 기타 Secrets 관련 오류
        st.sidebar.error(f"Secrets 로드 중 오류: {e}")
else:
    # 로컬 환경 등 Secrets 접근 불가 시 .env 파일 시도
    st.sidebar.info("Secrets 기능 사용 불가. 로컬 .env 파일을 확인합니다.")
    try:
        dotenv_path = os.path.join(BASE_DIR, '.env')
        if os.path.exists(dotenv_path):
            load_dotenv(dotenv_path=dotenv_path)
            NEWS_API_KEY = os.getenv("NEWS_API_KEY")
            FRED_API_KEY = os.getenv("FRED_API_KEY")
            if NEWS_API_KEY and FRED_API_KEY:
                # st.sidebar.success("API 키 로드 완료 (.env)")
                api_keys_loaded = True
            else:
                st.sidebar.error("로컬 .env 파일에서 API 키를 찾을 수 없습니다.")
        else:
            st.sidebar.error(f"로컬 .env 파일을 찾을 수 없습니다: {dotenv_path}")
    except Exception as e:
        st.sidebar.error(f".env 파일 로드 중 오류: {e}")

if not api_keys_loaded:
    st.sidebar.error("API 키 로드 실패! 뉴스/거시경제 데이터 관련 기능이 제한됩니다.")

# --- 사이드바 설정 ---
with st.sidebar:
    st.header("⚙️ 분석 설정")
    ticker_input = st.text_input("종목 티커 입력 (예: AAPL, 005930.KS)", value="AAPL")
    analysis_years = st.select_slider("분석 기간 선택", options=[1, 2, 3, 5, 7, 10], value=2)
    st.caption(f"과거 {analysis_years}년 데이터를 분석합니다.")
    forecast_days = st.number_input("예측 기간 (일)", min_value=7, max_value=90, value=30, step=7) # 예측 기간 조금 늘림
    st.caption(f"향후 {forecast_days}일 후까지 예측합니다.")

    st.divider() # 구분선

    # --- 리스크 트래커 입력 ---
    st.header("💰 보유 정보 입력 (선택)")
    avg_price = st.number_input("평단가 (Average Price)", min_value=0.0, value=0.0, format="%.2f")
    quantity = st.number_input("보유 수량 (Quantity)", min_value=0, value=0, step=1)
    st.caption("평단가를 입력하면 예측 기반 리스크 분석이 활성화됩니다. 보유 수량은 예상 손실액 계산에 사용됩니다.")
    # --------------------------

    st.divider()
    analyze_button = st.button("🚀 분석 시작!")

# --- 메인 화면 ---
results_placeholder = st.container()

# 캐시된 분석 함수 정의 (API 키를 인자로 받도록 수정)
@st.cache_data(ttl=timedelta(hours=1))
def run_cached_analysis(ticker, news_key, fred_key, years, days):
    """캐싱을 위한 분석 함수 래퍼"""
    if not news_key or not fred_key:
         st.error("API 키가 유효하지 않아 분석을 진행할 수 없습니다.")
         return None
    try:
        # analyze_stock 호출 시 fred_key 전달
        return sa.analyze_stock(ticker, news_key, fred_key, analysis_period_years=years, forecast_days=days)
    except Exception as e:
        st.error(f"분석 함수(stock_analysis.py) 실행 중 오류 발생: {e}")
        st.error("자세한 내용은 터미널 로그를 확인하세요.")
        return None

if analyze_button:
    if not ticker_input:
        results_placeholder.warning("종목 티커를 입력해주세요.")
    elif not api_keys_loaded: # API 키 로드 실패 시 분석 중단
        results_placeholder.error("API 키가 로드되지 않아 분석을 실행할 수 없습니다.")
    else:
        ticker_processed = ticker_input.strip().upper() # 공백 제거 및 대문자 변환
        with st.spinner(f"{ticker_processed} 종목 분석 중... 잠시만 기다려주세요..."):
            try:
                results = run_cached_analysis(ticker_processed, NEWS_API_KEY, FRED_API_KEY, analysis_years, forecast_days)

                results_placeholder.empty() # 이전 결과 지우기

                if results and isinstance(results, dict):
                    st.header(f"📈 {ticker_processed} 분석 결과")

                    # 1. 기본 정보
                    st.subheader("요약 정보")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("현재가 (최근 종가)", f"${results.get('current_price', 'N/A')}")
                    col2.metric("분석 시작일", results.get('analysis_period_start', 'N/A'))
                    col3.metric("분석 종료일", results.get('analysis_period_end', 'N/A'))

                    # 2. 기술적 분석 (차트)
                    st.subheader("기술적 분석 차트")
                    stock_chart_fig = results.get('stock_chart_fig')
                    if stock_chart_fig:
                        st.plotly_chart(stock_chart_fig, use_container_width=True)
                    else:
                        st.warning("주가 차트를 생성하지 못했습니다. 티커가 유효한지 확인하세요.")

                    # 3. 시장 심리
                    st.subheader("시장 심리 분석")
                    col_news, col_fng = st.columns([2, 1])
                    with col_news:
                        st.markdown("**📰 뉴스 감정 분석**")
                        # ... (뉴스 표시 로직 동일) ...
                        news_sentiment = results.get('news_sentiment', ["정보 없음"])
                        if isinstance(news_sentiment, list) and len(news_sentiment) > 0:
                            avg_sentiment_line = news_sentiment[0]
                            st.info(avg_sentiment_line)
                            with st.expander("최근 뉴스 목록 보기", expanded=False):
                                for line in news_sentiment[1:]:
                                    st.write(f"- {line}")
                        else:
                            st.write(news_sentiment)
                    with col_fng:
                        st.markdown("**😨 공포-탐욕 지수**")
                        # ... (F&G 표시 로직 동일) ...
                        fng_index = results.get('fear_greed_index', "N/A")
                        if isinstance(fng_index, dict):
                            st.metric(label="현재 지수", value=fng_index.get('value', 'N/A'), delta=fng_index.get('classification', ''))
                        else:
                            st.write(fng_index)

                    # 4. Prophet 예측 분석
                    st.subheader("Prophet 주가 예측")
                    forecast_fig = results.get('forecast_fig')
                    forecast_data_list = results.get('prophet_forecast') # 리스트 형태의 예측 결과

                    if forecast_fig:
                        st.plotly_chart(forecast_fig, use_container_width=True)
                    elif isinstance(forecast_data_list, str): # 예측 실패 또는 데이터 부족 메시지
                        st.info(forecast_data_list)
                    else:
                         st.warning("예측 차트를 생성하지 못했습니다.")

                    # 예측 데이터 테이블 (항상 표시 시도, 데이터 없으면 비어있음)
                    if isinstance(forecast_data_list, list):
                        st.markdown("**📊 예측 데이터 (최근 10일)**")
                        df_forecast_display = pd.DataFrame(forecast_data_list)
                        # 날짜 형식 변경 및 필요한 열 선택
                        df_forecast_display['ds'] = pd.to_datetime(df_forecast_display['ds']).dt.strftime('%Y-%m-%d')
                        st.dataframe(df_forecast_display[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(10).style.format({
                            "yhat": "{:.2f}", "yhat_lower": "{:.2f}", "yhat_upper": "{:.2f}"
                        }), use_container_width=True)


                    # 교차 검증 결과
                    cv_plot_path = results.get('cv_plot_path')
                    if cv_plot_path and os.path.exists(cv_plot_path):
                         st.markdown("**📉 교차 검증 결과 (MAPE)**")
                         try:
                             st.image(cv_plot_path, caption="MAPE: 평균 절대 백분율 오차 (낮을수록 좋음)")
                         except Exception as img_err:
                              st.warning(f"교차 검증 이미지를 표시하는 중 오류 발생: {img_err}")
                    # ---------------- 리스크 트래커 기능 추가 시작 ------------------
                    st.divider()
                    st.subheader("🚨 리스크 트래커 (예측 기반)")

                    # 평단가가 입력되었고, 예측 데이터가 정상적으로 있을 때만 실행
                    if avg_price > 0 and isinstance(forecast_data_list, list):
                        df_pred = pd.DataFrame(forecast_data_list)
                        # 데이터 타입 변환 (오류 방지)
                        df_pred['ds'] = pd.to_datetime(df_pred['ds'])
                        df_pred['yhat_lower'] = pd.to_numeric(df_pred['yhat_lower'], errors='coerce')
                        df_pred['yhat'] = pd.to_numeric(df_pred['yhat'], errors='coerce')
                        df_pred['yhat_upper'] = pd.to_numeric(df_pred['yhat_upper'], errors='coerce')
                        df_pred.dropna(subset=['yhat_lower', 'yhat'], inplace=True) # 계산 불가 행 제거

                        # 리스크 계산
                        df_pred['평단가'] = avg_price
                        df_pred['리스크 여부'] = df_pred['yhat_lower'] < df_pred['평단가']
                        df_pred['예상 손실률'] = ((df_pred['yhat_lower'] - df_pred['평단가']) / df_pred['평단가']) * 100
                        df_pred.loc[df_pred['리스크 여부'] == False, '예상 손실률'] = 0 # 리스크 없으면 손실률 0

                        if quantity > 0:
                            df_pred['보유 수량'] = quantity
                            df_pred['예상 손실액'] = (df_pred['yhat_lower'] - df_pred['평단가']) * df_pred['보유 수량']
                            df_pred.loc[df_pred['리스크 여부'] == False, '예상 손실액'] = 0 # 리스크 없으면 손실액 0
                        else:
                            df_pred['예상 손실액'] = 0

                        # 리스크 요약 정보 계산
                        risk_days = df_pred['리스크 여부'].sum()
                        max_loss_pct = df_pred['예상 손실률'].min() if risk_days > 0 else 0 # 음수이므로 min 사용
                        max_loss_amt = df_pred['예상 손실액'].min() if risk_days > 0 and quantity > 0 else 0

                        # 리스크 요약 출력 (옵션 C + B 일부)
                        st.markdown("##### 리스크 요약")
                        col_r1, col_r2, col_r3 = st.columns(3)
                        col_r1.metric("⚠️ 예측 하한가 < 평단가 일수", f"{risk_days} 일 / {forecast_days} 일")
                        col_r2.metric("📉 최대 예상 손실률", f"{max_loss_pct:.2f}%")
                        if quantity > 0:
                             col_r3.metric("💸 최대 예상 손실액", f"${max_loss_amt:,.2f}")

                        if risk_days > 0:
                            st.warning(f"향후 {forecast_days}일 중 **{risk_days}일** 동안 예측 하한선이 현재 평단가(${avg_price:.2f})보다 낮아질 가능성이 있습니다. 최대 **{max_loss_pct:.2f}%** 의 예상 손실률을 보입니다.")
                        else:
                            st.success(f"향후 {forecast_days}일 동안 예측 하한선이 현재 평단가(${avg_price:.2f})보다 낮아질 가능성은 없는 것으로 예측됩니다.")

                        # 리스크 시각화 (옵션 B)
                        st.markdown("##### 평단가 vs 예측 하한선 비교 차트")
                        fig_risk = go.Figure()
                        # 예측 구간
                        fig_risk.add_trace(go.Scatter(
                            x=df_pred['ds'], y=df_pred['yhat_upper'], mode='lines', line_color='rgba(0,100,80,0.2)', name='Upper Bound'))
                        fig_risk.add_trace(go.Scatter(
                            x=df_pred['ds'], y=df_pred['yhat_lower'], mode='lines', line_color='rgba(0,100,80,0.2)', name='Lower Bound',
                            fill='tonexty', fillcolor='rgba(0,100,80,0.1)')) # 구간 채우기
                        # 예측 중앙값
                        fig_risk.add_trace(go.Scatter(
                            x=df_pred['ds'], y=df_pred['yhat'], mode='lines', line=dict(dash='dash', color='rgba(0,100,80,0.6)'), name='Forecast (yhat)'))
                        # 평단가 선
                        fig_risk.add_hline(y=avg_price, line_dash="dot", line_color="red", annotation_text=f"평단가: ${avg_price:.2f}", annotation_position="bottom right")
                        # 리스크 구간 강조 (선택 사항)
                        df_risk_periods = df_pred[df_pred['리스크 여부']]
                        fig_risk.add_trace(go.Scatter(x=df_risk_periods['ds'], y=df_risk_periods['yhat_lower'], mode='markers', marker_symbol='x', marker_color='red', name='Risk Day (Lower < Avg Price)'))

                        fig_risk.update_layout(title=f"{ticker_processed} 예측 하한선 vs 평단가 비교",
                                               xaxis_title="날짜", yaxis_title="가격", hovermode="x unified", margin=dict(l=20, r=20, t=40, b=20))
                        st.plotly_chart(fig_risk, use_container_width=True)

                        # 리스크 발생일 테이블 (옵션 B)
                        if risk_days > 0:
                             with st.expander(f"리스크 예측일 상세 보기 ({risk_days}일)"):
                                 df_risk_days_display = df_pred[df_pred['리스크 여부']].copy()
                                 df_risk_days_display['ds'] = df_risk_days_display['ds'].dt.strftime('%Y-%m-%d')
                                 cols_to_show = ['ds', 'yhat_lower', '평단가', '예상 손실률']
                                 if quantity > 0:
                                     cols_to_show.append('예상 손실액')
                                 st.dataframe(df_risk_days_display[cols_to_show].style.format({
                                     "yhat_lower": "{:.2f}", "평단가": "{:.2f}",
                                     "예상 손실률": "{:.2f}%", "예상 손실액": "${:,.2f}"
                                 }), use_container_width=True)

                    elif avg_price <= 0:
                        st.info("사이드바에서 '평단가'를 0보다 큰 값으로 입력하시면 예측 기반 리스크 분석을 볼 수 있습니다.")
                    else:
                         # 예측 데이터가 리스트가 아니거나 없는 경우
                         st.warning("Prophet 예측 데이터가 유효하지 않아 리스크 분석을 수행할 수 없습니다.")

                    # ----------------- 리스크 트래커 기능 추가 끝 ------------------


                    # 5. 자동 분석 결과 '관찰' 요약 (기존 로직)
                    st.divider()
                    st.subheader("🧐 자동 분석 결과 요약 (참고용)")
                    # ... (기존 요약 로직 - 필요시 리스크 트래커 결과도 반영하도록 수정 가능) ...
                    summary = []
                    # 예측 차트 관찰
                    if isinstance(forecast_data_list, list) and len(forecast_data_list) > 0:
                        try:
                            start_pred = forecast_data_list[0]['yhat']
                            end_pred = forecast_data_list[-1]['yhat']
                            trend_obs = "상승" if end_pred > start_pred else "하락" if end_pred < start_pred else "횡보"
                            summary.append(f"- Prophet 예측 모델은 향후 {forecast_days}일간 전반적으로 완만한 **{trend_obs}** 추세를 보입니다 (과거 데이터 기반).")
                            summary.append(f"- 예측 마지막 날({forecast_data_list[-1]['ds']}) 예상 가격 범위는 약 ${forecast_data_list[-1]['yhat_lower']:.2f} ~ ${forecast_data_list[-1]['yhat_upper']:.2f} 입니다.")
                        except Exception as summary_err:
                             logging.warning(f"예측 요약 생성 중 오류: {summary_err}")
                             summary.append("- 예측 데이터 요약 중 오류 발생.")

                    # 뉴스 감정 관찰
                    if isinstance(news_sentiment, list) and len(news_sentiment) > 0:
                         try:
                            # 첫 줄에서 평균 점수 파싱 개선
                            sentiment_line = news_sentiment[0]
                            score_part = sentiment_line.split(":")[-1].strip()
                            avg_score = float(score_part)
                            sentiment_desc = "긍정적" if avg_score > 0.05 else "부정적" if avg_score < -0.05 else "중립적"
                            summary.append(f"- 최근 뉴스들의 평균 감성 점수는 {avg_score:.2f}로, 전반적으로 **{sentiment_desc}**인 분위기입니다.")
                         except Exception as summary_err:
                            logging.warning(f"뉴스 요약 생성 중 오류: {summary_err}")
                            summary.append("- 뉴스 감정 요약 중 오류 발생.")

                    # 공포-탐욕 지수 관찰
                    if isinstance(fng_index, dict):
                         summary.append(f"- 현재 공포-탐욕 지수는 {fng_index.get('value', 'N/A')}으로, 시장 참여자들은 **'{fng_index.get('classification', 'N/A')}'** 상태의 심리를 보이고 있습니다.")

                     # 리스크 트래커 요약 추가
                    if avg_price > 0 and isinstance(forecast_data_list, list) and risk_days is not None:
                        if risk_days > 0:
                            summary.append(f"- 예측 하한선 기준, 향후 {forecast_days}일 중 **{risk_days}일** 동안 평단가 하회 리스크가 예측됩니다 (최대 예상 손실률 {max_loss_pct:.2f}%).")
                        else:
                            summary.append(f"- 예측 하한선 기준, 향후 {forecast_days}일 동안 평단가 하회 리스크는 예측되지 않았습니다.")


                    if summary:
                        st.markdown("\n".join(summary))
                        st.caption("⚠️ **주의:** 위 요약은 자동 생성된 참고 정보이며, 투자 조언이 아닙니다. 모든 투자 결정은 본인의 판단과 책임 하에 이루어져야 합니다.") # 면책 조항 강조
                    else:
                        st.write("분석 요약을 생성할 수 없습니다.")

                elif results is None:
                    # run_cached_analysis 자체가 None을 반환한 경우 (API 키 오류 등)
                    results_placeholder.error("분석 실행 중 오류가 발생했거나 필수 정보(API 키 등)가 부족합니다.")
                else:
                    # results가 dict가 아닌 다른 타입인 경우 (예상치 못한 오류)
                    results_placeholder.error("분석 결과를 처리하는 데 실패했습니다. 입력값을 확인하거나 잠시 후 다시 시도해주세요.")

            except Exception as e:
                results_placeholder.error(f"분석 중 예기치 않은 오류가 발생했습니다: {e}")
                st.error("자세한 오류 내용은 아래와 같습니다.")
                st.exception(e) # Streamlit에서 오류 상세 정보 표시

else:
    # 앱 처음 실행 시 안내 메시지
    results_placeholder.info("⬅️ 왼쪽 사이드바에서 분석할 종목 티커와 기간, 그리고 선택적으로 보유 정보를 입력한 후 '분석 시작' 버튼을 클릭하세요.")