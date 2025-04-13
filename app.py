# app.py

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import os
from dotenv import load_dotenv
import traceback # 상세 오류 로깅 위해 추가

# --- 기본 경로 설정 ---
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()

# --- 분석 로직 가져오기 ---
# stock_analysis.py 파일에서 필요한 함수들을 가져옵니다.
try:
    import stock_analysis as sa
except ModuleNotFoundError:
    st.error(f"오류: stock_analysis.py 파일을 찾을 수 없습니다. app.py와 같은 폴더({BASE_DIR})에 저장해주세요.")
    st.stop() # 앱 실행 중지
except ImportError as ie:
     st.error(f"stock_analysis.py를 import하는 중 오류 발생: {ie}")
     st.info("필요한 라이브러리가 모두 설치되었는지 확인하세요. (예: pip install -r requirements.txt)")
     st.stop()

# --- Streamlit 페이지 설정 ---
st.set_page_config(page_title="주식 분석 도구", layout="wide", initial_sidebar_state="expanded")

st.title("📊 주식 분석 및 예측 도구")
st.markdown("과거 데이터와 여러 지표를 바탕으로 주식 정보를 분석하고 미래 주가를 예측합니다.")

# --- API 키 로드 (Secrets 우선, .env 차선) ---
NEWS_API_KEY = None
FRED_API_KEY = None
api_keys_loaded = False
try:
    # Streamlit Cloud 환경에서는 Secrets 사용
    NEWS_API_KEY = st.secrets["NEWS_API_KEY"]
    FRED_API_KEY = st.secrets["FRED_API_KEY"]
    st.sidebar.success("API 키 로드 완료 (Secrets)") # 사이드바에 표시
    api_keys_loaded = True
except KeyError:
    st.sidebar.warning("Streamlit Secrets에 API 키가 없습니다.")
except Exception:
    # 로컬 환경 등 Secrets 접근 불가 시 .env 파일 시도
    st.sidebar.info("Secrets 로드 실패. 로컬 .env 파일을 확인합니다.")
    try:
        dotenv_path = os.path.join(BASE_DIR, '.env')
        if os.path.exists(dotenv_path):
            load_dotenv(dotenv_path=dotenv_path)
            NEWS_API_KEY = os.getenv("NEWS_API_KEY")
            FRED_API_KEY = os.getenv("FRED_API_KEY")
            if NEWS_API_KEY and FRED_API_KEY:
                st.sidebar.success("API 키 로드 완료 (.env)")
                api_keys_loaded = True
            else:
                st.sidebar.error("로컬 .env 파일에서 API 키를 찾을 수 없습니다.")
        else:
            st.sidebar.error("로컬 .env 파일을 찾을 수 없습니다.")
    except Exception as e:
        st.sidebar.error(f".env 파일 로드 중 오류: {e}")

if not api_keys_loaded:
    st.sidebar.error("API 키 로드에 실패하여 일부 기능이 제한될 수 있습니다.")


# --- 사이드바 설정 ---
with st.sidebar:
    st.header("⚙️ 분석 설정")

    ticker_input = st.text_input("종목 티커 입력 (예: AAPL, 005930.KS)", value="AAPL")
    analysis_years = st.select_slider("분석 기간 선택", options=[1, 2, 3, 5, 7, 10], value=2)
    st.caption(f"과거 {analysis_years}년 데이터를 분석합니다.")
    forecast_days = st.number_input("예측 기간 (일)", min_value=7, max_value=180, value=30, step=7)
    st.caption(f"향후 {forecast_days}일 후까지 예측합니다.")
    analyze_button = st.button("🚀 분석 시작!")

# --- 메인 화면 ---
results_placeholder = st.container() # 결과를 표시할 영역

# 캐시된 분석 함수 정의
@st.cache_data(ttl=timedelta(hours=1)) # 1시간 동안 캐시 유지
def run_cached_analysis(ticker, news_key, fred_key, years, days):
    """캐싱을 위한 분석 함수 래퍼"""
    # analyze_stock 호출 시 로드된 API 키 전달
    if not news_key or not fred_key:
         # API 키가 없으면 오류를 발생시키거나 None을 반환
         # raise ValueError("API 키가 유효하지 않습니다.") # 오류 발생 시키기
         st.error("API 키가 유효하지 않아 분석을 진행할 수 없습니다.") # UI에 메시지 표시
         return None # 분석 결과 없음을 반환
    try:
        return sa.analyze_stock(ticker, news_key, fred_key, analysis_period_years=years, forecast_days=days)
    except Exception as e:
        st.error(f"분석 함수(stock_analysis.py) 실행 중 오류 발생: {e}")
        st.error("자세한 내용은 터미널 로그를 확인하세요.")
        # traceback.print_exc() # 터미널에 상세 오류 출력 (선택 사항)
        return None # 오류 시 None 반환


if analyze_button:
    if not ticker_input:
        results_placeholder.warning("종목 티커를 입력해주세요.")
    elif not api_keys_loaded:
        results_placeholder.error("API 키가 로드되지 않아 분석을 실행할 수 없습니다. Secrets 또는 .env 파일을 확인하세요.")
    else:
        with st.spinner(f"{ticker_input} 종목 분석 중... 잠시만 기다려주세요... (시간이 걸릴 수 있습니다)"):
            try:
                # 캐시된 함수 호출
                results = run_cached_analysis(ticker_input.strip().upper(), NEWS_API_KEY, FRED_API_KEY, analysis_years, forecast_days)

                # --- 결과 표시 ---
                results_placeholder.empty() # 이전 결과 지우기 (분석 성공 시)

                if results and isinstance(results, dict): # 결과가 유효한 딕셔너리인지 확인
                    st.header(f"📈 {ticker_input.strip().upper()} 분석 결과")

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
                        news_sentiment = results.get('news_sentiment', ["정보 없음"])
                        if isinstance(news_sentiment, list) and len(news_sentiment) > 0:
                            avg_sentiment_line = news_sentiment[0]
                            st.info(avg_sentiment_line)
                            with st.expander("최근 뉴스 목록 보기", expanded=False):
                                for line in news_sentiment[1:]:
                                    st.write(f"- {line}") # 리스트 항목처럼 보이게 수정
                        else:
                            st.write(news_sentiment)

                    with col_fng:
                        st.markdown("**😨 공포-탐욕 지수**")
                        fng_index = results.get('fear_greed_index', "N/A")
                        if isinstance(fng_index, dict):
                            st.metric(label="현재 지수", value=fng_index.get('value', 'N/A'), delta=fng_index.get('classification', ''))
                        else:
                            st.write(fng_index)

                    # 4. Prophet 예측 분석
                    st.subheader("Prophet 주가 예측")
                    forecast_fig = results.get('forecast_fig')
                    if forecast_fig:
                        st.plotly_chart(forecast_fig, use_container_width=True)
                    else:
                        # 예측 실패 메시지가 'prophet_forecast' 키에 있을 수 있음
                        forecast_status = results.get('prophet_forecast', "예측 정보 없음")
                        if forecast_status != "예측 실패" and "데이터 부족" not in str(forecast_status):
                            st.warning("예측 차트를 생성하지 못했습니다.")
                        # 데이터 부족 또는 예측 실패 메시지는 아래 테이블 부분에서 표시

                    # 예측 데이터 테이블
                    forecast_data = results.get('prophet_forecast')
                    if isinstance(forecast_data, list):
                        st.markdown("**📊 예측 데이터 (최근 10일)**")
                        df_forecast = pd.DataFrame(forecast_data)
                        st.dataframe(df_forecast.tail(10).style.format({
                            "yhat": "{:.2f}",
                            "yhat_lower": "{:.2f}",
                            "yhat_upper": "{:.2f}"
                        }), use_container_width=True)
                    elif forecast_data:
                         st.info(str(forecast_data)) # 예측 실패 또는 데이터 부족 메시지 표시


                    # 교차 검증 결과 (이미지 파일 표시)
                    cv_plot_path = results.get('cv_plot_path')
                    if cv_plot_path and os.path.exists(cv_plot_path):
                         st.markdown("**📉 교차 검증 결과 (MAPE)**")
                         st.image(cv_plot_path, caption="MAPE: 평균 절대 백분율 오차 (낮을수록 좋음)")
                    # 교차 검증 실패 시 별도 메시지 없음 (로그 확인)

                    # 5. 자동 분석 결과 '관찰'
                    st.subheader("자동 분석 결과 요약 (참고용)")
                    summary = []
                    # 예측 차트 관찰
                    if isinstance(forecast_data, list) and len(forecast_data) > 0:
                        try:
                            start_pred = forecast_data[0]['yhat']
                            end_pred = forecast_data[-1]['yhat']
                            trend_obs = "상승" if end_pred > start_pred else "하락" if end_pred < start_pred else "횡보"
                            summary.append(f"- Prophet 예측 모델은 향후 {forecast_days}일간 전반적으로 완만한 **{trend_obs}** 추세를 보입니다 (과거 데이터 기반).")
                            summary.append(f"- 예측 마지막 날({forecast_data[-1]['ds']}) 예상 가격 범위는 약 ${forecast_data[-1]['yhat_lower']:.2f} ~ ${forecast_data[-1]['yhat_upper']:.2f} 입니다.")
                        except Exception as summary_err:
                             logging.warning(f"예측 요약 생성 중 오류: {summary_err}")
                             summary.append("- 예측 데이터 요약 중 오류 발생.")

                    # 뉴스 감정 관찰
                    if isinstance(news_sentiment, list) and len(news_sentiment) > 0:
                         try:
                            avg_score_str = news_sentiment[0].split(":")[-1].strip()
                            avg_score = float(avg_score_str)
                            sentiment_desc = "긍정적" if avg_score > 0.05 else "부정적" if avg_score < -0.05 else "중립적"
                            summary.append(f"- 최근 뉴스들의 평균 감성 점수는 {avg_score:.2f}로, 전반적으로 **{sentiment_desc}**인 분위기입니다.")
                         except Exception as summary_err:
                            logging.warning(f"뉴스 요약 생성 중 오류: {summary_err}")
                            summary.append("- 뉴스 감정 요약 중 오류 발생.")

                    # 공포-탐욕 지수 관찰
                    if isinstance(fng_index, dict):
                         summary.append(f"- 현재 공포-탐욕 지수는 {fng_index.get('value', 'N/A')}으로, 시장 참여자들은 **'{fng_index.get('classification', 'N/A')}'** 상태의 심리를 보이고 있습니다.")

                    if summary:
                        st.markdown("\n".join(summary))
                        st.caption("주의: 위 요약은 자동 생성된 참고 정보이며, 투자 조언이 아닙니다. 모든 투자 결정은 본인의 판단과 책임 하에 이루어져야 합니다.") # 면책 조항 추가
                    else:
                        st.write("분석 요약을 생성할 수 없습니다.")

                else:
                    # results가 None이거나 dict가 아닌 경우
                    results_placeholder.error("분석 결과를 처리하는 데 실패했습니다. 입력값을 확인하거나 잠시 후 다시 시도해주세요.")

            except Exception as e:
                results_placeholder.error(f"분석 중 예기치 않은 오류가 발생했습니다: {e}")
                st.error("자세한 오류 내용은 아래와 같습니다.")
                st.exception(e) # Streamlit에서 오류 상세 정보 표시

else:
    # 앱 처음 실행 시 안내 메시지
    results_placeholder.info("⬅️ 왼쪽 사이드바에서 분석할 종목 티커와 기간을 설정한 후 '분석 시작' 버튼을 클릭하세요.")