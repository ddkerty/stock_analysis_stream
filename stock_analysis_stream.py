# stock_analysis.py

import os
import logging
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from textblob import TextBlob
import requests
from prophet import Prophet
from prophet.plot import plot_plotly
# import plotly.offline as py # HTML 저장 안 하므로 주석 처리 또는 삭제 가능
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from dotenv import load_dotenv
from fredapi import Fred
import traceback # 상세 오류 로깅 위해 추가
from prophet.diagnostics import cross_validation, performance_metrics # Prophet 진단 도구
from prophet.plot import plot_cross_validation_metric # 교차 검증 결과 시각화 도구
import matplotlib.pyplot as plt # 시각화 결과를 파일로 저장하기 위해 필요
import warnings

# 경고 메시지 숨기기 (선택 사항)
# warnings.filterwarnings('ignore')
# FutureWarnings 숨기기 (Plotly 관련)
warnings.simplefilter(action='ignore', category=FutureWarning)


# --- 기본 설정 및 절대 경로 정의 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 스크립트 파일 기준 절대 경로 설정
try:
    # 스크립트로 실행될 때
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # 대화형 환경(예: Jupyter Notebook) 등 __file__ 변수가 없는 경우 현재 작업 디렉토리 사용
    BASE_DIR = os.getcwd()
    logging.info(f"__file__ 변수 없음. 현재 작업 디렉토리 사용: {BASE_DIR}")

CHARTS_FOLDER = os.path.join(BASE_DIR, "charts")
DATA_FOLDER = os.path.join(BASE_DIR, "data")
FORECAST_FOLDER = os.path.join(BASE_DIR, "forecast")

# .env 파일 로드 시도 (로컬 실행용)
try:
    dotenv_path = os.path.join(BASE_DIR, '.env')
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path=dotenv_path)
        logging.info(f".env 파일 로드 성공: {dotenv_path}")
    else:
        logging.warning(f".env 파일을 찾을 수 없습니다: {dotenv_path}")
except Exception as e:
    logging.error(f".env 파일 로드 중 오류: {e}")

# --- 데이터 가져오기 함수 ---

# 1. 공포-탐욕 지수 가져오기
def get_fear_greed_index():
    """공포-탐욕 지수를 API에서 가져옵니다."""
    url = "https://api.alternative.me/fng/?limit=1&format=json&date_format=world"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()['data'][0]
        value = int(data['value'])
        classification = data['value_classification']
        logging.info(f"공포-탐욕 지수 가져오기 성공: {value} ({classification})")
        return value, classification
    except requests.exceptions.RequestException as e:
        logging.error(f"공포-탐욕 지수 API 요청 실패: {e}")
        return None, None
    except (KeyError, IndexError, ValueError) as e:
        logging.error(f"공포-탐욕 지수 데이터 처리 실패: {e}")
        return None, None
    except Exception as e:
        logging.error(f"알 수 없는 오류 발생 (공포-탐욕 지수): {e}")
        return None, None

# 2. 주가 데이터 수집
def get_stock_data(ticker, start_date=None, end_date=None, period="1y"):
    """지정된 종목의 주가 데이터를 yfinance로 가져옵니다."""
    try:
        stock = yf.Ticker(ticker)
        # 특정 티커 정보 가져오기 시도 (존재 여부 간접 확인)
        info = stock.info
        if not info or info.get('regularMarketPrice') is None:
             # 정보가 없거나 시장 가격이 없으면 유효하지 않은 티커로 간주 (상장폐지 등)
             logging.error(f"티커 '{ticker}'에 대한 유효한 정보를 찾을 수 없습니다. 상장 폐지되었거나 잘못된 티커일 수 있습니다.")
             return None

        if start_date and end_date:
            data = stock.history(start=start_date, end=end_date)
            logging.info(f"{ticker} 주가 데이터 가져오기 성공 ({start_date} ~ {end_date})")
        else:
            data = stock.history(period=period)
            logging.info(f"{ticker} 주가 데이터 가져오기 성공 (기간: {period})")

        if data.empty:
            logging.warning(f"{ticker} 주가 데이터가 비어있습니다.")
            return None
        if isinstance(data.index, pd.DatetimeIndex):
             data.index = data.index.tz_localize(None)
        return data
    except Exception as e:
        # yfinance에서 발생하는 다양한 오류 처리 (네트워크, 잘못된 티커 등)
        logging.error(f"티커 '{ticker}' 주가 데이터 가져오기 실패: {e}")
        return None

# 3. 매크로 경제 지표 수집 - fred_key 인자 추가
def get_macro_data(start_date, end_date=None, fred_key=None):
    """VIX, US10Y, 13주 국채(IRX), DXY, 연방기금 금리 데이터를 가져옵니다."""
    if end_date is None:
        end_date = datetime.today().strftime("%Y-%m-%d")

    tickers = {"^VIX": "VIX", "^TNX": "US10Y", "^IRX": "US13W", "DX-Y.NYB": "DXY"}
    df_macro = pd.DataFrame()

    all_yf_data = []
    for tk, label in tickers.items():
        try:
            tmp = yf.download(tk, start=start_date, end=end_date, progress=False)
            if not tmp.empty:
                tmp = tmp[['Close']].rename(columns={"Close": label})
                if isinstance(tmp.index, pd.DatetimeIndex):
                     tmp.index = tmp.index.tz_localize(None)
                all_yf_data.append(tmp)
                logging.info(f"{label} 데이터 가져오기 성공")
            else:
                logging.warning(f"{label} 데이터가 비어있습니다.")
        except Exception as e:
            logging.error(f"{label} 데이터 가져오기 실패: {e}")

    if all_yf_data:
        df_macro = pd.concat(all_yf_data, axis=1)
        if isinstance(df_macro.columns, pd.MultiIndex):
            df_macro.columns = df_macro.columns.get_level_values(-1)

    # FRED API 키 사용 부분 수정
    if fred_key:
        try:
            fred = Fred(api_key=fred_key)
            fedfunds = fred.get_series("FEDFUNDS", start_date=start_date, end_date=end_date)
            fedfunds = fedfunds.rename("FedFunds")
            if isinstance(fedfunds.index, pd.DatetimeIndex):
                 fedfunds.index = fedfunds.index.tz_localize(None)

            if not df_macro.empty:
                 df_macro = df_macro.merge(fedfunds, left_index=True, right_index=True, how='outer')
                 logging.info("연방기금 금리 데이터 병합 성공")
            else:
                 df_macro = pd.DataFrame(fedfunds)
                 logging.info("연방기금 금리 데이터 가져오기 성공 (df_macro 새로 생성)")
        except Exception as e:
            logging.error(f"연방기금 금리 데이터 가져오기/병합 실패: {e}")
            if 'FedFunds' not in df_macro.columns and not df_macro.empty:
                 df_macro['FedFunds'] = pd.NA
    else:
        logging.warning("FRED_API_KEY가 제공되지 않아 연방기금 금리 데이터를 가져오지 못했습니다.")
        if not df_macro.empty and 'FedFunds' not in df_macro.columns:
             df_macro['FedFunds'] = pd.NA

    if not df_macro.empty:
        df_macro.index = pd.to_datetime(df_macro.index).tz_localize(None)
        df_macro = df_macro.sort_index()
        df_macro = df_macro.ffill().bfill()
        df_macro = df_macro.reset_index().rename(columns={'index': 'Date'})
        df_macro["Date"] = pd.to_datetime(df_macro["Date"])
        logging.info("매크로 데이터 처리 완료.")
        return df_macro
    else:
        logging.warning("매크로 데이터를 가져오지 못했습니다.")
        return pd.DataFrame()

# --- 분석 및 시각화 함수 ---

# 4. 기술 차트 생성 - Figure 객체 반환
def plot_stock_chart(ticker, start_date=None, end_date=None, period="1y"):
    """주가 데이터를 기반으로 차트 Figure 객체를 생성하여 반환합니다."""
    df = get_stock_data(ticker, start_date=start_date, end_date=end_date, period=period)
    if df is None or df.empty:
        logging.error(f"{ticker} 차트 생성 실패: 주가 데이터 없음")
        return None
    try:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            vertical_spacing=0.03, row_heights=[0.7, 0.3])
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'],
                                     low=df['Low'], close=df['Close'], name='가격'), row=1, col=1)
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='거래량', marker_color='rgba(0,0,100,0.6)'), row=2, col=1)
        fig.update_layout(title=f'{ticker} 주가 및 거래량 차트', yaxis_title='가격', yaxis2_title='거래량',
                          xaxis_rangeslider_visible=False, hovermode='x unified', margin=dict(l=20, r=20, t=40, b=20))
        fig.update_yaxes(title_text="가격", row=1, col=1)
        fig.update_yaxes(title_text="거래량", row=2, col=1)
        logging.info(f"{ticker} 차트 생성 완료 (Figure 객체 반환)")
        return fig
    except Exception as e:
        logging.error(f"{ticker} 차트 Figure 생성 중 오류: {e}")
        return None

# 5. 뉴스 감정 분석
def get_news_sentiment(ticker, api_key):
    """지정된 종목에 대한 뉴스 기사의 감정을 분석합니다."""
    if not api_key:
        logging.warning("NEWS_API_KEY가 제공되지 않아 뉴스 감정 분석을 건너<0xEB>니다.")
        return ["뉴스 API 키가 설정되지 않았습니다."]
    url = f"https://newsapi.org/v2/everything?q={ticker}&pageSize=20&language=en&sortBy=publishedAt&apiKey={api_key}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        articles = response.json().get('articles', [])
        if not articles:
            logging.info(f"{ticker}: 관련된 뉴스를 찾지 못했습니다.")
            return ["관련 뉴스를 찾지 못했습니다."]
        output_lines = []
        total_polarity = 0
        analyzed_count = 0
        for i, article in enumerate(articles, 1):
            title = article.get('title', 'N/A')
            description = article.get('description')
            content = article.get('content')
            text_to_analyze = description or content or title or ""
            if text_to_analyze and text_to_analyze != "[Removed]":
                try:
                    blob = TextBlob(text_to_analyze)
                    polarity = blob.sentiment.polarity
                    output_lines.append(f"{i}. {title} | 감정: {polarity:.2f}")
                    total_polarity += polarity
                    analyzed_count += 1
                except Exception as text_err:
                     logging.warning(f"뉴스 텍스트 처리 중 오류 ({title}): {text_err}")
                     output_lines.append(f"{i}. {title} | 감정 분석 오류")
            else:
                output_lines.append(f"{i}. {title} | 분석할 내용 없음")
        avg_polarity = total_polarity / analyzed_count if analyzed_count > 0 else 0
        logging.info(f"{ticker} 뉴스 감정 분석 완료 (평균 감성: {avg_polarity:.2f})")
        output_lines.insert(0, f"총 {analyzed_count}개 뉴스 분석 | 평균 감성 점수: {avg_polarity:.2f}")
        return output_lines
    except requests.exceptions.RequestException as e:
        logging.error(f"뉴스 API 요청 실패: {e}")
        return [f"뉴스 데이터를 가져오지 못했습니다 (요청 오류: {e})."]
    except Exception as e:
        logging.error(f"뉴스 감정 분석 중 알 수 없는 오류 발생: {e}")
        return ["뉴스 감정 분석 중 오류가 발생했습니다."]

# 6. Prophet 예측 - Figure 및 경로 반환, 절대 경로 사용
def run_prophet_forecast(ticker, start_date, end_date=None, forecast_days=30, fred_key=None): # fred_key 인자 추가
    """Prophet 예측, 교차 검증 수행 후 예측 결과(dict), 예측 차트(fig), CV차트 경로(str) 반환"""
    if end_date is None:
        end_date = datetime.today().strftime("%Y-%m-%d")

    # --- 데이터 준비 ---
    df_stock = get_stock_data(ticker, start_date=start_date, end_date=end_date)
    if df_stock is None or df_stock.empty:
        logging.error(f"Prophet 예측 실패: {ticker} 주가 데이터 없음")
        return None, None, None

    df_stock = df_stock.reset_index()[["Date", "Close"]]
    df_stock["Date"] = pd.to_datetime(df_stock["Date"])

    # get_macro_data 호출 시 fred_key 전달
    df_macro = get_macro_data(start_date=start_date, end_date=end_date, fred_key=fred_key)

    if not df_macro.empty:
        # Date 컬럼 타입을 datetime64[ns]로 확실히 변환 (merge 오류 방지)
        df_stock['Date'] = pd.to_datetime(df_stock['Date'])
        df_macro['Date'] = pd.to_datetime(df_macro['Date'])
        df_merged = pd.merge(df_stock, df_macro, on="Date", how="left")
        logging.info(f"주가 데이터와 매크로 데이터 병합 완료.")
        macro_cols = ["VIX", "US10Y", "US13W", "DXY", "FedFunds"]
        for col in macro_cols:
            if col in df_merged.columns:
                if not pd.api.types.is_numeric_dtype(df_merged[col]):
                    df_merged[col] = pd.to_numeric(df_merged[col], errors='coerce')
                df_merged[col] = df_merged[col].ffill().bfill()
    else:
        logging.warning("매크로 데이터가 없어 주가 데이터만으로 Prophet 예측을 진행합니다.")
        df_merged = df_stock

    if df_merged['Close'].isnull().any():
        df_merged = df_merged.dropna(subset=['Close'])

    if df_merged.empty or len(df_merged) < 30:
        logging.error(f"Prophet 예측 실패: 사용할 최종 데이터가 부족합니다 ({len(df_merged)} 행).")
        return None, None, None

    logging.info(f"Prophet 학습용 데이터 준비 완료: {len(df_merged)} 행")
    os.makedirs(DATA_FOLDER, exist_ok=True)
    data_csv_path = os.path.join(DATA_FOLDER, f"{ticker}_merged_for_prophet.csv")
    try:
        df_merged.to_csv(data_csv_path, index=False)
        logging.info(f"학습용 데이터 저장 완료: {data_csv_path}")
    except Exception as save_err:
        logging.error(f"학습용 데이터 저장 실패: {save_err}")

    # --- Prophet 모델 설정 및 학습 ---
    df_prophet = df_merged.rename(columns={"Date": "ds", "Close": "y"})
    df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])

    m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False,
                changepoint_prior_scale=0.05)

    regressors = []
    if not df_macro.empty:
        for col in ["VIX", "US10Y", "US13W", "DXY", "FedFunds"]:
             if col in df_prophet.columns:
                if pd.api.types.is_numeric_dtype(df_prophet[col]):
                    if df_prophet[col].isnull().sum() == 0:
                        m.add_regressor(col)
                        regressors.append(col)
                        logging.info(f"Prophet Regressor 추가: {col}")
                    else:
                        logging.warning(f"Regressor '{col}'에 결측치가 있어 추가하지 않습니다.")
                else:
                    logging.warning(f"Regressor '{col}'가 숫자 타입이 아니라 추가하지 않습니다. (Type: {df_prophet[col].dtype})")

    forecast_result_dict = None
    fig_forecast = None
    cv_plot_path = None

    try:
        m.fit(df_prophet[['ds', 'y'] + regressors])
        logging.info("Prophet 모델 학습 완료.")

        os.makedirs(FORECAST_FOLDER, exist_ok=True) # 폴더 미리 생성

        # --- 교차 검증 ---
        try:
            initial_days = '365 days'
            period_days = '90 days'
            horizon_days = f'{forecast_days} days'
            logging.info(f"Prophet 교차 검증 시작 (initial={initial_days}, period={period_days}, horizon={horizon_days})...")
            df_cv = cross_validation(m, initial=initial_days, period=period_days, horizon=horizon_days, parallel=None)
            logging.info("Prophet 교차 검증 완료.")
            df_p = performance_metrics(df_cv)
            logging.info(f"Prophet 성능 지표 (상위 5개):\n{df_p.head().to_string()}")
            fig_cv = plot_cross_validation_metric(df_cv, metric='mape')
            plt.title(f'{ticker} Cross Validation MAPE (Lower is Better)')
            cv_plot_path = os.path.join(FORECAST_FOLDER, f"{ticker}_cv_mape_plot.png")
            logging.debug(f"교차 검증 그래프 저장 경로: {cv_plot_path}")
            fig_cv.savefig(cv_plot_path)
            plt.close(fig_cv)
            logging.info(f"교차 검증 MAPE 차트 저장 완료: {cv_plot_path}")
        except Exception as cv_err:
            logging.error(f"Prophet 교차 검증/성능 평가 중 오류 발생: {cv_err}")
            cv_plot_path = None

        # --- 미래 예측 ---
        logging.info("미래 예측 시작...")
        future = m.make_future_dataframe(periods=forecast_days)
        if regressors:
            # merge를 위해 원본 df_merged 사용 (Date 컬럼 존재 가정)
            # Date 컬럼 타입을 datetime64[ns]로 변환
            temp_merged = df_merged.copy()
            temp_merged['Date'] = pd.to_datetime(temp_merged['Date'])

            future = future.merge(temp_merged[['Date'] + regressors],
                                  left_on='ds', right_on='Date', how='left')
            future = future.drop(columns=['Date']) # 병합 후 Date 컬럼 제거

            for col in regressors:
                if col in temp_merged.columns:
                    # 마지막 NaN 아닌 값 찾기 개선
                    non_na_series = temp_merged[col].dropna()
                    last_val = non_na_series.iloc[-1] if not non_na_series.empty else 0
                    if non_na_series.empty:
                        logging.warning(f"Regressor '{col}'의 모든 과거 값이 NaN입니다. 미래 값을 0으로 채웁니다.")

                    # 미래 값 채우기 (ffill 후 마지막 값으로 nan 채우기)
                    future[col] = future[col].ffill()
                    future[col].fillna(last_val, inplace=True)

        forecast = m.predict(future)
        logging.info("미래 예측 완료.")

        # --- 결과 저장 및 Figure 생성 ---
        csv_fn = os.path.join(FORECAST_FOLDER, f"{ticker}_forecast_data.csv")
        forecast_to_save = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
        forecast_to_save['ds'] = forecast_to_save['ds'].dt.strftime('%Y-%m-%d')
        forecast_to_save.to_csv(csv_fn, index=False)
        logging.info(f"예측 결과 데이터 저장 완료: {csv_fn}")

        fig_forecast = plot_plotly(m, forecast)
        fig_forecast.update_layout(title=f'{ticker} Price Forecast (Next {forecast_days} Days)', margin=dict(l=20, r=20, t=40, b=20))
        logging.info(f"예측 결과 Figure 객체 생성 완료.")

        forecast_result_dict = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(forecast_days).to_dict('records')
        for record in forecast_result_dict:
            record['ds'] = record['ds'].strftime('%Y-%m-%d')

        return forecast_result_dict, fig_forecast, cv_plot_path

    except Exception as e:
        logging.error(f"Prophet 모델 학습 또는 예측 중 오류 발생: {e}")
        logging.error(traceback.format_exc())
        return None, None, None

# --- 메인 분석 함수 ---

# 7. 통합 분석 - fred_key 인자 추가 및 반환값 처리
def analyze_stock(ticker, news_key, fred_key, analysis_period_years=2, forecast_days=30):
    """모든 데이터를 종합하여 주식 분석 결과를 반환합니다."""
    logging.info(f"--- {ticker} 주식 분석 시작 ---")
    output_results = {}

    # 날짜 설정 시 시간 정보 제거
    try:
        end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        start_date = end_date - relativedelta(years=analysis_period_years)
        start_date_str = start_date.strftime("%Y-%m-%d")
        end_date_str = end_date.strftime("%Y-%m-%d") # 현재 날짜까지 포함하도록 조정
        logging.info(f"분석 기간: {start_date_str} ~ {end_date_str}")
    except Exception as date_err:
        logging.error(f"날짜 설정 중 오류: {date_err}")
        return None # 날짜 설정 실패 시 분석 중단

    # 기본 정보 및 데이터 포인트
    df_stock_full = get_stock_data(ticker, start_date=start_date_str, end_date=end_date_str)
    if df_stock_full is not None and not df_stock_full.empty:
        output_results['current_price'] = f"{df_stock_full['Close'].iloc[-1]:.2f}" if not df_stock_full['Close'].empty else "N/A"
        output_results['analysis_period_start'] = start_date_str
        output_results['analysis_period_end'] = end_date_str # end_date를 오늘 날짜 문자열로
        output_results['data_points'] = len(df_stock_full)
    else:
        # get_stock_data에서 이미 오류/경고 로깅됨
        output_results['current_price'] = "N/A"
        output_results['analysis_period_start'] = start_date_str
        output_results['analysis_period_end'] = end_date_str
        output_results['data_points'] = 0
        logging.warning(f"{ticker}의 주가 정보를 가져올 수 없어 일부 분석이 제한됩니다.")

    # 기술 차트 Figure 객체
    # 주의: df_stock_full이 None 이어도 plot_stock_chart는 내부에서 다시 데이터를 가져옴
    stock_chart_fig = plot_stock_chart(ticker, start_date=start_date_str, end_date=end_date_str)
    output_results['stock_chart_fig'] = stock_chart_fig

    # 뉴스 감정 분석
    output_results['news_sentiment'] = get_news_sentiment(ticker, news_key)

    # 공포-탐욕 지수
    fg_value, fg_class = get_fear_greed_index()
    output_results['fear_greed_index'] = {'value': fg_value, 'classification': fg_class} if fg_value is not None else "N/A"

    # Prophet 예측 (데이터 포인트가 충분할 때만 실행)
    if output_results['data_points'] > 30:
        forecast_dict, forecast_fig, cv_plot_path = run_prophet_forecast(
            ticker, start_date=start_date_str, end_date=end_date_str,
            forecast_days=forecast_days, fred_key=fred_key # fred_key 전달
        )
        output_results['prophet_forecast'] = forecast_dict if forecast_dict is not None else "예측 실패"
        output_results['forecast_fig'] = forecast_fig
        output_results['cv_plot_path'] = cv_plot_path
    else:
         output_results['prophet_forecast'] = f"데이터 부족 ({output_results['data_points']}개)으로 예측 불가"
         output_results['forecast_fig'] = None
         output_results['cv_plot_path'] = None
         logging.warning(f"데이터 부족 ({output_results['data_points']}개)으로 Prophet 예측을 건너<0xEB>니다.")

    logging.info(f"--- {ticker} 주식 분석 완료 ---")
    return output_results

# --- 메인 실행 부분 (직접 실행 시 테스트용) ---
if __name__ == "__main__":
    print(f"stock_analysis.py를 직접 실행합니다 (테스트 목적, Base directory: {BASE_DIR}).")
    target_ticker = "AAPL" # 테스트할 티커

    # 로컬 테스트 시 .env 파일 로드 확인
    news_api_key_local = os.getenv("NEWS_API_KEY")
    fred_api_key_local = os.getenv("FRED_API_KEY")

    if not news_api_key_local or not fred_api_key_local:
        print("경고: 로컬 테스트를 위한 API 키가 .env 파일에 없습니다.")
        test_results = None
    else:
        test_results = analyze_stock(ticker=target_ticker,
                                     news_key=news_api_key_local,
                                     fred_key=fred_api_key_local, # fred_key 전달
                                     analysis_period_years=1,
                                     forecast_days=15)

    print("\n--- 테스트 실행 결과 요약 ---")
    if test_results:
        for key, value in test_results.items():
             if 'fig' in key and value is not None:
                  print(f"- {key.replace('_', ' ').title()}: Plotly Figure 객체 생성됨")
             elif key == 'prophet_forecast':
                 print(f"- {key.replace('_', ' ').title()}: {type(value)}")
             elif key == 'news_sentiment':
                 print(f"- {key.replace('_', ' ').title()}: {len(value) if isinstance(value, list) else 0} 항목")
             else:
                  print(f"- {key.replace('_', ' ').title()}: {value}")
    else:
        print("테스트 분석 실패.")

    print("\n--- 테스트 실행 종료 ---")