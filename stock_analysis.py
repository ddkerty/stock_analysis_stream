# stock_analysis.py (Finnhub API 적용 및 레이트 리미터 추가)

import os
import logging
import pandas as pd
import numpy as np
# import yfinance as yf # yfinance 의존성 제거 시도
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# from textblob import TextBlob # Finnhub 뉴스 감정 분석 사용 시 불필요
import requests # FRED API 등 일부 HTTP 요청에 필요
from prophet import Prophet
from prophet.plot import plot_plotly
from datetime import datetime, timedelta, time as dt_time # dt_time 추가
from dateutil.relativedelta import relativedelta
from dotenv import load_dotenv
from fredapi import Fred
import traceback
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_cross_validation_metric
import matplotlib.pyplot as plt
import warnings
# import locale # 현재 직접 사용되지 않음
import re
import pandas_ta as ta

# Finnhub 및 레이트 리미터 관련 import
import finnhub
from ratelimit import limits, RateLimitException
from backoff import on_exception, expo
import time as time_module # time import 충돌 방지

# 경고 메시지 및 로깅 설정
warnings.simplefilter(action='ignore', category=FutureWarning)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 경로 설정 ---
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()
    logging.info(f"__file__ 없음. CWD 사용: {BASE_DIR}")

CHARTS_FOLDER = os.path.join(BASE_DIR, "charts")
DATA_FOLDER = os.path.join(BASE_DIR, "data")
FORECAST_FOLDER = os.path.join(BASE_DIR, "forecast")

# --- API 키 및 클라이언트 관리 ---
# 이 모듈이 app.py와 독립적으로 실행될 수도 있으므로, 자체적으로 키 로드 및 클라이언트 초기화 로직을 가짐
# app.py에서 client를 전달받는 형태로 변경하는 것이 더 좋을 수 있음
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
NEWS_API_KEY_ORIGINAL = os.getenv("NEWS_API_KEY") # Finnhub News 사용 시 이 키는 불필요할 수 있음
FRED_API_KEY = os.getenv("FRED_API_KEY")
finnhub_client_stock_analysis = None

if not FINNHUB_API_KEY:
    logging.warning("stock_analysis.py: Finnhub API 키가 환경 변수에 없습니다.")
if not FRED_API_KEY:
    logging.warning("stock_analysis.py: FRED API 키가 환경 변수에 없습니다.")

def initialize_finnhub_client():
    global finnhub_client_stock_analysis, FINNHUB_API_KEY
    if finnhub_client_stock_analysis is None and FINNHUB_API_KEY:
        finnhub_client_stock_analysis = finnhub.Client(api_key=FINNHUB_API_KEY)
        logging.info("stock_analysis.py: Finnhub 클라이언트 초기화 완료.")
    elif not FINNHUB_API_KEY:
        logging.error("stock_analysis.py: Finnhub API 키가 없어 클라이언트를 초기화할 수 없습니다.")
    return finnhub_client_stock_analysis

# 레이트 리미터 설정 (app.py와 동일하게)
CALLS = 55
PERIOD = 60

@on_exception(expo, RateLimitException, max_tries=3, logger=logging)
@limits(calls=CALLS, period=PERIOD)
def call_finnhub_api_with_limit_sa(api_function, *args, **kwargs):
    """stock_analysis용 레이트 리밋 적용 Finnhub API 호출 함수"""
    try:
        # logging.info(f"SA - Calling Finnhub API (rate-limited): {api_function.__name__}")
        return api_function(*args, **kwargs)
    except RateLimitException as rle:
        logging.warning(f"SA - Rate limit exceeded for {api_function.__name__}. Retrying... Details: {rle}")
        raise
    except finnhub.FinnhubAPIException as api_e:
        logging.error(f"SA - Finnhub API Exception for {api_function.__name__}: {api_e}")
        raise
    except Exception as e:
        logging.error(f"SA - Error in call_finnhub_api_with_limit_sa for {api_function.__name__}: {e}")
        raise

# --- 데이터 가져오기 함수들 (Finnhub으로 수정) ---

def get_finnhub_stock_data(client, ticker, resolution="D", start_date_str=None, end_date_str=None, period_years=None):
    """Finnhub API를 사용하여 주가 데이터(캔들) 가져오기"""
    if not client:
        logging.error(f"SA - Finnhub 클라이언트가 초기화되지 않았습니다 ({ticker}).")
        return None

    if start_date_str and end_date_str:
        start_dt = datetime.strptime(start_date_str, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date_str, "%Y-%m-%d")
    elif period_years:
        end_dt = datetime.now()
        start_dt = end_dt - relativedelta(years=period_years)
    else: # 기본값: 약 1년
        end_dt = datetime.now()
        start_dt = end_dt - relativedelta(years=1)

    # Finnhub은 datetime 객체가 아닌 timestamp를 사용
    start_timestamp = int(datetime.combine(start_dt, dt_time.min).timestamp())
    end_timestamp = int(datetime.combine(end_dt, dt_time.max).timestamp())

    try:
        logging.info(f"SA - Finnhub 캔들 요청: {ticker}, Res: {resolution}, Start: {start_dt.strftime('%Y-%m-%d')}, End: {end_dt.strftime('%Y-%m-%d')}")
        res = call_finnhub_api_with_limit_sa(client.stock_candles, ticker, resolution, start_timestamp, end_timestamp)

        if res and res.get('s') == 'ok':
            df = pd.DataFrame(res)
            if df.empty or 't' not in df.columns:
                logging.warning(f"SA - {ticker}: Finnhub에서 캔들 데이터가 비어있거나 시간 정보가 없습니다.")
                return pd.DataFrame()
            df['t'] = pd.to_datetime(df['t'], unit='s', utc=True).dt.tz_convert('Asia/Seoul').dt.tz_localize(None)
            df.set_index('t', inplace=True)
            df.rename(columns={'o': 'Open', 'h': 'High', 'l': 'Low', 'c': 'Close', 'v': 'Volume'}, inplace=True)
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
            df.dropna(subset=['Open', 'High', 'Low', 'Close'], inplace=True) # NaN 가격 데이터 제거
            return df
        elif res and res.get('s') == 'no_data':
            logging.warning(f"SA - {ticker}: Finnhub에 해당 기간 데이터 없음.")
            return pd.DataFrame()
        else:
            logging.error(f"SA - Finnhub 캔들 API 오류 ({ticker}): {res.get('s') if res else '응답 없음'}")
            return None
    except RateLimitException:
        logging.error(f"SA - Finnhub API 호출 빈도 제한 초과 ({ticker}).")
        return None
    except Exception as e:
        logging.error(f"SA - Finnhub 캔들 데이터 요청 중 예상치 못한 오류 ({ticker}): {e}\n{traceback.format_exc()}")
        return None


def get_fear_greed_index():
    """공포-탐욕 지수 가져오기 (기존과 동일)"""
    url = "https://api.alternative.me/fng/?limit=1&format=json&date_format=world"
    value, classification = None, None
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json().get('data', [])
        if data and isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
            value_str = data[0].get('value')
            classification_str = data[0].get('value_classification')
            if value_str is not None and classification_str is not None:
                try:
                    value = int(value_str)
                    classification = classification_str
                    logging.info(f"SA - F&G 성공: {value} ({classification})")
                except (ValueError, TypeError):
                    logging.warning(f"SA - F&G 값 변환 오류: {value_str}")
            else:
                logging.warning("SA - F&G 데이터 구조 오류.")
        else:
            logging.warning("SA - F&G 데이터 형식 오류.")
    except requests.exceptions.RequestException as e:
        logging.error(f"SA - F&G API 요청 오류: {e}")
    except Exception as e:
        logging.error(f"SA - F&G 처리 오류: {e}")
    return value, classification

def get_macro_data(start_date_str, end_date_str=None, fred_key_param=None):
    """매크로 지표 데이터 가져오기 (FRED는 유지, yfinance 부분은 제거 또는 대체 필요)"""
    if fred_key_param is None: fred_key_param = FRED_API_KEY # 전역 변수 사용
    if end_date_str is None: end_date_str = datetime.today().strftime("%Y-%m-%d")

    # yfinance로 가져오던 지표들 (VIX, 금리 등)은 Finnhub에서 직접적인 대체재를 찾거나,
    # 다른 소스를 사용해야 함. 일단 여기서는 제외하고 FRED만 사용.
    # yf_tickers = {"^VIX": "VIX", "^TNX": "US10Y", "^IRX": "US13W", "DX-Y.NYB": "DXY"}
    fred_series = {"FEDFUNDS": "FedFunds", "CPIAUCSL": "CPI", "UNRATE": "UnemploymentRate"} # 예시 지표 추가
    expected_cols = ['Date'] + list(fred_series.values()) # yf_tickers 제외
    df_macro = pd.DataFrame()

    # yfinance 부분은 주석 처리 또는 삭제
    # ...

    if fred_key_param:
        try:
            fred = Fred(api_key=fred_key_param)
            fred_data_list = []
            for series_id, label in fred_series.items():
                s = fred.get_series(series_id, observation_start=start_date_str, observation_end=end_date_str).rename(label)
                s.index = pd.to_datetime(s.index).tz_localize(None) # tz_localize(None) 추가
                fred_data_list.append(s)
            if fred_data_list:
                df_macro = pd.concat(fred_data_list, axis=1)
                logging.info("SA - FRED 데이터 가져오기/병합 성공.")
        except Exception as e:
            logging.error(f"SA - FRED 데이터 가져오기 실패: {e}")
    else:
        logging.warning("SA - FRED API 키 없어 매크로(FRED) 스킵.")

    if not df_macro.empty:
        # 컬럼 존재 확인 및 NaN 채우기
        for col in expected_cols:
            if col != 'Date' and col not in df_macro.columns:
                df_macro[col] = pd.NA # 없는 컬럼은 NA로
        for col in df_macro.columns: # 숫자형 변환
            if col != 'Date': df_macro[col] = pd.to_numeric(df_macro[col], errors='coerce')

        df_macro = df_macro.sort_index().ffill().bfill() # 시계열 특성 고려하여 ffill 후 bfill
        df_macro = df_macro.reset_index().rename(columns={'index': 'Date'})
        df_macro["Date"] = pd.to_datetime(df_macro["Date"])
        logging.info("SA - 매크로 데이터 처리 완료.")
        # 반환할 컬럼만 선택 (expected_cols 순서대로, 없는건 NA로 채워짐)
        return df_macro[[col for col in expected_cols if col in df_macro.columns]]
    else:
        logging.warning("SA - 매크로 데이터 가져오기 최종 실패 (FRED 데이터 없음 또는 오류).")
        return pd.DataFrame(columns=expected_cols)


def format_market_cap_fh(mc):
    """시가총액 숫자 포맷팅 (기존과 동일)"""
    if isinstance(mc, (int, float)) and mc > 0:
        # Finnhub의 company_profile2는 marketCapitalization을 이미 백만 단위로 줄 수 있음 (확인 필요)
        # 만약 그대로 숫자라면 기존 로직 사용, 백만 단위라면 조정 필요
        # 여기서는 일단 입력값이 실제 시총 값이라고 가정
        if mc >= 1e12: return f"${mc / 1e12:.2f} T"
        elif mc >= 1e9: return f"${mc / 1e9:.2f} B"
        elif mc >= 1e6: return f"${mc / 1e6:.2f} M"
        else: return f"${mc:,.0f}"
    return "N/A"

def get_fundamental_data_finnhub(client, ticker):
    """Finnhub API를 사용하여 주요 기본 지표 가져오기"""
    if not client:
        logging.error(f"SA - Finnhub 클라이언트가 초기화되지 않았습니다 ({ticker}).")
        return {key: "N/A" for key in ["시가총액", "PER", "EPS", "배당수익률", "베타", "업종", "산업", "요약", "웹사이트"]}

    fundamentals = {key: "N/A" for key in ["시가총액", "PER", "EPS", "배당수익률", "베타", "업종", "산업", "요약", "웹사이트"]}
    try:
        # 기업 프로필 (시총, 업종, 산업, 요약, 웹사이트 등)
        profile = call_finnhub_api_with_limit_sa(client.company_profile2, symbol=ticker)
        if profile:
            # Finnhub의 marketCapitalization은 보통 백만 단위이므로 *1e6 필요할 수 있음. 문서 확인!
            # 여기서는 API가 실제 값을 준다고 가정하고 format_market_cap_fh 직접 사용
            mc = profile.get('marketCapitalization')
            if isinstance(mc, (int, float)):
                 fundamentals["시가총액"] = format_market_cap_fh(mc * 1_000_000) # 백만 단위 조정 가정
            else:
                 fundamentals["시가총액"] = "N/A"

            fundamentals["업종"] = profile.get('finnhubIndustry', "N/A") # 'sector' 대신 'finnhubIndustry'
            fundamentals["산업"] = profile.get('ipo', "N/A") # Finnhub은 'industry'를 명확히 제공하지 않음. IPO 날짜로 대체하거나 다른 필드 찾아야 함.
            fundamentals["요약"] = profile.get('description') if profile.get('description') else "N/A" # 'longBusinessSummary' 대신 'description'
            fundamentals["웹사이트"] = profile.get('weburl', "N/A")
            logging.info(f"SA - {ticker} Finnhub 프로필 로드 성공.")
        else:
            logging.warning(f"SA - {ticker} Finnhub 프로필 정보 없음.")

        # 주요 재무 지표 (PER, EPS, 배당수익률, 베타 등)
        # company_basic_financials는 'metric' 필드 아래에 여러 지표를 포함
        basic_financials = call_finnhub_api_with_limit_sa(client.company_basic_financials, ticker, 'all')
        if basic_financials and basic_financials.get('metric'):
            metrics = basic_financials['metric']
            # PER: peNormalizedAnnual 또는 peTTM. 분기별 데이터는 series.quarterly.pe 사용 가능
            fundamentals["PER"] = f"{metrics.get('peTTM', metrics.get('peNormalizedAnnual', 'N/A')):.2f}" if isinstance(metrics.get('peTTM', metrics.get('peNormalizedAnnual')), (float, int)) else "N/A"
            # EPS: epsNormalizedAnnual 또는 epsTTM
            fundamentals["EPS"] = f"{metrics.get('epsTTM', metrics.get('epsNormalizedAnnual', 'N/A')):.2f}" if isinstance(metrics.get('epsTTM', metrics.get('epsNormalizedAnnual')), (float, int)) else "N/A"
            # 배당수익률: dividendYieldAnnual 또는 dividendYieldIndicatedAnnual
            div_yield = metrics.get('dividendYieldAnnual', metrics.get('dividendYieldIndicatedAnnual'))
            fundamentals["배당수익률"] = f"{div_yield * 100:.2f}%" if isinstance(div_yield, (float, int)) and div_yield > 0 else "N/A"
            # 베타: beta
            fundamentals["베타"] = f"{metrics.get('beta', 'N/A'):.2f}"  if isinstance(metrics.get('beta'), (float, int)) else "N/A"
            logging.info(f"SA - {ticker} Finnhub 기본 재무 지표 로드 성공.")
        else:
            logging.warning(f"SA - {ticker} Finnhub 기본 재무 지표 없음.")

        return fundamentals
    except RateLimitException:
        logging.error(f"SA - Finnhub API 호출 빈도 제한 초과 ({ticker}, 기본 정보).")
        return fundamentals # 이미 채워진 값 또는 N/A 반환
    except Exception as e:
        logging.error(f"SA - Finnhub 기본 정보 가져오기 실패 ({ticker}): {e}\n{traceback.format_exc()}")
        return fundamentals


# --- 재무 추세 함수들 (Finnhub Basic Financials 기반으로 수정 시도) ---
# Finnhub 무료 API는 상세 재무제표 항목을 제공하지 않으므로,
# company_basic_financials에서 제공하는 'series' (annual 또는 quarterly) 데이터를 활용해야 합니다.
# 이는 기존 yfinance 기반 함수와 매우 다를 수 있습니다.

def get_financial_metric_trend_finnhub(client, ticker, metric_name, num_periods=4, period_type='quarterly'):
    """Finnhub company_basic_financials에서 특정 재무 지표의 추세를 가져옵니다."""
    if not client: return None
    logging.info(f"SA - {ticker}: Finnhub {period_type} '{metric_name}' 추세 ({num_periods} 기간)...")
    try:
        financials = call_finnhub_api_with_limit_sa(client.company_basic_financials, ticker, 'all')
        if not financials or 'series' not in financials or period_type not in financials['series']:
            logging.warning(f"SA - {ticker}: Finnhub에 {period_type} 재무 시리즈 데이터 없음.")
            return None

        series_data = financials['series'][period_type]
        if metric_name not in series_data or not isinstance(series_data[metric_name], list):
            logging.warning(f"SA - {ticker}: Finnhub {period_type} 시리즈에 '{metric_name}' 없음 또는 리스트 아님.")
            return None

        trend_data = []
        # 데이터는 최신순으로 정렬되어 있다고 가정 (Finnhub 문서 확인 필요)
        # 또는 'period' 필드를 기준으로 정렬 필요
        # 여기서는 API가 최신순으로 제공한다고 가정하고 상위 num_periods개 사용
        for item in sorted(series_data[metric_name], key=lambda x: x.get('period', ''), reverse=True)[:num_periods]:
            if 'period' in item and 'v' in item: # 'v'는 값(value)
                value = item['v']
                # 특정 지표에 대해 % 변환 등이 필요할 수 있음
                if metric_name in ['operatingMarginAnnual', 'operatingMarginQuarterly',
                                   'roeTTM', 'roeAnnual', 'roeQuarterly', # roe는 이미 %일 수 있음, 확인 필요
                                   'netMarginTTM', 'netMarginAnnual', 'netMarginQuarterly']: # 예시: 마진율
                    value *= 100 # %로 표시하기 위함 (API가 이미 %로 주면 이 줄은 불필요)

                trend_data.append({
                    "Date": item['period'], # YYYY-MM-DD 형식의 문자열
                    # 컬럼명을 app.py에서 사용하는 이름으로 맞춰주는 것이 좋음
                    # 예: 'Op Margin (%)', 'ROE (%)', 'D/E Ratio', 'Current Ratio'
                    # 여기서는 일반적인 컬럼명 사용
                    f"{metric_name.replace(period_type.capitalize(),'').replace('Annual','').replace('Quarterly','').replace('TTM','')} ({'%' if '%' in metric_name or 'Margin' in metric_name or 'roe' in metric_name.lower() else 'Ratio' if 'Ratio' in metric_name else ''})".strip() : round(value, 2)
                })
        if not trend_data:
            logging.warning(f"SA - {ticker}: Finnhub에서 '{metric_name}'에 대한 유효한 추세 데이터 ({num_periods}개)를 찾지 못함.")
            return None

        # 날짜 오름차순으로 정렬
        df_trend = pd.DataFrame(trend_data).sort_values(by="Date", ascending=True)
        logging.info(f"SA - {ticker}: Finnhub {metric_name} 추세 ({len(df_trend)}개) 계산 완료.")
        return df_trend.to_dict('records') # app.py의 기존 형식과 맞춤

    except RateLimitException:
        logging.error(f"SA - Finnhub API 호출 빈도 제한 초과 ({ticker}, 재무 추세).")
        return None
    except Exception as e:
        logging.error(f"SA - {ticker}: Finnhub 재무 지표('{metric_name}') 추세 계산 오류: {e}\n{traceback.format_exc()}")
        return None

# 기존 재무 추세 함수들을 위 헬퍼 함수를 사용하도록 변경
def get_operating_margin_trend_fh(client, ticker, num_periods=4):
    # Finnhub basic financials에서 operatingMarginTTM, operatingMarginAnnual, operatingMarginQuarterly 등을 확인
    # 여기서는 분기별 우선, 없으면 연간 시도
    trend = get_financial_metric_trend_finnhub(client, ticker, 'operatingMarginQuarterly', num_periods, 'quarterly')
    if not trend:
        trend = get_financial_metric_trend_finnhub(client, ticker, 'operatingMarginAnnual', num_periods, 'annual')
    # app.py에서 기대하는 컬럼명 'Op Margin (%)'으로 변경
    if trend:
        return [{"Date": rec["Date"], "Op Margin (%)": rec[next(k for k in rec if k != 'Date')]} for rec in trend]
    return []


def get_roe_trend_fh(client, ticker, num_periods=4):
    # roeTTM, roeAnnual, roeQuarterly 등
    trend = get_financial_metric_trend_finnhub(client, ticker, 'roeQuarterly', num_periods, 'quarterly')
    if not trend:
        trend = get_financial_metric_trend_finnhub(client, ticker, 'roeAnnual', num_periods, 'annual')
    if trend:
        return [{"Date": rec["Date"], "ROE (%)": rec[next(k for k in rec if k != 'Date')]} for rec in trend]
    return []


def get_debt_to_equity_trend_fh(client, ticker, num_periods=4):
    # Finnhub basic financials는 D/E Ratio를 직접 제공할 수 있음 (예: debtEquityRatioQuarterly, debtEquityRatioAnnual)
    # totalDebtToEquityTTM, totalDebtToEquityAnnual, totalDebtToEquityQuarterly 등
    trend = get_financial_metric_trend_finnhub(client, ticker, 'totalDebtToEquityQuarterly', num_periods, 'quarterly')
    if not trend:
        trend = get_financial_metric_trend_finnhub(client, ticker, 'totalDebtToEquityAnnual', num_periods, 'annual')
    if trend:
        return [{"Date": rec["Date"], "D/E Ratio": rec[next(k for k in rec if k != 'Date')]} for rec in trend]
    return []

def get_current_ratio_trend_fh(client, ticker, num_periods=4):
    # currentRatioQuarterly, currentRatioAnnual 등
    trend = get_financial_metric_trend_finnhub(client, ticker, 'currentRatioQuarterly', num_periods, 'quarterly')
    if not trend:
        trend = get_financial_metric_trend_finnhub(client, ticker, 'currentRatioAnnual', num_periods, 'annual')
    if trend:
        return [{"Date": rec["Date"], "Current Ratio": rec[next(k for k in rec if k != 'Date')]} for rec in trend]
    return []


# --- 분석 및 시각화 함수들 ---
def plot_stock_chart_fh(client, ticker, start_date_str=None, end_date_str=None, period_years=None, resolution="D"):
    """주가 차트 Figure 객체 반환 (Finnhub 데이터 사용)"""
    df = get_finnhub_stock_data(client, ticker, resolution, start_date_str, end_date_str, period_years)
    if df is None or df.empty:
        logging.error(f"SA - {ticker} Finnhub 차트 실패: 데이터 없음")
        fig = go.Figure() # 빈 Figure 반환
        fig.update_layout(title=f'{ticker} 주가/거래량 차트 (데이터 없음)', xaxis_title="날짜", yaxis_title="가격")
        return fig # None 대신 빈 Figure 반환하여 app.py에서 오류 방지
    try:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='가격'), row=1, col=1)
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='거래량', marker_color='rgba(0,0,100,0.6)'), row=2, col=1)
        fig.update_layout(title=f'{ticker} 주가/거래량 차트 (Finnhub)', yaxis_title='가격', yaxis2_title='거래량', xaxis_rangeslider_visible=False, hovermode='x unified', margin=dict(l=20, r=20, t=40, b=20))
        fig.update_yaxes(title_text="가격", row=1, col=1)
        fig.update_yaxes(title_text="거래량", row=2, col=1)
        logging.info(f"SA - {ticker} Finnhub 차트 생성 완료")
        return fig
    except Exception as e:
        logging.error(f"SA - {ticker} Finnhub 차트 생성 오류: {e}")
        fig = go.Figure()
        fig.update_layout(title=f'{ticker} 주가/거래량 차트 (생성 오류)', xaxis_title="날짜", yaxis_title="가격")
        return fig

def get_news_sentiment_finnhub(client, ticker):
    """Finnhub API를 사용하여 뉴스 및 감정 분석"""
    if not client:
        logging.warning("SA - Finnhub 클라이언트 없음 (뉴스 분석).")
        return ["Finnhub 클라이언트 미설정."]

    output = []
    total_sentiment_score = 0
    article_count = 0

    try:
        # 최근 뉴스 가져오기 (Finnhub은 날짜 범위 지정 가능)
        # 예: 최근 7일 뉴스
        today_str = datetime.now().strftime("%Y-%m-%d")
        seven_days_ago_str = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        news_items = call_finnhub_api_with_limit_sa(client.company_news, ticker, _from=seven_days_ago_str, to=today_str)

        if not news_items:
            logging.info(f"SA - {ticker}: Finnhub 관련 뉴스 없음 (최근 7일).")
            return ["Finnhub 관련 뉴스 없음 (최근 7일)."]

        for i, article in enumerate(news_items[:20], 1): # 최대 20개 처리
            headline = article.get('headline', 'N/A')
            summary = article.get('summary', '')
            source = article.get('source', '')
            url = article.get('url', '#')
            # Finnhub news_sentiment는 별도 호출 필요 (뉴스 ID 기반이 아님)
            # 여기서는 뉴스 헤드라인과 요약을 기반으로 간단히 표시하거나,
            # 만약 Finnhub의 'sentiment' 필드가 뉴스 객체에 있다면 사용
            sentiment_score = article.get('sentiment', None) # 이 필드가 있는지 확인 필요, 없으면 TextBlob 등 사용
            # Finnhub 자체 company_news에는 sentiment 필드가 없을 수 있음.
            # news_sentiment 엔드포인트는 티커에 대한 전체적인 감정 점수를 제공.
            # 여기서는 개별 기사 감정은 없다고 가정하고, 헤드라인만 나열

            output.append(f"{i}. [{source}] {headline} - <a href='{url}' target='_blank'>기사보기</a>")
            # sentiment_score가 있다면 활용
            # if sentiment_score is not None:
            #     output[-1] += f" (감정: {sentiment_score:.2f})"
            #     total_sentiment_score += sentiment_score
            #     article_count +=1

        # 티커에 대한 전반적인 뉴스 감정 (Finnhub news_sentiment 엔드포인트)
        company_sentiment = call_finnhub_api_with_limit_sa(client.news_sentiment, ticker)
        if company_sentiment and company_sentiment.get('sentiment') and company_sentiment['sentiment'].get('companyNewsScore') is not None:
            avg_pol = company_sentiment['sentiment']['companyNewsScore']
            buzz = company_sentiment.get('buzz', {}).get('articlesInLastWeek', 'N/A')
            output.insert(0, f"Finnhub 뉴스 감정 분석: 평균 점수 {avg_pol:.2f} (지난 주 기사 수: {buzz})")
            article_count = 1 # 대표 점수가 있으므로 count=1로 설정하여 평균 표시
            total_sentiment_score = avg_pol # 평균 점수를 대표로 사용
        elif article_count > 0: # 개별 기사 감정이 있었다면 (현재 로직에서는 이 부분 실행 안됨)
            avg_pol = total_sentiment_score / article_count
            output.insert(0, f"총 {article_count}개 분석 | 평균 감성: {avg_pol:.2f} (TextBlob 기반 - 가정)")
        else:
            output.insert(0, f"총 {len(news_items[:20])}개 뉴스 헤드라인 (감정 분석은 요약 정보 참고)")


        logging.info(f"SA - {ticker} Finnhub 뉴스 분석 완료.")
        return output

    except RateLimitException:
        logging.error(f"SA - Finnhub API 호출 빈도 제한 초과 ({ticker}, 뉴스).")
        return ["Finnhub 뉴스 API 호출 빈도 제한."]
    except Exception as e:
        logging.error(f"SA - Finnhub 뉴스 분석 오류 ({ticker}): {e}\n{traceback.format_exc()}")
        return [f"Finnhub 뉴스 분석 중 오류 발생: {e}"]


def run_prophet_forecast_fh(client, ticker, start_date_str, end_date_str=None, forecast_days=30, fred_key_param=None, changepoint_prior_scale=0.05):
    logging.info(f"SA - {ticker}: Finnhub Prophet 예측 시작 (cp_prior={changepoint_prior_scale})...")
    if end_date_str is None: end_date_str = datetime.today().strftime("%Y-%m-%d")

    # 1. 주가 데이터 로딩 (Finnhub 사용)
    df_stock_initial = get_finnhub_stock_data(client, ticker, "D", start_date_str, end_date_str)
    if df_stock_initial is None or df_stock_initial.empty:
        logging.error(f"SA - Prophet 실패: {ticker} Finnhub 주가 데이터 로딩 실패 또는 없음.")
        return None, None, None, 0.0 # MAPE는 0.0 또는 다른 기본값

    df_stock_processed = df_stock_initial.reset_index()[["t", "Close", "Open", "High", "Low", "Volume"]].copy() # 't' 사용
    df_stock_processed.rename(columns={"t": "Date"}, inplace=True) # 'Date'로 변경
    df_stock_processed["Date"] = pd.to_datetime(df_stock_processed["Date"])

    if df_stock_processed["Close"].isnull().any():
        rows_before = len(df_stock_processed)
        df_stock_processed.dropna(subset=["Close"], inplace=True)
        logging.warning(f"SA - {ticker}: 'Close' 컬럼 NaN 값으로 인해 {rows_before - len(df_stock_processed)} 행 제거됨 (Prophet).")
    if df_stock_processed.empty:
        logging.error(f"SA - {ticker}: 'Close'가 유효한 데이터가 없습니다 (Prophet).")
        return None, None, None, 0.0

    logging.info(f"SA - {ticker}: Finnhub 초기 주가 데이터 로딩 및 기본 처리 완료 (Shape: {df_stock_processed.shape}).")

    # 2. 매크로 데이터 로딩 및 병합 (FRED는 유지)
    df_macro = get_macro_data(start_date_str, end_date_str, fred_key_param)
    macro_cols_available_for_prophet = []
    if df_macro is not None and not df_macro.empty:
        df_merged = pd.merge(df_stock_processed, df_macro, on="Date", how="left")
        logging.info(f"SA - {ticker}: 주가/매크로 데이터 병합 완료 (Prophet).")
        # 사용할 매크로 컬럼 (FRED에서 온 것들)
        fred_regressors = [col for col in df_macro.columns if col != 'Date']
        for col in fred_regressors:
            if col in df_merged.columns:
                df_merged[col] = pd.to_numeric(df_merged[col], errors='coerce').ffill().bfill()
                if not df_merged[col].isnull().any(): # NaN 없는 유효한 Regressor만 추가
                    macro_cols_available_for_prophet.append(col)
        logging.info(f"SA - {ticker}: 매크로 변수 NaN 처리 완료 (Prophet). 사용 가능 Regressor: {macro_cols_available_for_prophet}")
    else:
        df_merged = df_stock_processed # 매크로 없으면 주가 데이터만 사용
        logging.warning(f"SA - {ticker}: 매크로 데이터 없음. 주가 데이터만 사용 (Prophet).")


    # 3. 기술적 지표 계산 (pandas_ta 사용)
    tech_indicators_to_add_prophet = []
    try:
        df_merged.ta.rsi(close='Close', length=14, append=True, col_names=('RSI_14_Prophet'))
        df_merged.ta.macd(close='Close', fast=12, slow=26, signal=9, append=True, col_names=('MACD_12_26_9_Prophet', 'MACDh_12_26_9_Prophet', 'MACDs_12_26_9_Prophet'))
        logging.info(f"SA - {ticker}: RSI, MACD 계산 완료 (Prophet).")

        # MACD는 선만 사용 (MACD_12_26_9_Prophet)
        tech_candidates_prophet = ['RSI_14_Prophet', 'MACDs_12_26_9_Prophet'] # 컬럼명에 _Prophet 추가하여 구분
        for ti in tech_candidates_prophet:
            if ti in df_merged.columns:
                df_merged[ti] = df_merged[ti].ffill().bfill() # NaN 처리
                if not df_merged[ti].isnull().any():
                    tech_indicators_to_add_prophet.append(ti)
                else:
                    logging.warning(f"SA - {ticker}: 기술 지표 '{ti}'에 처리 후에도 NaN 존재하여 Regressor에서 제외 (Prophet).")
            else:
                logging.warning(f"SA - {ticker}: 기술 지표 '{ti}'가 생성되지 않았습니다 (Prophet).")
    except Exception as ta_err:
        logging.error(f"SA - {ticker}: 기술적 지표 계산 중 오류 (Prophet): {ta_err}")
        tech_indicators_to_add_prophet = []

    # 4. 최종 데이터 검증 및 Prophet 준비
    if df_merged.empty or len(df_merged) < 30:
        logging.error(f"SA - Prophet 실패: 최종 데이터 부족({len(df_merged)}).")
        return None, None, None, 0.0

    df_prophet_input = df_merged.rename(columns={"Date": "ds", "Close": "y"})
    df_prophet_input['ds'] = pd.to_datetime(df_prophet_input['ds'])

    m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False, changepoint_prior_scale=changepoint_prior_scale)

    all_prophet_regressors = macro_cols_available_for_prophet + tech_indicators_to_add_prophet
    if all_prophet_regressors:
        for regressor in all_prophet_regressors:
            if regressor in df_prophet_input.columns: # 최종 확인
                 m.add_regressor(regressor)
        logging.info(f"SA - {ticker}: Prophet Regressors 추가됨: {all_prophet_regressors}")
    else:
        logging.info(f"SA - {ticker}: 유효한 Prophet Regressor 없음.")


    # 5. 학습, 예측, CV 실행
    forecast_dict, fig_fcst, cv_path, mape = None, None, None, 0.0 # mape 기본값 0.0
    try:
        logging.info(f"SA - {ticker}: Prophet 모델 학습 시작 (Regressors: {all_prophet_regressors})...")
        m.fit(df_prophet_input[['ds', 'y'] + all_prophet_regressors]) # Regressor 포함하여 학습
        logging.info(f"SA - {ticker}: Prophet 학습 완료.")
        os.makedirs(FORECAST_FOLDER, exist_ok=True)

        # CV (기존 로직 유지)
        try:
            data_len_days = (df_prophet_input['ds'].max() - df_prophet_input['ds'].min()).days
            initial_cv_days = max(180, int(data_len_days * 0.5))
            period_cv_days = max(30, int(initial_cv_days * 0.2))
            horizon_cv_days = forecast_days
            initial_cv, period_cv, horizon_cv = f'{initial_cv_days} days', f'{period_cv_days} days', f'{horizon_cv_days} days'

            if len(df_prophet_input) > initial_cv_days + horizon_cv_days + period_cv_days:
                df_cv = cross_validation(m, initial=initial_cv, period=period_cv, horizon=horizon_cv, parallel=None) # parallel=None 권장
                df_p = performance_metrics(df_cv)
                mape = df_p["mape"].mean() * 100
                logging.info(f"SA - Prophet CV 평균 MAPE ({ticker}): {mape:.2f}%")
                fig_cv = plot_cross_validation_metric(df_cv, metric='mape')
                plt.title(f'{ticker} CV MAPE (Finnhub, cp={changepoint_prior_scale})')
                cv_path = os.path.join(FORECAST_FOLDER, f"{ticker}_finnhub_cv_mape_cp{changepoint_prior_scale}.png")
                fig_cv.savefig(cv_path)
                plt.close(fig_cv)
            else:
                logging.warning(f"SA - {ticker}: 데이터 기간 부족하여 Prophet CV 건너<0xEB>니다.")
                cv_path = None; mape = 0.0 # CV 못하면 MAPE 0으로
        except Exception as cv_e:
            logging.error(f"SA - Prophet CV 중 오류 ({ticker}): {cv_e}")
            cv_path = None; mape = 0.0

        # 미래 예측
        future = m.make_future_dataframe(periods=forecast_days)
        # 미래 Regressor 값 처리 (ffill 후 마지막 값으로 채우기)
        if all_prophet_regressors:
            temp_m_fh = df_prophet_input[['ds'] + all_prophet_regressors].copy()
            future = future.merge(temp_m_fh, on='ds', how='left')
            for col in all_prophet_regressors:
                if col in future.columns:
                    last_valid_val = df_prophet_input[col].dropna().iloc[-1] if not df_prophet_input[col].dropna().empty else 0
                    future[col] = future[col].ffill().fillna(last_valid_val)

        forecast = m.predict(future)
        csv_fn = os.path.join(FORECAST_FOLDER, f"{ticker}_finnhub_forecast_cp{changepoint_prior_scale}.csv")
        forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy().assign(ds=lambda dfx: dfx['ds'].dt.strftime('%Y-%m-%d')).to_csv(csv_fn, index=False)
        fig_fcst = plot_plotly(m, forecast)
        fig_fcst.update_layout(title=f'{ticker} Price Forecast (Finnhub, cp={changepoint_prior_scale})', margin=dict(l=20,r=20,t=40,b=20))
        forecast_dict = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(forecast_days).to_dict('records')
        for rec in forecast_dict: rec['ds'] = rec['ds'].strftime('%Y-%m-%d')

        return forecast_dict, fig_fcst, cv_path, mape
    except Exception as e:
        logging.error(f"SA - Prophet 학습/예측 단계에서 오류 발생 ({ticker}): {e}\n{traceback.format_exc()}")
        return None, None, None, 0.0


# --- 메인 분석 함수 (Finnhub 적용) ---
def analyze_stock(ticker, finnhub_client_param, news_api_key_unused, fred_api_key_param,
                  analysis_period_years=2, forecast_days=30, num_trend_periods=4, changepoint_prior_scale=0.05): # num_trend_periods를 받음
    logging.info(f"--- {ticker} Finnhub 주식 분석 시작 (cp_prior={changepoint_prior_scale}) ---")
    output_results = {"error": None}

    # Finnhub 클라이언트가 제대로 전달되었는지 확인
    if not finnhub_client_param:
        logging.error(f"SA - Finnhub 클라이언트가 analyze_stock 함수에 전달되지 않았습니다 ({ticker}).")
        output_results["error"] = "Finnhub 클라이언트 설정 오류"
        return output_results # 오류 상태로 즉시 반환

    try:
        end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        start_date = end_date - relativedelta(years=analysis_period_years)
        start_date_str = start_date.strftime("%Y-%m-%d")
        end_date_str = end_date.strftime("%Y-%m-%d")
    except Exception as e:
        logging.error(f"SA - 날짜 설정 오류: {e}")
        output_results["error"] = f"날짜 설정 오류: {e}"
        return output_results

    # 주가 데이터 로드 (Finnhub)
    df_stock_full = get_finnhub_stock_data(finnhub_client_param, ticker, "D", start_date_str, end_date_str)
    stock_data_valid = df_stock_full is not None and not df_stock_full.empty

    if stock_data_valid:
        output_results['current_price'] = f"{df_stock_full['Close'].iloc[-1]:.2f}" if not df_stock_full['Close'].empty and pd.notna(df_stock_full['Close'].iloc[-1]) else "N/A"
        output_results['data_points'] = len(df_stock_full)
    else:
        output_results['current_price'] = "N/A"; output_results['data_points'] = 0
        logging.warning(f"SA - {ticker}: Finnhub 분석 기간 내 유효 주가 정보 없음.")
        # 주가 데이터 없으면 많은 분석 불가, 하지만 일부 정보(펀더멘탈 등)는 시도 가능

    output_results['analysis_period_start'] = start_date_str
    output_results['analysis_period_end'] = end_date_str

    output_results['stock_chart_fig'] = plot_stock_chart_fh(finnhub_client_param, ticker, start_date_str, end_date_str, resolution="D")
    output_results['fundamentals'] = get_fundamental_data_finnhub(finnhub_client_param, ticker)

    # 재무 추세 (Finnhub basic financials 기반)
    # app.py에서 사용하는 컬럼명과 일치시키도록 각 함수 내부에서 조정됨
    output_results['operating_margin_trend'] = get_operating_margin_trend_fh(finnhub_client_param, ticker, num_trend_periods) or [] # num_periods -> num_trend_periods로 변경
    output_results['roe_trend'] = get_roe_trend_fh(finnhub_client_param, ticker, num_trend_periods) or [] # num_periods -> num_trend_periods로 변경
    output_results['debt_to_equity_trend'] = get_debt_to_equity_trend_fh(finnhub_client_param, ticker, num_trend_periods) or [] # num_periods -> num_trend_periods로 변경
    output_results['current_ratio_trend'] = get_current_ratio_trend_fh(finnhub_client_param, ticker, num_trend_periods) or [] # num_periods -> num_trend_periods로 변경

    output_results['news_sentiment'] = get_news_sentiment_finnhub(finnhub_client_param, ticker) or ["Finnhub 뉴스 분석 실패"]
    fg_value, fg_class = get_fear_greed_index()
    output_results['fear_greed_index'] = {'value': fg_value, 'classification': fg_class} if fg_value is not None else "N/A"

    # Prophet 예측 (Finnhub 주가 데이터 사용)
    if stock_data_valid and output_results['data_points'] > 30:
        prophet_res = run_prophet_forecast_fh(
            finnhub_client_param, ticker, start_date_str, end_date_str,
            forecast_days, fred_api_key_param, changepoint_prior_scale
        )
        if prophet_res and isinstance(prophet_res, tuple) and len(prophet_res) == 4:
            fc_list, fc_fig, cv_path_fh, mape_fh = prophet_res
            output_results['prophet_forecast'] = fc_list or "Finnhub 예측 실패"
            output_results['forecast_fig'] = fc_fig
            output_results['cv_plot_path'] = cv_path_fh
            output_results['mape'] = mape_fh
            output_results['warn_high_mape'] = (mape_fh > 20) if mape_fh is not None else False
        else:
            output_results['prophet_forecast'] = "Finnhub 예측 실행 오류"
            output_results['forecast_fig'] = None; output_results['cv_plot_path'] = None
            output_results['mape'] = 0.0; output_results['warn_high_mape'] = False
    else:
        msg = f"Finnhub 데이터 부족({output_results['data_points']})" if stock_data_valid else "Finnhub 주가 정보 없음"
        output_results['prophet_forecast'] = f"{msg}으로 Finnhub 예측 불가"
        output_results['forecast_fig'] = None; output_results['cv_plot_path'] = None
        output_results['mape'] = 0.0; output_results['warn_high_mape'] = False

    logging.info(f"--- {ticker} Finnhub 주식 분석 완료 ---")
    return output_results


# --- 메인 실행 부분 (테스트용 - Finnhub 적용) ---
if __name__ == "__main__":
    print(f"stock_analysis.py (Finnhub Version) 직접 실행 (테스트 목적, Base directory: {BASE_DIR}).")
    # .env 파일에서 API 키 로드 시도 (테스트 실행 시)
    dotenv_path_sa_main = os.path.join(BASE_DIR, '.env')
    if os.path.exists(dotenv_path_sa_main):
        load_dotenv(dotenv_path=dotenv_path_sa_main)
        FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY") # 전역 변수 업데이트
        FRED_API_KEY = os.getenv("FRED_API_KEY")       # 전역 변수 업데이트
        NEWS_API_KEY_ORIGINAL = os.getenv("NEWS_API_KEY")
        logging.info("SA - 테스트: .env 파일에서 API 키 로드 시도 완료.")

    target_ticker_fh = "AAPL" # 테스트용 티커
    cps_test_fh = 0.05

    # Finnhub 클라이언트 초기화
    test_finnhub_client = initialize_finnhub_client()

    if not test_finnhub_client:
        print("테스트 실패: Finnhub 클라이언트 초기화 불가. API 키를 확인하세요.")
    else:
        print(f"Finnhub 클라이언트 사용 가능. {target_ticker_fh} 분석 테스트 시작...")
        test_results_fh = analyze_stock(
            ticker=target_ticker_fh,
            finnhub_client_param=test_finnhub_client,
            news_api_key_unused=NEWS_API_KEY_ORIGINAL, # Finnhub 뉴스 사용으로 이 키는 불필요
            fred_api_key_param=FRED_API_KEY,
            analysis_period_years=1, forecast_days=15, num_trend_periods=4,
            changepoint_prior_scale=cps_test_fh
        )

        print(f"\n--- Finnhub 테스트 실행 결과 요약 (Changepoint Prior: {cps_test_fh}) ---")
        if test_results_fh and test_results_fh.get("error") is None:
            for key, value in test_results_fh.items():
                if key == "error": continue # 오류는 이미 위에서 처리
                if 'fig' in key and value is not None and isinstance(value, go.Figure): # Figure 객체인지 확인
                    print(f"- {key.replace('_',' ').title()}: Plotly Figure 생성됨 (표시 안 함)")
                elif key == 'fundamentals' and isinstance(value, dict):
                    print("- Fundamentals (Finnhub):")
                    for k_f, v_f in value.items():
                        if k_f == '요약' and isinstance(v_f, str) and len(v_f) > 60: print(f"    - {k_f}: {v_f[:60]}...")
                        else: print(f"    - {k_f}: {v_f}")
                elif '_trend' in key and isinstance(value, list):
                    print(f"- {key.replace('_',' ').title()} ({len(value)} 기간, Finnhub):")
                    for item_t in value[:2]: print(f"    - {item_t}") # 최대 2개만 출력
                    if len(value) > 2: print("     ...")
                elif key == 'prophet_forecast':
                    status_pf = "예측 실패/오류"
                    if isinstance(value, list) and value: status_pf = f"{len(value)}일 예측 생성 (첫 날: {value[0].get('ds')} ~ {value[0].get('yhat'):.2f})"
                    elif isinstance(value, str): status_pf = value
                    print(f"- Prophet Forecast (Finnhub): {status_pf}")
                elif key == 'news_sentiment':
                    status_ns = "뉴스 분석 실패/오류"
                    if isinstance(value, list) and value: status_ns = f"{len(value)-1 if value[0].startswith('Finnhub 뉴스 감정 분석') else len(value)}개 뉴스/요약 (첫 줄: {value[0][:60]}...)"
                    print(f"- News Sentiment (Finnhub): {status_ns}")
                else:
                    print(f"- {key.replace('_',' ').title()}: {value}")
        elif test_results_fh and test_results_fh.get("error"):
            print(f"분석 중 오류 발생: {test_results_fh['error']}")
        else:
            print("Finnhub 테스트 분석 실패 (결과 없음 또는 알 수 없는 오류).")

    print("\n--- Finnhub 테스트 실행 종료 ---")