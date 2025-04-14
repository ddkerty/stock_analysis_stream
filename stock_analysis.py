# stock_analysis.py (get_operating_margin_trend 개선: 컬럼명 탐색, 기간 파라미터화)

import os
import logging
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from textblob import TextBlob
import requests
from prophet import Prophet
from prophet.plot import plot_plotly
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from dotenv import load_dotenv
from fredapi import Fred
import traceback
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_cross_validation_metric
import matplotlib.pyplot as plt
import warnings
import locale
import re # 정규 표현식 사용 위해 추가

warnings.simplefilter(action='ignore', category=FutureWarning)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()
    logging.info(f"__file__ 변수 없음. 현재 작업 디렉토리 사용: {BASE_DIR}")

CHARTS_FOLDER = os.path.join(BASE_DIR, "charts")
DATA_FOLDER = os.path.join(BASE_DIR, "data")
FORECAST_FOLDER = os.path.join(BASE_DIR, "forecast")

try:
    dotenv_path = os.path.join(BASE_DIR, '.env')
    if os.path.exists(dotenv_path): load_dotenv(dotenv_path=dotenv_path); logging.info(f".env 로드 성공: {dotenv_path}")
    else: logging.warning(f".env 파일 없음: {dotenv_path}")
except Exception as e: logging.error(f".env 로드 오류: {e}")

NEWS_API_KEY = os.getenv("NEWS_API_KEY")
FRED_API_KEY = os.getenv("FRED_API_KEY")

if not NEWS_API_KEY: logging.warning("NEWS_API_KEY 없음.")
if not FRED_API_KEY: logging.warning("FRED_API_KEY 없음.")
if NEWS_API_KEY and FRED_API_KEY: logging.info("API 키 로드 시도 완료.")


# --- 데이터 가져오기 함수들 ---
# get_fear_greed_index, get_stock_data, get_macro_data 함수는 이전과 동일하게 유지
# ... (이전 답변의 get_fear_greed_index, get_stock_data, get_macro_data 코드 붙여넣기) ...
def get_fear_greed_index():
    """공포-탐욕 지수를 API에서 가져옵니다."""
    url = "https://api.alternative.me/fng/?limit=1&format=json&date_format=world"
    try:
        response = requests.get(url, timeout=10); response.raise_for_status()
        data = response.json()['data'][0]; value = int(data['value']); classification = data['value_classification']
        logging.info(f"공포-탐욕 지수 가져오기 성공: {value} ({classification})"); return value, classification
    except Exception as e: logging.error(f"공포-탐욕 지수 오류: {e}"); return None, None

def get_stock_data(ticker, start_date=None, end_date=None, period="1y"):
    """지정된 종목의 주가 데이터를 yfinance로 가져옵니다."""
    try:
        stock = yf.Ticker(ticker)
        if start_date and end_date: data = stock.history(start=start_date, end=end_date); logging.info(f"{ticker} 주가 성공 ({start_date}~{end_date})")
        else: data = stock.history(period=period); logging.info(f"{ticker} 주가 성공 (기간: {period})")
        if data.empty: logging.warning(f"{ticker} 주가 데이터 비어있음."); return None
        if isinstance(data.index, pd.DatetimeIndex): data.index = data.index.tz_localize(None)
        return data
    except Exception as e: logging.error(f"티커 '{ticker}' 주가 데이터 실패: {e}"); return None

def get_macro_data(start_date, end_date=None, fred_key=None):
    """VIX, US10Y, 13주 국채(IRX), DXY, 연방기금 금리 데이터를 가져옵니다."""
    if end_date is None: end_date = datetime.today().strftime("%Y-%m-%d")
    tickers = {"^VIX": "VIX", "^TNX": "US10Y", "^IRX": "US13W", "DX-Y.NYB": "DXY"}; df_macro = pd.DataFrame()
    all_yf_data = []
    for tk, label in tickers.items():
        try:
            tmp = yf.download(tk, start=start_date, end=end_date, progress=False, timeout=10)
            if not tmp.empty:
                tmp = tmp[['Close']].rename(columns={"Close": label})
                if isinstance(tmp.index, pd.DatetimeIndex): tmp.index = tmp.index.tz_localize(None)
                all_yf_data.append(tmp); logging.info(f"{label} 데이터 성공")
            else: logging.warning(f"{label} 데이터 비어있음.")
        except Exception as e: logging.error(f"{label} 데이터 실패: {e}")
    if all_yf_data:
        df_macro = pd.concat(all_yf_data, axis=1)
        if isinstance(df_macro.columns, pd.MultiIndex): df_macro.columns = df_macro.columns.get_level_values(-1)
    if fred_key:
        try:
            fred = Fred(api_key=fred_key); fedfunds = fred.get_series("FEDFUNDS", start_date=start_date, end_date=end_date).rename("FedFunds")
            if isinstance(fedfunds.index, pd.DatetimeIndex): fedfunds.index = fedfunds.index.tz_localize(None)
            if not df_macro.empty: df_macro = df_macro.merge(fedfunds, left_index=True, right_index=True, how='outer')
            else: df_macro = pd.DataFrame(fedfunds)
            logging.info("FRED 데이터 병합/가져오기 성공")
        except Exception as e: logging.error(f"FRED 데이터 실패: {e}"); # Fallback logic for FedFunds column if needed
    else: logging.warning("FRED 키 없어 FRED 데이터 스킵.")
    if not df_macro.empty:
        if 'FedFunds' not in df_macro.columns: df_macro['FedFunds'] = pd.NA # Ensure FedFunds column exists if skipped/failed
        df_macro.index = pd.to_datetime(df_macro.index).tz_localize(None)
        df_macro = df_macro.sort_index().ffill().bfill().reset_index().rename(columns={'index': 'Date'}); df_macro["Date"] = pd.to_datetime(df_macro["Date"])
        logging.info("매크로 데이터 처리 완료."); return df_macro
    else: logging.warning("매크로 데이터 가져오기 실패."); return pd.DataFrame()


# --- 기본적 분석 데이터 가져오기 함수 ---

def format_market_cap(mc):
    # ... (이전과 동일) ...
    """시가총액 숫자를 읽기 쉬운 문자열(T/B/M 단위)로 변환합니다."""
    if isinstance(mc, (int, float)) and mc > 0:
        if mc >= 1e12: return f"${mc / 1e12:.2f} T"
        elif mc >= 1e9: return f"${mc / 1e9:.2f} B"
        elif mc >= 1e6: return f"${mc / 1e6:.2f} M"
        else: return f"${mc:,.0f}"
    return "N/A"

def get_fundamental_data(ticker):
    # ... (이전과 동일) ...
    """yfinance의 .info를 사용하여 주요 기본적 분석 지표를 가져옵니다."""
    logging.info(f"{ticker}: 기본 정보(.info) 가져오기 시도...")
    fundamentals = {key: "N/A" for key in ["시가총액", "PER", "EPS", "배당수익률", "베타", "업종", "산업", "요약"]}
    try:
        stock = yf.Ticker(ticker); info = stock.info
        if not info or info.get('regularMarketPrice') is None: logging.warning(f"'{ticker}' 유효 .info 데이터 없음."); return fundamentals
        fundamentals["시가총액"] = format_market_cap(info.get("marketCap"))
        fwd_pe = info.get('forwardPE'); trl_pe = info.get('trailingPE')
        if isinstance(fwd_pe, (int, float)): fundamentals["PER"] = f"{fwd_pe:.2f} (Forward)"
        elif isinstance(trl_pe, (int, float)): fundamentals["PER"] = f"{trl_pe:.2f} (Trailing)"
        eps_val = info.get('trailingEps'); fundamentals["EPS"] = f"{eps_val:.2f}" if isinstance(eps_val, (int, float)) else "N/A"
        div_yield = info.get('dividendYield'); fundamentals["배당수익률"] = f"{div_yield * 100:.2f}%" if isinstance(div_yield, (int, float)) and div_yield > 0 else "N/A"
        beta_val = info.get('beta'); fundamentals["베타"] = f"{beta_val:.2f}" if isinstance(beta_val, (int, float)) else "N/A"
        fundamentals["업종"] = info.get("sector", "N/A"); fundamentals["산업"] = info.get("industry", "N/A"); fundamentals["요약"] = info.get("longBusinessSummary", "N/A")
        logging.info(f"{ticker} 기본 정보 가져오기 성공."); return fundamentals
    except Exception as e: logging.error(f"{ticker} 기본 정보(.info) 실패: {e}"); return fundamentals

# stock_analysis.py 파일에 아래 함수 추가

import numpy as np # 숫자 계산 및 NaN 처리 위해 추가
import re # 컬럼명 탐색 위해 추가

# (find_financial_statement_item 헬퍼 함수는 이미 있다고 가정, 없다면 이전 답변 참고하여 추가)
def find_financial_statement_item(index, keywords, contains_mode=True, case_sensitive=False):
    """재무제표 인덱스에서 키워드를 포함하거나 일치하는 항목 이름을 찾습니다."""
    if not isinstance(index, pd.Index): return None
    pattern = r'\s*'.join(keywords); flags = 0 if case_sensitive else re.IGNORECASE
    for item in index:
        if contains_mode:
             try:
                  if re.search(pattern, item, flags=flags): return item
             except TypeError: continue
        else:
            cleaned_item = re.sub(r'\W+', '', str(item)).lower(); cleaned_pattern = re.sub(r'\W+', '', ''.join(keywords)).lower()
            if cleaned_item == cleaned_pattern: return item
    first_keyword_pattern = keywords[0]
    for item in index:
         try:
              if re.search(first_keyword_pattern, item, flags=flags): logging.warning(f"정확 항목명 매칭 실패. '{keywords}' 대신 '{item}' 시도."); return item
         except TypeError: continue
    return None

def get_roe_trend(ticker, num_periods=4):
    """최근 분기별 ROE(%) 추세를 계산하여 반환합니다 (딕셔너리 리스트)."""
    logging.info(f"{ticker}: 분기별 ROE 추세 가져오기 시도 (최근 {num_periods} 분기)...")
    try:
        stock = yf.Ticker(ticker)
        # 분기별 손익계산서 및 대차대조표 가져오기
        qf = stock.quarterly_financials
        qbs = stock.quarterly_balance_sheet

        if qf.empty or qbs.empty:
            logging.warning(f"{ticker}: 분기별 재무 또는 대차대조표 데이터 없음.")
            return None

        # 필요한 항목 이름 찾기 (유연하게)
        # 순이익: Net Income, Net Income From Continuing Operations 등
        net_income_col = find_financial_statement_item(qf.index, ['Net', 'Income'])
        # 자본총계: Stockholders Equity, Total Equity Gross Minority Interest 등
        equity_col = find_financial_statement_item(qbs.index, ['Stockholder', 'Equity']) or \
                     find_financial_statement_item(qbs.index, ['Total', 'Equity', 'Gross'])

        if not net_income_col or not equity_col:
             logging.warning(f"{ticker}: 순이익('{net_income_col}') 또는 자본총계('{equity_col}') 항목 찾기 실패.")
             # logging.debug(f"손익계산서 항목: {qf.index.tolist()}")
             # logging.debug(f"대차대조표 항목: {qbs.index.tolist()}")
             return None

        # 최신 N개 분기 데이터 선택 및 정렬
        qf_recent = qf.loc[[net_income_col]].iloc[:, :num_periods].T
        qbs_recent = qbs.loc[[equity_col]].iloc[:, :num_periods].T

        # 두 데이터프레임의 인덱스(날짜)를 기준으로 합치기 (Outer join으로 최대한 살림)
        df_trend = pd.merge(qf_recent, qbs_recent, left_index=True, right_index=True, how='outer')
        df_trend.index = pd.to_datetime(df_trend.index)
        df_trend = df_trend.sort_index(ascending=True)

        # 숫자형 변환 및 오류 처리
        df_trend[net_income_col] = pd.to_numeric(df_trend[net_income_col], errors='coerce')
        df_trend[equity_col] = pd.to_numeric(df_trend[equity_col], errors='coerce')

        # ROE 계산: 순이익 / 자본총계 * 100
        # 자본총계가 0 이하인 경우 계산 불가 (NaN 처리)
        df_trend[equity_col] = df_trend[equity_col].apply(lambda x: x if x > 0 else np.nan)
        df_trend.dropna(subset=[net_income_col, equity_col], inplace=True) # 계산 불가능 행 제거

        if df_trend.empty:
             logging.warning(f"{ticker}: ROE 계산 가능한 데이터 부족.")
             return None

        df_trend['ROE (%)'] = (df_trend[net_income_col] / df_trend[equity_col]) * 100
        df_trend['ROE (%)'] = df_trend['ROE (%)'].round(2)

        # 결과 포맷팅
        result_trend = df_trend[['ROE (%)']].reset_index()
        result_trend.rename(columns={'index': 'Date'}, inplace=True)
        result_trend['Date'] = result_trend['Date'].dt.strftime('%Y-%m-%d')

        logging.info(f"{ticker}: 최근 {len(result_trend)}개 분기 ROE(%) 추세 계산 완료.")
        return result_trend.to_dict('records')

    except Exception as e:
        logging.error(f"{ticker}: ROE 추세 계산 중 오류: {e}")
        # logging.error(traceback.format_exc()) # 상세 디버깅 시
        return None
# --- 🍀 영업이익률 추세 계산 함수 (개선 버전) ---

def find_financial_statement_item(index, keywords, contains_mode=True, case_sensitive=False):
    """재무제표 인덱스에서 키워드를 포함하거나 일치하는 항목 이름을 찾습니다."""
    if not isinstance(index, pd.Index): return None

    # 정규식 패턴 생성 (공백/특수문자 무시, 대소문자 구분 옵션)
    pattern = r'\s*'.join(keywords) # 키워드 사이에 유연한 공백 허용
    flags = 0 if case_sensitive else re.IGNORECASE

    for item in index:
        if contains_mode:
            # 키워드가 순서대로 포함되는지 확인 (더 유연)
             try:
                  if re.search(pattern, item, flags=flags):
                       return item
             except TypeError: # 인덱스에 숫자가 섞여있을 경우 대비
                  continue
        else:
            # 키워드와 거의 일치하는지 확인 (덜 유연)
            cleaned_item = re.sub(r'\W+', '', item).lower() # 특수문자/공백 제거, 소문자화
            cleaned_pattern = re.sub(r'\W+', '', ''.join(keywords)).lower()
            if cleaned_item == cleaned_pattern:
                return item

    # 직접적인 키워드 매칭 실패 시, 첫 번째 키워드만으로 포함 검색 시도 (Fallback)
    first_keyword_pattern = keywords[0]
    for item in index:
         try:
              if re.search(first_keyword_pattern, item, flags=flags):
                  logging.warning(f"정확한 항목명 매칭 실패. '{keywords}' 대신 '{item}' 사용 시도.")
                  return item
         except TypeError:
              continue

    return None # 최종적으로 못 찾으면 None 반환


def get_operating_margin_trend(ticker, num_periods=4):
    """최근 분기별 영업이익률 추세를 계산하여 반환합니다 (딕셔너리 리스트)."""
    logging.info(f"{ticker}: 분기별 영업이익률 추세 가져오기 시도 (최근 {num_periods} 분기)...")
    try:
        stock = yf.Ticker(ticker)
        qf = stock.quarterly_financials # 분기별 손익계산서

        if qf.empty: logging.warning(f"{ticker}: 분기별 재무 데이터 없음."); return None

        # --- 개선된 컬럼명 탐색 로직 ---
        # 다양한 이름 형식에 대응 ('Total Revenue', 'Revenue', 'Total Operating Revenue' 등)
        revenue_col = find_financial_statement_item(qf.index, ['Total', 'Revenue']) or \
                      find_financial_statement_item(qf.index, ['Revenue'])
        # ('Operating Income', 'Operating Income Loss', 'OperatingIncome')
        op_income_col = find_financial_statement_item(qf.index, ['Operating', 'Income']) or \
                        find_financial_statement_item(qf.index, ['OperatingIncome'])
        # -------------------------------

        if not revenue_col or not op_income_col:
             logging.warning(f"{ticker}: 분기 재무 데이터에서 매출 또는 영업이익 항목 찾기 실패.")
             logging.debug(f"찾으려는 항목: Total Revenue({revenue_col}), Operating Income({op_income_col})")
             # logging.debug(f"사용 가능한 항목: {qf.index.tolist()}")
             return None

        # 데이터 선택 및 처리 (최신순 컬럼 가정)
        qf_recent = qf.iloc[:, :num_periods]
        df_trend = qf_recent.loc[[revenue_col, op_income_col]].T
        df_trend.index = pd.to_datetime(df_trend.index)
        df_trend = df_trend.sort_index(ascending=True)

        df_trend[revenue_col] = pd.to_numeric(df_trend[revenue_col], errors='coerce')
        df_trend[op_income_col] = pd.to_numeric(df_trend[op_income_col], errors='coerce')

        df_trend.replace(0, np.nan, inplace=True)
        df_trend.dropna(subset=[revenue_col, op_income_col], inplace=True)

        if df_trend.empty: logging.warning(f"{ticker}: 영업이익률 계산 가능 데이터 부족."); return None

        # 영업이익률 계산 (%)
        df_trend['Operating Margin (%)'] = (df_trend[op_income_col] / df_trend[revenue_col]) * 100
        df_trend['Operating Margin (%)'] = df_trend['Operating Margin (%)'].round(2)

        result_trend = df_trend[['Operating Margin (%)']].reset_index()
        result_trend.rename(columns={'index': 'Date'}, inplace=True)
        result_trend['Date'] = result_trend['Date'].dt.strftime('%Y-%m-%d')

        logging.info(f"{ticker}: 최근 {len(result_trend)}개 분기 영업이익률 추세 계산 완료.")
        return result_trend.to_dict('records')

    except Exception as e:
        logging.error(f"{ticker}: 영업이익률 추세 계산 중 오류: {e}")
        return None

# --- 기존 분석 및 시각화 함수들 ---

# plot_stock_chart, get_news_sentiment, run_prophet_forecast 함수는 이전과 동일하게 유지
# ... (이전 답변의 plot_stock_chart, get_news_sentiment, run_prophet_forecast 코드 붙여넣기) ...
def plot_stock_chart(ticker, start_date=None, end_date=None, period="1y"):
    """주가 데이터를 기반으로 차트 Figure 객체를 생성하여 반환합니다."""
    df = get_stock_data(ticker, start_date=start_date, end_date=end_date, period=period)
    if df is None or df.empty: logging.error(f"{ticker} 차트 생성 실패: 주가 데이터 없음"); return None
    try:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='가격'), row=1, col=1)
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='거래량', marker_color='rgba(0,0,100,0.6)'), row=2, col=1)
        fig.update_layout(title=f'{ticker} 주가 및 거래량 차트', yaxis_title='가격', yaxis2_title='거래량', xaxis_rangeslider_visible=False, hovermode='x unified', margin=dict(l=20, r=20, t=40, b=20))
        fig.update_yaxes(title_text="가격", row=1, col=1); fig.update_yaxes(title_text="거래량", row=2, col=1)
        logging.info(f"{ticker} 차트 생성 완료 (Figure 객체 반환)"); return fig
    except Exception as e: logging.error(f"{ticker} 차트 Figure 생성 중 오류: {e}"); return None

def get_news_sentiment(ticker, api_key):
    """지정된 종목에 대한 뉴스 기사의 감정을 분석합니다."""
    if not api_key: logging.warning("NEWS_API_KEY 없음."); return ["뉴스 API 키 미설정."]
    url = f"https://newsapi.org/v2/everything?q={ticker}&pageSize=20&language=en&sortBy=publishedAt&apiKey={api_key}"
    try:
        response = requests.get(url, timeout=10); response.raise_for_status()
        articles = response.json().get('articles', [])
        if not articles: return ["관련 뉴스 없음."]
        output_lines, total_polarity, analyzed_count = [], 0, 0
        for i, article in enumerate(articles, 1):
            title = article.get('title', 'N/A')
            text = article.get('description') or article.get('content') or title or ""
            if text and text != "[Removed]":
                try: blob = TextBlob(text); pol = blob.sentiment.polarity; output_lines.append(f"{i}. {title} | 감정: {pol:.2f}"); total_polarity += pol; analyzed_count += 1
                except Exception as text_err: output_lines.append(f"{i}. {title} | 감정 분석 오류")
            else: output_lines.append(f"{i}. {title} | 내용 없음")
        avg_pol = total_polarity / analyzed_count if analyzed_count > 0 else 0
        logging.info(f"{ticker} 뉴스 감정 분석 완료 (평균: {avg_pol:.2f})")
        output_lines.insert(0, f"총 {analyzed_count}개 뉴스 분석 | 평균 감성 점수: {avg_pol:.2f}"); return output_lines
    except requests.exceptions.RequestException as e: return [f"뉴스 API 요청 실패: {e}"]
    except Exception as e: logging.error(f"뉴스 분석 오류: {e}"); return ["뉴스 분석 중 오류 발생."]

def run_prophet_forecast(ticker, start_date, end_date=None, forecast_days=30, fred_key=None):
    """Prophet 예측, 교차 검증 수행 후 예측 결과(dict), 예측 차트(fig), CV차트 경로(str) 반환"""
    if end_date is None: end_date = datetime.today().strftime("%Y-%m-%d")
    df_stock = get_stock_data(ticker, start_date=start_date, end_date=end_date)
    if df_stock is None or df_stock.empty: return None, None, None
    df_stock = df_stock.reset_index()[["Date", "Close"]]; df_stock["Date"] = pd.to_datetime(df_stock["Date"])
    df_macro = get_macro_data(start_date=start_date, end_date=end_date, fred_key=fred_key)
    if not df_macro.empty:
        df_stock['Date'] = pd.to_datetime(df_stock['Date']); df_macro['Date'] = pd.to_datetime(df_macro['Date'])
        df_merged = pd.merge(df_stock, df_macro, on="Date", how="left"); logging.info(f"주가/매크로 데이터 병합 완료.")
        macro_cols = ["VIX", "US10Y", "US13W", "DXY", "FedFunds"]
        for col in macro_cols:
            if col in df_merged.columns:
                if not pd.api.types.is_numeric_dtype(df_merged[col]): df_merged[col] = pd.to_numeric(df_merged[col], errors='coerce')
                df_merged[col] = df_merged[col].ffill().bfill()
    else: logging.warning("매크로 데이터 없어 주가 데이터만 사용."); df_merged = df_stock
    if df_merged['Close'].isnull().any(): df_merged = df_merged.dropna(subset=['Close'])
    if df_merged.empty or len(df_merged) < 30: logging.error(f"Prophet 실패: 데이터 부족 ({len(df_merged)} 행)."); return None, None, None
    logging.info(f"Prophet 학습 데이터 준비 완료: {len(df_merged)} 행")
    os.makedirs(DATA_FOLDER, exist_ok=True); data_csv_path = os.path.join(DATA_FOLDER, f"{ticker}_merged_for_prophet.csv")
    try: df_merged.to_csv(data_csv_path, index=False); logging.info(f"학습 데이터 저장 완료: {data_csv_path}")
    except Exception as save_err: logging.error(f"학습 데이터 저장 실패: {save_err}")
    df_prophet = df_merged.rename(columns={"Date": "ds", "Close": "y"}); df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])
    m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False, changepoint_prior_scale=0.05)
    regressors = []
    if not df_macro.empty:
        for col in macro_cols:
             if col in df_prophet.columns and pd.api.types.is_numeric_dtype(df_prophet[col]) and df_prophet[col].isnull().sum() == 0:
                 m.add_regressor(col); regressors.append(col); logging.info(f"Regressor 추가: {col}")
             elif col in df_prophet.columns: logging.warning(f"Regressor '{col}' 타입/결측치 문제로 추가 안 함.")
    forecast_result_dict, fig_forecast, cv_plot_path = None, None, None
    try:
        m.fit(df_prophet[['ds', 'y'] + regressors]); logging.info("Prophet 모델 학습 완료.")
        os.makedirs(FORECAST_FOLDER, exist_ok=True)
        try: # Cross-validation block
            initial_days, period_days, horizon_days = '365 days', '90 days', f'{forecast_days} days'
            logging.info(f"Prophet CV 시작..."); df_cv = cross_validation(m, initial=initial_days, period=period_days, horizon=horizon_days, parallel=None)
            logging.info("Prophet CV 완료."); df_p = performance_metrics(df_cv); logging.info(f"Prophet 성능 지표:\n{df_p.head().to_string()}")
            fig_cv = plot_cross_validation_metric(df_cv, metric='mape'); plt.title(f'{ticker} Cross Validation MAPE'); cv_plot_path = os.path.join(FORECAST_FOLDER, f"{ticker}_cv_mape_plot.png")
            fig_cv.savefig(cv_plot_path); plt.close(fig_cv); logging.info(f"CV MAPE 차트 저장: {cv_plot_path}")
        except Exception as cv_err: logging.error(f"Prophet CV 오류: {cv_err}"); cv_plot_path = None
        logging.info("미래 예측 시작..."); future = m.make_future_dataframe(periods=forecast_days)
        if regressors: # Regressor future value handling
            temp_merged = df_merged.copy(); temp_merged['Date'] = pd.to_datetime(temp_merged['Date'])
            future = future.merge(temp_merged[['Date'] + regressors], left_on='ds', right_on='Date', how='left').drop(columns=['Date'])
            for col in regressors:
                if col in temp_merged.columns:
                    non_na = temp_merged[col].dropna(); last_val = non_na.iloc[-1] if not non_na.empty else 0
                    if non_na.empty: logging.warning(f"Regressor '{col}' 과거 값 모두 NaN.")
                    future[col] = future[col].ffill().fillna(last_val)
        forecast = m.predict(future); logging.info("미래 예측 완료.")
        csv_fn = os.path.join(FORECAST_FOLDER, f"{ticker}_forecast_data.csv")
        forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy().assign(ds=lambda df: df['ds'].dt.strftime('%Y-%m-%d')).to_csv(csv_fn, index=False)
        logging.info(f"예측 데이터 저장: {csv_fn}")
        fig_forecast = plot_plotly(m, forecast); fig_forecast.update_layout(title=f'{ticker} Price Forecast', margin=dict(l=20, r=20, t=40, b=20)); logging.info(f"예측 Figure 생성 완료.")
        forecast_result_dict = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(forecast_days).to_dict('records')
        for record in forecast_result_dict: record['ds'] = record['ds'].strftime('%Y-%m-%d')
        return forecast_result_dict, fig_forecast, cv_plot_path
    except Exception as e: logging.error(f"Prophet 학습/예측 오류: {e}"); logging.error(traceback.format_exc()); return None, None, None

# --- 메인 분석 함수 ---

# 7. 통합 분석 - num_trend_periods 인자 추가 및 전달
def analyze_stock(ticker, news_key, fred_key, analysis_period_years=2, forecast_days=30, num_trend_periods=4): # num_trend_periods 추가
    """모든 데이터를 종합하여 주식 분석 결과를 반환합니다."""
    logging.info(f"--- {ticker} 주식 분석 시작 ---")
    output_results = {}
    try:
        end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        start_date = end_date - relativedelta(years=analysis_period_years)
        start_date_str = start_date.strftime("%Y-%m-%d"); end_date_str = end_date.strftime("%Y-%m-%d")
        logging.info(f"분석 기간: {start_date_str} ~ {end_date_str}")
    except Exception as date_err: logging.error(f"날짜 설정 오류: {date_err}"); return None

    df_stock_full = get_stock_data(ticker, start_date=start_date_str, end_date=end_date_str)
    if df_stock_full is not None and not df_stock_full.empty:
        output_results['current_price'] = f"{df_stock_full['Close'].iloc[-1]:.2f}" if not df_stock_full['Close'].empty else "N/A"
        output_results['analysis_period_start'] = start_date_str; output_results['analysis_period_end'] = end_date_str
        output_results['data_points'] = len(df_stock_full)
    else:
        output_results['current_price'] = "N/A"; output_results['analysis_period_start'] = start_date_str
        output_results['analysis_period_end'] = end_date_str; output_results['data_points'] = 0
        logging.warning(f"{ticker} 주가 정보 가져오기 실패. 분석 제한됨.")

    stock_chart_fig = plot_stock_chart(ticker, start_date=start_date_str, end_date=end_date_str)
    output_results['stock_chart_fig'] = stock_chart_fig

    output_results['fundamentals'] = get_fundamental_data(ticker)
    # --- 영업이익률 추세 호출 시 num_periods 전달 ---
    output_results['operating_margin_trend'] = get_operating_margin_trend(ticker, num_periods=num_trend_periods)
    # -----------------------------------------
    # --- 🍀 ROE 추세 데이터 추가 ---
    output_results['roe_trend'] = get_roe_trend(ticker, num_periods=num_trend_periods)
    # -----------------------------

    output_results['news_sentiment'] = get_news_sentiment(ticker, news_key)
    fg_value, fg_class = get_fear_greed_index()
    output_results['fear_greed_index'] = {'value': fg_value, 'classification': fg_class} if fg_value is not None else "N/A"

    if output_results['data_points'] > 30 and output_results['current_price'] != "N/A":
        forecast_dict, forecast_fig, cv_plot_path = run_prophet_forecast(
            ticker, start_date=start_date_str, end_date=end_date_str,
            forecast_days=forecast_days, fred_key=fred_key)
        output_results['prophet_forecast'] = forecast_dict if forecast_dict is not None else "예측 실패"
        output_results['forecast_fig'] = forecast_fig; output_results['cv_plot_path'] = cv_plot_path
    else:
         forecast_msg = f"데이터 부족 ({output_results['data_points']}개)" if output_results['data_points'] <= 30 else "주가 정보 없음"
         output_results['prophet_forecast'] = f"{forecast_msg}으로 예측 불가"; output_results['forecast_fig'] = None
         output_results['cv_plot_path'] = None; logging.warning(f"{forecast_msg}으로 Prophet 예측 건너<0xEB>뜀.")

    logging.info(f"--- {ticker} 주식 분석 완료 ---")
    return output_results
elif key == 'roe_trend' and isinstance(value, list): print(f"- {key.replace('_', ' ').title()}: {len(value)} 분기 데이터"); [print(f"    - {item}") for item in value]




# --- 메인 실행 부분 (직접 실행 시 테스트용) ---
if __name__ == "__main__":
    # ... (이전과 동일, 테스트 출력 부분에 op margin 추가됨) ...
    print(f"stock_analysis.py 직접 실행 (테스트 목적, Base directory: {BASE_DIR}).")
    target_ticker = "AAPL"
    news_api_key_local = os.getenv("NEWS_API_KEY"); fred_api_key_local = os.getenv("FRED_API_KEY")
    if not news_api_key_local or not fred_api_key_local: print("경고: 로컬 테스트 API 키 없음."); test_results = None
    else: test_results = analyze_stock(ticker=target_ticker, news_key=news_api_key_local, fred_key=fred_api_key_local, analysis_period_years=1, forecast_days=15, num_trend_periods=5) # 테스트 시 5분기 요청
    print("\n--- 테스트 실행 결과 요약 ---")
    if test_results:
        for key, value in test_results.items():
             if 'fig' in key and value is not None: print(f"- {key.replace('_', ' ').title()}: Plotly Figure 객체 생성됨")
             elif key == 'fundamentals' and isinstance(value, dict): print(f"- {key.replace('_', ' ').title()}:"); [print(f"    - {f_key}: {f_value}") for f_key, f_value in value.items()]
             elif key == 'operating_margin_trend' and isinstance(value, list): print(f"- {key.replace('_', ' ').title()}: {len(value)} 분기 데이터"); [print(f"    - {item}") for item in value]
             elif key == 'prophet_forecast': print(f"- {key.replace('_', ' ').title()}: {type(value)}")
             elif key == 'news_sentiment': print(f"- {key.replace('_', ' ').title()}: {len(value) if isinstance(value, list) else 0} 항목")
             else: print(f"- {key.replace('_', ' ').title()}: {value}")
    else: print("테스트 분석 실패.")
    print("\n--- 테스트 실행 종료 ---")