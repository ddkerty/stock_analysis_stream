# stock_analysis.py (재무 항목 탐색 로직 개선 및 오류 수정)

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
import re

warnings.simplefilter(action='ignore', category=FutureWarning)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 경로 및 API 키 설정 ---
try: BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError: BASE_DIR = os.getcwd()
CHARTS_FOLDER = os.path.join(BASE_DIR, "charts"); DATA_FOLDER = os.path.join(BASE_DIR, "data"); FORECAST_FOLDER = os.path.join(BASE_DIR, "forecast")
try:
    dotenv_path = os.path.join(BASE_DIR, '.env')
    if os.path.exists(dotenv_path): load_dotenv(dotenv_path=dotenv_path); logging.info(f".env 로드 성공: {dotenv_path}")
    else: logging.warning(f".env 파일 없음: {dotenv_path}")
except Exception as e: logging.error(f".env 로드 오류: {e}")
NEWS_API_KEY = os.getenv("NEWS_API_KEY"); FRED_API_KEY = os.getenv("FRED_API_KEY")
if not NEWS_API_KEY: logging.warning("NEWS_API_KEY 없음.")
if not FRED_API_KEY: logging.warning("FRED_API_KEY 없음.")
if NEWS_API_KEY and FRED_API_KEY: logging.info("API 키 로드 시도 완료.")

# --- 데이터 가져오기 함수들 ---
# get_fear_greed_index, get_stock_data, get_macro_data 는 이전과 동일
def get_fear_greed_index():
    """공포-탐욕 지수를 API에서 가져옵니다."""
    url = "https://api.alternative.me/fng/?limit=1&format=json&date_format=world"
    try: response = requests.get(url, timeout=10); response.raise_for_status(); data = response.json()['data'][0]; value = int(data['value']); classification = data['value_classification']; logging.info(f"F&G 성공: {value} ({classification})"); return value, classification
    except Exception as e: logging.error(f"F&G 오류: {e}"); return None, None

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
            if not tmp.empty: tmp = tmp[['Close']].rename(columns={"Close": label}); tmp.index=pd.to_datetime(tmp.index).tz_localize(None); all_yf_data.append(tmp); logging.info(f"{label} 성공")
            else: logging.warning(f"{label} 비어있음.")
        except Exception as e: logging.error(f"{label} 실패: {e}")
    if all_yf_data:
        df_macro = pd.concat(all_yf_data, axis=1)
        if isinstance(df_macro.columns, pd.MultiIndex): df_macro.columns = df_macro.columns.get_level_values(-1)
    if fred_key:
        try:
            fred = Fred(api_key=fred_key); fedfunds = fred.get_series("FEDFUNDS", start_date=start_date, end_date=end_date).rename("FedFunds")
            fedfunds.index=pd.to_datetime(fedfunds.index).tz_localize(None)
            if not df_macro.empty: df_macro = df_macro.merge(fedfunds, left_index=True, right_index=True, how='outer')
            else: df_macro = pd.DataFrame(fedfunds)
            logging.info("FRED 병합/가져오기 성공")
        except Exception as e: logging.error(f"FRED 실패: {e}");
    else: logging.warning("FRED 키 없어 스킵.")
    if not df_macro.empty:
        if 'FedFunds' not in df_macro.columns: df_macro['FedFunds'] = pd.NA
        df_macro = df_macro.sort_index().ffill().bfill().reset_index().rename(columns={'index': 'Date'}); df_macro["Date"] = pd.to_datetime(df_macro["Date"])
        logging.info("매크로 처리 완료."); return df_macro
    else: logging.warning("매크로 가져오기 실패."); return pd.DataFrame()

# --- 기본적 분석 데이터 가져오기 함수들 ---

def format_market_cap(mc):
    """시가총액 숫자를 읽기 쉬운 문자열(T/B/M 단위)로 변환합니다."""
    if isinstance(mc, (int, float)) and mc > 0:
        if mc >= 1e12: return f"${mc / 1e12:.2f} T"
        elif mc >= 1e9: return f"${mc / 1e9:.2f} B"
        elif mc >= 1e6: return f"${mc / 1e6:.2f} M"
        else: return f"${mc:,.0f}"
    return "N/A"

# --- ⭐ 수정된 부분: find_financial_statement_item 함수 ---
def find_financial_statement_item(index, keywords, exact_match_keywords=None, case_sensitive=False):
    """
    재무제표 인덱스에서 키워드를 포함하거나 정확히 일치하는 항목 이름을 찾습니다.
    정확히 일치하는 키워드를 먼저 시도하고, 없으면 포함 키워드를 검색합니다.
    잘못된 Fallback 로직을 제거하여 안정성을 높였습니다.
    """
    if not isinstance(index, pd.Index): return None
    flags = 0 if case_sensitive else re.IGNORECASE

    # 1순위: 정확히 일치하는 키워드 시도 (있다면)
    if exact_match_keywords:
        for exact_key in exact_match_keywords:
            if exact_key in index:
                logging.debug(f"정확한 키워드 매칭 성공: '{exact_key}' for {keywords}")
                return exact_key

    # 2순위: 제공된 키워드들이 순서대로 모두 포함되는지 확인 (더 정교한 패턴)
    # 예: ['Total', 'Revenue'] -> "Total Revenue", "Total Operating Revenue" 등 매칭 시도
    # 단어 경계(\b)를 사용하여 부분 단어 매칭 방지 (예: 'debt'가 'indebtedness'에 매칭되는 것 방지)
    pattern = r'\b' + r'\b.*\b'.join(keywords) + r'\b' # 단어 경계 추가
    try:
        matches = [item for item in index if isinstance(item, str) and re.search(pattern, item, flags=flags)]
        if matches:
            # 여러 개 매칭 시 가장 짧은 것을 선택 (더 구체적일 가능성)
            best_match = min(matches, key=len)
            logging.debug(f"포함 키워드 매칭 성공: '{best_match}' for {keywords}")
            return best_match
    except Exception as e:
        logging.warning(f"항목명 검색 중 정규식 오류 발생 ({keywords}): {e}")

    # 3순위: (제거됨) - 이전의 불안정한 첫 단어 Fallback 로직 제거

    logging.warning(f"재무제표 항목명 찾기 실패: {keywords}")
    return None # 최종적으로 못 찾으면 None 반환
# --- 수정된 부분 끝 ---

def get_fundamental_data(ticker):
    """yfinance의 .info를 사용하여 주요 기본적 분석 지표를 가져옵니다."""
    logging.info(f"{ticker}: 기본 정보(.info) 가져오기 시도...")
    fundamentals = {key: "N/A" for key in ["시가총액", "PER", "EPS", "배당수익률", "베타", "업종", "산업", "요약"]}
    try:
        stock = yf.Ticker(ticker); info = stock.info
        if not info or info.get('regularMarketPrice') is None: logging.warning(f"'{ticker}' 유효 .info 없음."); return fundamentals
        fundamentals["시가총액"] = format_market_cap(info.get("marketCap"))
        fwd_pe = info.get('forwardPE'); trl_pe = info.get('trailingPE')
        if isinstance(fwd_pe, (int, float)): fundamentals["PER"] = f"{fwd_pe:.2f} (Fwd)" # 라벨 수정
        elif isinstance(trl_pe, (int, float)): fundamentals["PER"] = f"{trl_pe:.2f} (Trl)" # 라벨 수정
        eps_val = info.get('trailingEps'); fundamentals["EPS"] = f"{eps_val:.2f}" if isinstance(eps_val, (int, float)) else "N/A"
        div_yield = info.get('dividendYield'); fundamentals["배당수익률"] = f"{div_yield * 100:.2f}%" if isinstance(div_yield, (int, float)) and div_yield > 0 else "N/A"
        beta_val = info.get('beta'); fundamentals["베타"] = f"{beta_val:.2f}" if isinstance(beta_val, (int, float)) else "N/A"
        fundamentals["업종"] = info.get("sector", "N/A"); fundamentals["산업"] = info.get("industry", "N/A"); fundamentals["요약"] = info.get("longBusinessSummary", "N/A")
        logging.info(f"{ticker} 기본 정보 가져오기 성공."); return fundamentals
    except Exception as e: logging.error(f"{ticker} 기본 정보(.info) 실패: {e}"); return fundamentals

def get_operating_margin_trend(ticker, num_periods=4):
    """최근 분기별 영업이익률 추세를 계산하여 반환합니다."""
    logging.info(f"{ticker}: 영업이익률 추세 (최근 {num_periods}분기)...")
    try:
        stock = yf.Ticker(ticker); qf = stock.quarterly_financials
        if qf.empty: logging.warning(f"{ticker}: 분기 재무 없음."); return None
        # --- ⭐ 수정된 부분: 개선된 find_financial_statement_item 사용 ---
        revenue_col = find_financial_statement_item(qf.index, ['Total', 'Revenue'], exact_match_keywords=['Total Revenue', 'Revenue'])
        op_income_col = find_financial_statement_item(qf.index, ['Operating', 'Income'], exact_match_keywords=['Operating Income'])
        # --- 수정된 부분 끝 ---
        if not revenue_col or not op_income_col: logging.warning(f"{ticker}: 매출/영업이익 항목 못찾음."); return None
        qf_recent = qf.iloc[:, :num_periods]; df_trend = qf_recent.loc[[revenue_col, op_income_col]].T.sort_index()
        df_trend.index = pd.to_datetime(df_trend.index)
        df_trend[revenue_col] = pd.to_numeric(df_trend[revenue_col], errors='coerce'); df_trend[op_income_col] = pd.to_numeric(df_trend[op_income_col], errors='coerce')
        df_trend.replace(0, np.nan, inplace=True); df_trend.dropna(subset=[revenue_col, op_income_col], inplace=True)
        if df_trend.empty: logging.warning(f"{ticker}: 영업이익률 계산 데이터 부족."); return None
        df_trend['Operating Margin (%)'] = (df_trend[op_income_col] / df_trend[revenue_col]) * 100; df_trend['Operating Margin (%)'] = df_trend['Operating Margin (%)'].round(2)
        result = df_trend[['Operating Margin (%)']].reset_index().rename(columns={'index': 'Date'}); result['Date'] = result['Date'].dt.strftime('%Y-%m-%d')
        logging.info(f"{ticker}: {len(result)}개 분기 영업이익률 계산 완료."); return result.to_dict('records')
    except Exception as e: logging.error(f"{ticker}: 영업이익률 계산 오류: {e}"); return None

def get_roe_trend(ticker, num_periods=4):
    """최근 분기별 ROE(%) 추세를 계산하여 반환합니다."""
    logging.info(f"{ticker}: ROE 추세 (최근 {num_periods}분기)...")
    try:
        stock = yf.Ticker(ticker); qf = stock.quarterly_financials; qbs = stock.quarterly_balance_sheet
        if qf.empty or qbs.empty: logging.warning(f"{ticker}: 분기 재무/대차대조표 없음."); return None
        # --- ⭐ 수정된 부분: 개선된 find_financial_statement_item 사용 ---
        net_income_col = find_financial_statement_item(qf.index, ['Net', 'Income'], exact_match_keywords=['Net Income', 'Net Income Common Stockholders', 'Net Income From Continuing Operations'])
        equity_col = find_financial_statement_item(qbs.index, ['Stockholder', 'Equity'], exact_match_keywords=['Total Stockholder Equity', 'Stockholders Equity']) or \
                     find_financial_statement_item(qbs.index, ['Total', 'Equity']) # Fallback if specific not found
        # --- 수정된 부분 끝 ---
        if not net_income_col or not equity_col: logging.warning(f"{ticker}: 순이익/자본총계 항목 못찾음."); return None
        qf_recent = qf.loc[[net_income_col]].iloc[:, :num_periods].T; qbs_recent = qbs.loc[[equity_col]].iloc[:, :num_periods].T
        df_trend = pd.merge(qf_recent, qbs_recent, left_index=True, right_index=True, how='outer').sort_index()
        df_trend.index = pd.to_datetime(df_trend.index)
        df_trend[net_income_col] = pd.to_numeric(df_trend[net_income_col], errors='coerce'); df_trend[equity_col] = pd.to_numeric(df_trend[equity_col], errors='coerce')
        df_trend[equity_col] = df_trend[equity_col].apply(lambda x: x if pd.notna(x) and x > 0 else np.nan); df_trend.dropna(subset=[net_income_col, equity_col], inplace=True)
        if df_trend.empty: logging.warning(f"{ticker}: ROE 계산 데이터 부족."); return None
        df_trend['ROE (%)'] = (df_trend[net_income_col] / df_trend[equity_col]) * 100; df_trend['ROE (%)'] = df_trend['ROE (%)'].round(2)
        result = df_trend[['ROE (%)']].reset_index().rename(columns={'index': 'Date'}); result['Date'] = result['Date'].dt.strftime('%Y-%m-%d')
        logging.info(f"{ticker}: {len(result)}개 분기 ROE 계산 완료."); return result.to_dict('records')
    except Exception as e: logging.error(f"{ticker}: ROE 계산 오류: {e}"); return None

# --- 🍀 부채비율 추세 계산 함수 (개선된 find_financial_statement_item 사용) ---
def get_debt_to_equity_trend(ticker, num_periods=4):
    """최근 분기별 부채비율(D/E Ratio) 추세를 계산하여 반환합니다."""
    logging.info(f"{ticker}: 부채비율 추세 (최근 {num_periods}분기)...")
    try:
        stock = yf.Ticker(ticker); qbs = stock.quarterly_balance_sheet
        if qbs.empty: logging.warning(f"{ticker}: 분기 대차대조표 없음."); return None
        # --- ⭐ 수정된 부분: 개선된 find_financial_statement_item 사용 ---
        equity_col = find_financial_statement_item(qbs.index, ['Stockholder', 'Equity'], exact_match_keywords=['Total Stockholder Equity', 'Stockholders Equity']) or \
                     find_financial_statement_item(qbs.index, ['Total', 'Equity'])
        total_debt_col = find_financial_statement_item(qbs.index, ['Total', 'Debt'], exact_match_keywords=['Total Debt'])
        short_debt_col = find_financial_statement_item(qbs.index, ['Short', 'Long', 'Term', 'Debt'], exact_match_keywords=['Short Long Term Debt']) or \
                         find_financial_statement_item(qbs.index, ['Current', 'Debt'], exact_match_keywords=['Current Debt']) # 단기 부채
        long_debt_col = find_financial_statement_item(qbs.index, ['Long', 'Term', 'Debt'], exact_match_keywords=['Long Term Debt']) # 장기 부채
        # --- 수정된 부분 끝 ---

        if not equity_col: logging.warning(f"{ticker}: 자본총계 항목 못찾음."); return None
        required_cols = [equity_col]; use_total_debt = False; calculate_debt = False
        if total_debt_col: required_cols.append(total_debt_col); use_total_debt = True; logging.info(f"{ticker}: 'Total Debt' 항목 사용.")
        elif short_debt_col and long_debt_col: required_cols.extend([short_debt_col, long_debt_col]); calculate_debt = True; logging.info(f"{ticker}: 단기+장기 부채 합산 사용.")
        else: logging.warning(f"{ticker}: 총부채 관련 항목 못찾음."); return None

        required_cols = list(set(required_cols)); qbs_recent = qbs.loc[required_cols].iloc[:, :num_periods].T
        df_trend = qbs_recent.copy(); df_trend.index = pd.to_datetime(df_trend.index); df_trend = df_trend.sort_index(ascending=True)
        for col in required_cols: df_trend[col] = pd.to_numeric(df_trend[col], errors='coerce')
        df_trend[equity_col] = df_trend[equity_col].apply(lambda x: x if pd.notna(x) and x > 0 else np.nan)
        if use_total_debt: df_trend['Calculated Total Debt'] = df_trend[total_debt_col]
        elif calculate_debt: df_trend['Calculated Total Debt'] = df_trend[short_debt_col].fillna(0) + df_trend[long_debt_col].fillna(0)
        else: return None # Should not happen based on earlier check

        df_trend.dropna(subset=['Calculated Total Debt', equity_col], inplace=True)
        if df_trend.empty: logging.warning(f"{ticker}: 부채비율 계산 데이터 부족."); return None
        df_trend['D/E Ratio'] = df_trend['Calculated Total Debt'] / df_trend[equity_col]; df_trend['D/E Ratio'] = df_trend['D/E Ratio'].round(2)
        result = df_trend[['D/E Ratio']].reset_index().rename(columns={'index': 'Date'}); result['Date'] = result['Date'].dt.strftime('%Y-%m-%d')
        logging.info(f"{ticker}: {len(result)}개 분기 부채비율 계산 완료."); return result.to_dict('records')
    except Exception as e: logging.error(f"{ticker}: 부채비율 계산 오류: {e}"); return None

# --- 🍀 유동비율 추세 계산 함수 (개선된 find_financial_statement_item 사용) ---
def get_current_ratio_trend(ticker, num_periods=4):
    """최근 분기별 유동비율 추세를 계산하여 반환합니다."""
    logging.info(f"{ticker}: 유동비율 추세 (최근 {num_periods}분기)...")
    try:
        stock = yf.Ticker(ticker); qbs = stock.quarterly_balance_sheet
        if qbs.empty: logging.warning(f"{ticker}: 분기 대차대조표 없음."); return None
        # --- ⭐ 수정된 부분: 개선된 find_financial_statement_item 사용 ---
        current_assets_col = find_financial_statement_item(qbs.index, ['Total', 'Current', 'Assets'], exact_match_keywords=['Total Current Assets'])
        current_liab_col = find_financial_statement_item(qbs.index, ['Total', 'Current', 'Liabilities'], exact_match_keywords=['Total Current Liabilities'])
        # --- 수정된 부분 끝 ---

        # *** 여기! 수정된 부분: 필수 항목 체크 후 진행 ***
        if not current_assets_col or not current_liab_col:
            logging.warning(f"{ticker}: 유동자산('{current_assets_col}') 또는 유동부채('{current_liab_col}') 항목 찾기 실패하여 유동비율 계산 불가.")
            return None # 여기서 None 반환

        required_cols = [current_assets_col, current_liab_col]
        qbs_recent = qbs.loc[required_cols].iloc[:, :num_periods].T
        df_trend = qbs_recent.copy()
        df_trend.index = pd.to_datetime(df_trend.index); df_trend = df_trend.sort_index(ascending=True)
        df_trend[current_assets_col] = pd.to_numeric(df_trend[current_assets_col], errors='coerce')
        df_trend[current_liab_col] = pd.to_numeric(df_trend[current_liab_col], errors='coerce')
        df_trend[current_liab_col] = df_trend[current_liab_col].apply(lambda x: x if pd.notna(x) and x > 0 else np.nan)
        df_trend.dropna(subset=[current_assets_col, current_liab_col], inplace=True)
        if df_trend.empty: logging.warning(f"{ticker}: 유동비율 계산 가능 데이터 부족."); return None
        df_trend['Current Ratio'] = df_trend[current_assets_col] / df_trend[current_liab_col]; df_trend['Current Ratio'] = df_trend['Current Ratio'].round(2)
        result = df_trend[['Current Ratio']].reset_index().rename(columns={'index': 'Date'}); result['Date'] = result['Date'].dt.strftime('%Y-%m-%d')
        logging.info(f"{ticker}: {len(result)}개 분기 유동비율 계산 완료."); return result.to_dict('records')
    except Exception as e: logging.error(f"{ticker}: 유동비율 계산 오류: {e}"); return None


# --- 기존 분석 및 시각화 함수들 ---
# plot_stock_chart, get_news_sentiment, run_prophet_forecast 는 이전과 동일
def plot_stock_chart(ticker, start_date=None, end_date=None, period="1y"):
    # ... (이전 코드와 동일) ...
    df = get_stock_data(ticker, start_date=start_date, end_date=end_date, period=period)
    if df is None or df.empty: logging.error(f"{ticker} 차트 실패: 데이터 없음"); return None
    try: fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3]); fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='가격'), row=1, col=1); fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='거래량', marker_color='rgba(0,0,100,0.6)'), row=2, col=1); fig.update_layout(title=f'{ticker} 주가/거래량 차트', yaxis_title='가격', yaxis2_title='거래량', xaxis_rangeslider_visible=False, hovermode='x unified', margin=dict(l=20, r=20, t=40, b=20)); fig.update_yaxes(title_text="가격", row=1, col=1); fig.update_yaxes(title_text="거래량", row=2, col=1); logging.info(f"{ticker} 차트 생성 완료"); return fig
    except Exception as e: logging.error(f"{ticker} 차트 생성 오류: {e}"); return None

def get_news_sentiment(ticker, api_key):
    # ... (이전 코드와 동일) ...
    if not api_key: logging.warning("NEWS_API_KEY 없음."); return ["뉴스 API 키 미설정."]
    url = f"https://newsapi.org/v2/everything?q={ticker}&pageSize=20&language=en&sortBy=publishedAt&apiKey={api_key}"
    try: response = requests.get(url, timeout=10); response.raise_for_status(); articles = response.json().get('articles', []);
    if not articles: return ["관련 뉴스 없음."]; output, total_pol, count = [], 0, 0
    for i, article in enumerate(articles, 1): title = article.get('title', 'N/A'); text = article.get('description') or article.get('content') or title or "";
    if text and text != "[Removed]": try: blob = TextBlob(text); pol = blob.sentiment.polarity; output.append(f"{i}. {title} | 감정: {pol:.2f}"); total_pol += pol; count += 1; except: output.append(f"{i}. {title} | 감정 분석 오류")
    else: output.append(f"{i}. {title} | 내용 없음")
    avg_pol = total_pol / count if count > 0 else 0; logging.info(f"{ticker} 뉴스 분석 완료 (평균: {avg_pol:.2f})"); output.insert(0, f"총 {count}개 분석 | 평균 감성: {avg_pol:.2f}"); return output
    except requests.exceptions.RequestException as e: return [f"뉴스 API 요청 실패: {e}"]
    except Exception as e: logging.error(f"뉴스 분석 오류: {e}"); return ["뉴스 분석 중 오류 발생."]

def run_prophet_forecast(ticker, start_date, end_date=None, forecast_days=30, fred_key=None):
    # ... (이전 코드와 동일) ...
    if end_date is None: end_date = datetime.today().strftime("%Y-%m-%d"); df_stock = get_stock_data(ticker, start_date=start_date, end_date=end_date)
    if df_stock is None or df_stock.empty: return None, None, None
    df_stock = df_stock.reset_index()[["Date", "Close"]]; df_stock["Date"] = pd.to_datetime(df_stock["Date"]); df_macro = get_macro_data(start_date=start_date, end_date=end_date, fred_key=fred_key)
    if not df_macro.empty: df_stock['Date'] = pd.to_datetime(df_stock['Date']); df_macro['Date'] = pd.to_datetime(df_macro['Date']); df_merged = pd.merge(df_stock, df_macro, on="Date", how="left"); logging.info(f"주가/매크로 병합 완료."); macro_cols = ["VIX", "US10Y", "US13W", "DXY", "FedFunds"];
    for col in macro_cols:
        if col in df_merged.columns:
            if not pd.api.types.is_numeric_dtype(df_merged[col]): df_merged[col] = pd.to_numeric(df_merged[col], errors='coerce')
            df_merged[col] = df_merged[col].ffill().bfill()
    else: logging.warning("매크로 데이터 없어 주가만 사용."); df_merged = df_stock
    if df_merged['Close'].isnull().any(): df_merged = df_merged.dropna(subset=['Close'])
    if df_merged.empty or len(df_merged) < 30: logging.error(f"Prophet 실패: 데이터 부족."); return None, None, None
    logging.info(f"Prophet 학습 데이터 준비: {len(df_merged)} 행"); os.makedirs(DATA_FOLDER, exist_ok=True); data_csv_path = os.path.join(DATA_FOLDER, f"{ticker}_merged_for_prophet.csv")
    try: df_merged.to_csv(data_csv_path, index=False); logging.info(f"학습 데이터 저장: {data_csv_path}")
    except Exception as e: logging.error(f"학습 데이터 저장 실패: {e}")
    df_prophet = df_merged.rename(columns={"Date": "ds", "Close": "y"}); df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])
    m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False, changepoint_prior_scale=0.05); regressors = []
    if not df_macro.empty:
        for col in macro_cols:
             if col in df_prophet.columns and pd.api.types.is_numeric_dtype(df_prophet[col]) and df_prophet[col].isnull().sum() == 0: m.add_regressor(col); regressors.append(col); logging.info(f"Regressor 추가: {col}")
             elif col in df_prophet.columns: logging.warning(f"Regressor '{col}' 문제로 추가 안 함.")
    forecast_dict, fig_fcst, cv_path = None, None, None
    try:
        m.fit(df_prophet[['ds', 'y'] + regressors]); logging.info("Prophet 학습 완료."); os.makedirs(FORECAST_FOLDER, exist_ok=True)
        try: init, period, horizon = '365 days', '90 days', f'{forecast_days} days'; logging.info(f"Prophet CV 시작..."); df_cv = cross_validation(m, initial=init, period=period, horizon=horizon, parallel=None); logging.info("Prophet CV 완료."); df_p = performance_metrics(df_cv); logging.info(f"Prophet 성능:\n{df_p.head().to_string()}"); fig_cv = plot_cross_validation_metric(df_cv, metric='mape'); plt.title(f'{ticker} CV MAPE'); cv_path = os.path.join(FORECAST_FOLDER, f"{ticker}_cv_mape_plot.png"); fig_cv.savefig(cv_path); plt.close(fig_cv); logging.info(f"CV MAPE 차트 저장: {cv_path}")
        except Exception as cv_e: logging.error(f"Prophet CV 오류: {cv_e}"); cv_path = None
        logging.info("미래 예측 시작..."); future = m.make_future_dataframe(periods=forecast_days)
        if regressors: temp_m = df_merged.copy(); temp_m['Date'] = pd.to_datetime(temp_m['Date']); future = future.merge(temp_m[['Date'] + regressors], left_on='ds', right_on='Date', how='left').drop(columns=['Date'])
        for col in regressors:
            if col in temp_m.columns: non_na = temp_m[col].dropna(); last_val = non_na.iloc[-1] if not non_na.empty else 0; future[col] = future[col].ffill().fillna(last_val)
        forecast = m.predict(future); logging.info("미래 예측 완료.")
        csv_fn = os.path.join(FORECAST_FOLDER, f"{ticker}_forecast_data.csv"); forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy().assign(ds=lambda dfx: dfx['ds'].dt.strftime('%Y-%m-%d')).to_csv(csv_fn, index=False); logging.info(f"예측 데이터 저장: {csv_fn}")
        fig_fcst = plot_plotly(m, forecast); fig_fcst.update_layout(title=f'{ticker} Price Forecast', margin=dict(l=20,r=20,t=40,b=20)); logging.info(f"예측 Figure 생성.")
        forecast_dict = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(forecast_days).to_dict('records'); [rec.update({'ds': rec['ds'].strftime('%Y-%m-%d')}) for rec in forecast_dict]
        return forecast_dict, fig_fcst, cv_path
    except Exception as e: logging.error(f"Prophet 학습/예측 오류: {e}"); logging.error(traceback.format_exc()); return None, None, None


# --- 메인 분석 함수 ---

# 7. 통합 분석 - 신규 재무 추세 결과 추가
def analyze_stock(ticker, news_key, fred_key, analysis_period_years=2, forecast_days=30, num_trend_periods=4):
    """모든 데이터를 종합하여 주식 분석 결과를 반환합니다."""
    logging.info(f"--- {ticker} 주식 분석 시작 ---"); output_results = {}
    try: end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0); start_date = end_date - relativedelta(years=analysis_period_years); start_date_str = start_date.strftime("%Y-%m-%d"); end_date_str = end_date.strftime("%Y-%m-%d"); logging.info(f"분석 기간: {start_date_str} ~ {end_date_str}")
    except Exception as e: logging.error(f"날짜 설정 오류: {e}"); return None
    df_stock_full = get_stock_data(ticker, start_date=start_date_str, end_date=end_date_str)
    if df_stock_full is not None and not df_stock_full.empty: output_results['current_price'] = f"{df_stock_full['Close'].iloc[-1]:.2f}" if not df_stock_full['Close'].empty else "N/A"; output_results['analysis_period_start'] = start_date_str; output_results['analysis_period_end'] = end_date_str; output_results['data_points'] = len(df_stock_full)
    else: output_results['current_price'] = "N/A"; output_results['analysis_period_start'] = start_date_str; output_results['analysis_period_end'] = end_date_str; output_results['data_points'] = 0; logging.warning(f"{ticker} 주가 정보 실패.")
    output_results['stock_chart_fig'] = plot_stock_chart(ticker, start_date=start_date_str, end_date=end_date_str)
    output_results['fundamentals'] = get_fundamental_data(ticker)
    output_results['operating_margin_trend'] = get_operating_margin_trend(ticker, num_periods=num_trend_periods)
    output_results['roe_trend'] = get_roe_trend(ticker, num_periods=num_trend_periods)
    output_results['debt_to_equity_trend'] = get_debt_to_equity_trend(ticker, num_periods=num_trend_periods) # 호출 추가
    output_results['current_ratio_trend'] = get_current_ratio_trend(ticker, num_periods=num_trend_periods) # 호출 추가
    output_results['news_sentiment'] = get_news_sentiment(ticker, news_key)
    fg_value, fg_class = get_fear_greed_index(); output_results['fear_greed_index'] = {'value': fg_value, 'classification': fg_class} if fg_value is not None else "N/A"
    if output_results['data_points'] > 30 and output_results['current_price'] != "N/A":
        forecast_dict, forecast_fig, cv_plot_path = run_prophet_forecast(ticker, start_date=start_date_str, end_date=end_date_str, forecast_days=forecast_days, fred_key=fred_key)
        output_results['prophet_forecast'] = forecast_dict if forecast_dict is not None else "예측 실패"; output_results['forecast_fig'] = forecast_fig; output_results['cv_plot_path'] = cv_plot_path
    else: msg = f"데이터 부족 ({output_results['data_points']})" if output_results['data_points'] <= 30 else "주가 정보 없음"; output_results['prophet_forecast'] = f"{msg}으로 예측 불가"; output_results['forecast_fig'] = None; output_results['cv_plot_path'] = None; logging.warning(f"{msg} Prophet 예측 건너뜀.")
    logging.info(f"--- {ticker} 주식 분석 완료 ---"); return output_results

# --- 메인 실행 부분 (테스트용) ---
if __name__ == "__main__":
    # ... (이전과 동일, 테스트 출력 부분에 D/E, Current Ratio 추가됨) ...
    print(f"stock_analysis.py 직접 실행 (테스트 목적, Base directory: {BASE_DIR}).")
    target_ticker = "MSFT" # 다른 티커로 테스트
    news_key = os.getenv("NEWS_API_KEY"); fred_key = os.getenv("FRED_API_KEY")
    if not news_key or not fred_key: print("경고: API 키 없음."); test_results = None
    else: test_results = analyze_stock(ticker=target_ticker, news_key=news_key, fred_key=fred_key, analysis_period_years=1, forecast_days=15, num_trend_periods=5)
    print("\n--- 테스트 실행 결과 요약 ---")
    if test_results:
        for key, value in test_results.items():
            if 'fig' in key and value is not None: print(f"- {key.replace('_',' ').title()}: Plotly Figure 생성됨")
            elif key == 'fundamentals' and isinstance(value, dict): print(f"- Fundamentals:"); [print(f"    - {k}: {v}") for k, v in value.items()]
            elif '_trend' in key and isinstance(value, list): print(f"- {key.replace('_',' ').title()}: {len(value)} 분기"); [print(f"    - {item}") for item in value]
            elif key == 'prophet_forecast': print(f"- Prophet Forecast: {type(value)}")
            elif key == 'news_sentiment': print(f"- News Sentiment: {len(value) if isinstance(value, list) else 0} 항목")
            else: print(f"- {key.replace('_',' ').title()}: {value}")
    else: print("테스트 분석 실패.")
    print("\n--- 테스트 실행 종료 ---")