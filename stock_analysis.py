# stock_analysis.py (오류 수정 최종 검토 버전)

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

# 경고 메시지 및 로깅 설정
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

def get_fear_greed_index():
    """공포-탐욕 지수 가져오기 (안정성 개선)"""
    url = "https://api.alternative.me/fng/?limit=1&format=json&date_format=world"
    try: response = requests.get(url, timeout=10); response.raise_for_status(); data = response.json().get('data', [])
    if data and isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
        value_str = data[0].get('value'); classification = data[0].get('value_classification')
        if value_str is not None and classification is not None:
            try: value = int(value_str); logging.info(f"F&G 성공: {value} ({classification})"); return value, classification
            except (ValueError, TypeError): logging.warning(f"F&G 값 변환 오류: {value_str}"); return None, None
        else: logging.warning("F&G 데이터 구조 오류."); return None, None
    else: logging.warning("F&G 데이터 비어있거나 형식 오류."); return None, None
    except requests.exceptions.RequestException as e: logging.error(f"F&G API 요청 오류: {e}"); return None, None
    except Exception as e: logging.error(f"F&G 처리 오류: {e}"); return None, None

def get_stock_data(ticker, start_date=None, end_date=None, period="1y"):
    """주가 데이터 가져오기 (NaN 처리 개선)"""
    try:
        stock = yf.Ticker(ticker)
        if start_date and end_date: data = stock.history(start=start_date, end=end_date, auto_adjust=False)
        else: data = stock.history(period=period, auto_adjust=False)
        logging.info(f"{ticker} 주가 가져오기 시도 완료.")
        if data.empty: logging.warning(f"{ticker} 주가 데이터 비어있음."); return None
        if isinstance(data.index, pd.DatetimeIndex): data.index = data.index.tz_localize(None)
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols: logging.warning(f"{ticker}: 누락 컬럼 {missing_cols} -> NaN으로 채움.");
        for col in missing_cols: data[col] = np.nan
        for col in required_cols: data[col] = pd.to_numeric(data[col], errors='coerce')
        return data
    except Exception as e: logging.error(f"티커 '{ticker}' 주가 실패: {e}"); return None

def get_macro_data(start_date, end_date=None, fred_key=None):
    """매크로 지표 데이터 가져오기 (빈 DF 컬럼 명시)"""
    if end_date is None: end_date = datetime.today().strftime("%Y-%m-%d")
    yf_tickers = {"^VIX": "VIX", "^TNX": "US10Y", "^IRX": "US13W", "DX-Y.NYB": "DXY"}; fred_series = {"FEDFUNDS": "FedFunds"}
    expected_cols = ['Date'] + list(yf_tickers.values()) + list(fred_series.values()); df_macro = pd.DataFrame(); all_yf_data = []
    for tk, label in yf_tickers.items():
        try: tmp = yf.download(tk, start=start_date, end=end_date, progress=False, timeout=15);
        if not tmp.empty: tmp = tmp[['Close']].rename(columns={"Close": label}); tmp.index=pd.to_datetime(tmp.index).tz_localize(None); all_yf_data.append(tmp); logging.info(f"{label} 성공")
        else: logging.warning(f"{label} 비어있음.")
        except Exception as e: logging.error(f"{label} 실패: {e}")
    if all_yf_data:
        df_macro = pd.concat(all_yf_data, axis=1)
        if isinstance(df_macro.columns, pd.MultiIndex): df_macro.columns = df_macro.columns.get_level_values(-1)
    if fred_key:
        try: fred = Fred(api_key=fred_key); fred_data = [];
        for series_id, label in fred_series.items(): s = fred.get_series(series_id, start_date=start_date, end_date=end_date).rename(label); s.index = pd.to_datetime(s.index).tz_localize(None); fred_data.append(s)
        if fred_data: df_fred = pd.concat(fred_data, axis=1);
        if not df_macro.empty: df_macro = df_macro.merge(df_fred, left_index=True, right_index=True, how='outer')
        else: df_macro = df_fred; logging.info("FRED 병합/가져오기 성공")
        except Exception as e: logging.error(f"FRED 실패: {e}");
    else: logging.warning("FRED 키 없어 스킵.")
    if not df_macro.empty:
        for col in expected_cols:
            if col != 'Date' and col not in df_macro.columns: df_macro[col] = pd.NA; logging.warning(f"매크로 '{col}' 없어 NaN 추가.")
        for col in df_macro.columns:
            if col != 'Date': df_macro[col] = pd.to_numeric(df_macro[col], errors='coerce')
        df_macro = df_macro.sort_index().ffill().bfill(); df_macro = df_macro.reset_index().rename(columns={'index': 'Date'}); df_macro["Date"] = pd.to_datetime(df_macro["Date"]); logging.info("매크로 처리 완료."); return df_macro[expected_cols]
    else: logging.warning("매크로 가져오기 최종 실패."); return pd.DataFrame(columns=expected_cols)

# --- 기본적 분석 데이터 가져오기 함수들 ---

def format_market_cap(mc):
    """시가총액 숫자 포맷팅"""
    if isinstance(mc, (int, float)) and mc > 0:
        if mc >= 1e12: return f"${mc / 1e12:.2f} T"
        elif mc >= 1e9: return f"${mc / 1e9:.2f} B"
        elif mc >= 1e6: return f"${mc / 1e6:.2f} M"
        else: return f"${mc:,.0f}"
    return "N/A"

# --- ⭐ 수정된 부분: find_financial_statement_item 함수 (오류 처리/로직 개선) ---
def find_financial_statement_item(index, keywords, exact_match_keywords=None, case_sensitive=False):
    """재무제표 인덱스에서 항목 이름 찾기 (안정성 강화)"""
    if not isinstance(index, pd.Index): return None
    flags = 0 if case_sensitive else re.IGNORECASE # flags 정의 위치 확인

    # 1순위: 정확히 일치하는 이름 시도
    if exact_match_keywords:
        for exact_key in exact_match_keywords:
            if exact_key in index:
                logging.debug(f"정확 키워드 매칭: '{exact_key}' for {keywords}")
                return exact_key

    # 2순위: 키워드 단어 경계 포함 매칭
    # 각 키워드를 re.escape 처리하여 특수 문자 문제 방지
    pattern_keywords = [re.escape(k) for k in keywords]
    pattern = r'\b' + r'\b.*\b'.join(pattern_keywords) + r'\b'
    matches = []
    for item in index:
        if isinstance(item, str): # 문자열 인덱스만 처리
            try:
                if re.search(pattern, item, flags=flags):
                    matches.append(item)
            except Exception as e:
                # 정규식 오류 발생 시에도 함수가 중단되지 않도록 처리
                logging.warning(f"항목명 검색 중 정규식 오류 발생 ('{item}', keywords: {keywords}): {e}")
                continue # 다음 항목 검색 계속

    if matches:
        best_match = min(matches, key=len)
        logging.debug(f"포함 키워드 매칭: '{best_match}' for {keywords}")
        return best_match

    # Fallback 로직 제거됨 (잘못된 매칭 방지)
    logging.warning(f"재무 항목 찾기 최종 실패: {keywords}")
    return None
# --- 수정된 부분 끝 ---

def get_fundamental_data(ticker):
    """yfinance .info 사용하여 주요 기본 지표 가져오기."""
    # ... (이전 최종 버전과 동일) ...
    logging.info(f"{ticker}: .info 가져오기..."); fundamentals = {key: "N/A" for key in ["시가총액", "PER", "EPS", "배당수익률", "베타", "업종", "산업", "요약"]}
    try: stock = yf.Ticker(ticker); info = stock.info;
    if not info or info.get('regularMarketPrice') is None: logging.warning(f"'{ticker}' 유효 .info 없음."); return fundamentals
    fundamentals["시가총액"] = format_market_cap(info.get("marketCap"))
    fwd_pe = info.get('forwardPE'); trl_pe = info.get('trailingPE')
    if isinstance(fwd_pe, (int, float)): fundamentals["PER"] = f"{fwd_pe:.2f} (Fwd)"
    elif isinstance(trl_pe, (int, float)): fundamentals["PER"] = f"{trl_pe:.2f} (Trl)"
    eps_val = info.get('trailingEps'); fundamentals["EPS"] = f"{eps_val:.2f}" if isinstance(eps_val, (int, float)) else "N/A"
    div_yield = info.get('dividendYield'); fundamentals["배당수익률"] = f"{div_yield * 100:.2f}%" if isinstance(div_yield, (int, float)) and div_yield > 0 else "N/A"
    beta_val = info.get('beta'); fundamentals["베타"] = f"{beta_val:.2f}" if isinstance(beta_val, (int, float)) else "N/A"
    fundamentals["업종"] = info.get("sector", "N/A"); fundamentals["산업"] = info.get("industry", "N/A"); fundamentals["요약"] = info.get("longBusinessSummary", "N/A")
    logging.info(f"{ticker} .info 성공."); return fundamentals
    except Exception as e: logging.error(f"{ticker} .info 실패: {e}"); return fundamentals

def get_operating_margin_trend(ticker, num_periods=4):
    """최근 분기별 영업이익률 추세 계산."""
    # ... (이전 최종 버전과 동일 - 개선된 find_financial_statement_item 사용) ...
    logging.info(f"{ticker}: 영업이익률 추세 ({num_periods}분기)..."); try: stock = yf.Ticker(ticker); qf = stock.quarterly_financials;
    if qf.empty: logging.warning(f"{ticker}: 분기 재무 없음."); return None
    revenue_col = find_financial_statement_item(qf.index, ['Total', 'Revenue'], exact_match_keywords=['Total Revenue', 'Revenue'])
    op_income_col = find_financial_statement_item(qf.index, ['Operating', 'Income'], exact_match_keywords=['Operating Income', 'Operating Income Loss']) # 손실 가능성 이름 추가
    if not revenue_col or not op_income_col: logging.warning(f"{ticker}: 매출/영업이익 항목 못찾음."); return None
    qf_recent = qf.iloc[:, :num_periods]; df = qf_recent.loc[[revenue_col, op_income_col]].T.sort_index(); df.index = pd.to_datetime(df.index)
    df[revenue_col] = pd.to_numeric(df[revenue_col], errors='coerce'); df[op_income_col] = pd.to_numeric(df[op_income_col], errors='coerce')
    df[revenue_col] = df[revenue_col].replace(0, np.nan) # 매출 0은 비정상으로 간주
    df.dropna(subset=[revenue_col, op_income_col], inplace=True);
    if df.empty: logging.warning(f"{ticker}: 영업이익률 계산 데이터 부족."); return None
    df['Operating Margin (%)'] = (df[op_income_col] / df[revenue_col]) * 100; df['Operating Margin (%)'] = df['Operating Margin (%)'].round(2)
    res = df[['Operating Margin (%)']].reset_index().rename(columns={'index':'Date'}); res['Date'] = res['Date'].dt.strftime('%Y-%m-%d'); logging.info(f"{ticker}: {len(res)}개 영업이익률 계산 완료."); return res.to_dict('records')
    except Exception as e: logging.error(f"{ticker}: 영업이익률 계산 오류: {e}"); return None

def get_roe_trend(ticker, num_periods=4):
    """최근 분기별 ROE(%) 추세 계산."""
    # ... (이전 최종 버전과 동일 - 개선된 find_financial_statement_item 사용) ...
    logging.info(f"{ticker}: ROE 추세 ({num_periods}분기)..."); try: stock = yf.Ticker(ticker); qf = stock.quarterly_financials; qbs = stock.quarterly_balance_sheet;
    if qf.empty or qbs.empty: logging.warning(f"{ticker}: 분기 재무/대차대조표 없음."); return None
    ni_col = find_financial_statement_item(qf.index, ['Net', 'Income'], ['Net Income', 'Net Income Common Stockholders', 'Net Income From Continuing Operations'])
    eq_col = find_financial_statement_item(qbs.index, ['Stockholder', 'Equity'], ['Total Stockholder Equity', 'Stockholders Equity']) or find_financial_statement_item(qbs.index, ['Total', 'Equity'])
    if not ni_col or not eq_col: logging.warning(f"{ticker}: 순이익/자본 항목 못찾음."); return None
    qf_r = qf.loc[[ni_col]].iloc[:, :num_periods].T; qbs_r = qbs.loc[[eq_col]].iloc[:, :num_periods].T; df = pd.merge(qf_r, qbs_r, left_index=True, right_index=True, how='outer').sort_index(); df.index = pd.to_datetime(df.index)
    df[ni_col] = pd.to_numeric(df[ni_col], errors='coerce'); df[eq_col] = pd.to_numeric(df[eq_col], errors='coerce'); df[eq_col] = df[eq_col].apply(lambda x: x if pd.notna(x) and x > 0 else np.nan); df.dropna(subset=[ni_col, eq_col], inplace=True);
    if df.empty: logging.warning(f"{ticker}: ROE 계산 데이터 부족."); return None
    df['ROE (%)'] = (df[ni_col] / df[eq_col]) * 100; df['ROE (%)'] = df['ROE (%)'].round(2)
    res = df[['ROE (%)']].reset_index().rename(columns={'index':'Date'}); res['Date'] = res['Date'].dt.strftime('%Y-%m-%d'); logging.info(f"{ticker}: {len(res)}개 ROE 계산 완료."); return res.to_dict('records')
    except Exception as e: logging.error(f"{ticker}: ROE 계산 오류: {e}"); return None

def get_debt_to_equity_trend(ticker, num_periods=4):
    """최근 분기별 부채비율(D/E Ratio) 추세 계산."""
    # ... (이전 최종 버전과 동일 - 개선된 find_financial_statement_item 사용) ...
    logging.info(f"{ticker}: 부채비율 추세 ({num_periods}분기)..."); try: stock = yf.Ticker(ticker); qbs = stock.quarterly_balance_sheet;
    if qbs.empty: logging.warning(f"{ticker}: 분기 대차대조표 없음."); return None
    eq_col = find_financial_statement_item(qbs.index, ['Stockholder', 'Equity'], ['Total Stockholder Equity', 'Stockholders Equity']) or find_financial_statement_item(qbs.index, ['Total', 'Equity'])
    if not eq_col: logging.warning(f"{ticker}: 자본 항목 못찾음."); return None
    td_col = find_financial_statement_item(qbs.index, ['Total', 'Debt']); sd_col = find_financial_statement_item(qbs.index, ['Current', 'Debt']); ld_col = find_financial_statement_item(qbs.index, ['Long', 'Term', 'Debt'])
    req_cols = [eq_col]; use_td = False; calc_d = False
    if td_col: req_cols.append(td_col); use_td = True; logging.info(f"{ticker}: Total Debt 사용.")
    elif sd_col and ld_col: req_cols.extend([sd_col, ld_col]); calc_d = True; logging.info(f"{ticker}: 단기+장기 부채 합산.")
    else: logging.warning(f"{ticker}: 총부채 관련 항목 못찾음."); return None
    req_cols = list(set(req_cols)); qbs_r = qbs.loc[req_cols].iloc[:, :num_periods].T; df = qbs_r.copy(); df.index = pd.to_datetime(df.index); df = df.sort_index()
    for col in req_cols: df[col] = pd.to_numeric(df[col], errors='coerce')
    df[eq_col] = df[eq_col].apply(lambda x: x if pd.notna(x) and x != 0 else np.nan) # 0도 NaN 처리
    if use_td: df['Calc Debt'] = df[td_col]
    elif calc_d: df['Calc Debt'] = df[sd_col].fillna(0) + df[ld_col].fillna(0)
    else: return None
    df.dropna(subset=['Calc Debt', eq_col], inplace=True)
    if df.empty: logging.warning(f"{ticker}: 부채비율 계산 데이터 부족."); return None
    df['D/E Ratio'] = df['Calc Debt'] / df[eq_col]; df['D/E Ratio'] = df['D/E Ratio'].round(2)
    res = df[['D/E Ratio']].reset_index().rename(columns={'index':'Date'}); res['Date'] = res['Date'].dt.strftime('%Y-%m-%d'); logging.info(f"{ticker}: {len(res)}개 부채비율 계산 완료."); return res.to_dict('records')
    except Exception as e: logging.error(f"{ticker}: 부채비율 계산 오류: {e}"); return None

def get_current_ratio_trend(ticker, num_periods=4):
    """최근 분기별 유동비율 추세 계산."""
    # ... (이전 최종 버전과 동일 - 개선된 find_financial_statement_item 사용 및 오류 수정 확인) ...
    logging.info(f"{ticker}: 유동비율 추세 ({num_periods}분기)..."); try: stock = yf.Ticker(ticker); qbs = stock.quarterly_balance_sheet;
    if qbs.empty: logging.warning(f"{ticker}: 분기 대차대조표 없음."); return None
    ca_col = find_financial_statement_item(qbs.index, ['Total', 'Current', 'Assets']); cl_col = find_financial_statement_item(qbs.index, ['Total', 'Current', 'Liabilities'])
    # --- ⭐ 수정된 부분: 필수 항목 체크 강화 ---
    if not ca_col or not cl_col: logging.warning(f"{ticker}: 유동자산/부채 항목 못찾음."); return None
    if ca_col == cl_col: logging.error(f"{ticker}: 유동자산/부채 항목 동일 식별('{ca_col}')."); return None
    # --- 수정된 부분 끝 ---
    req_cols = [ca_col, cl_col]; qbs_r = qbs.loc[req_cols].iloc[:, :num_periods].T; df = qbs_r.copy(); df.index = pd.to_datetime(df.index); df = df.sort_index()
    df[ca_col] = pd.to_numeric(df[ca_col], errors='coerce'); df[cl_col] = pd.to_numeric(df[cl_col], errors='coerce')
    df[cl_col] = df[cl_col].apply(lambda x: x if pd.notna(x) and x > 0 else np.nan); df.dropna(subset=[ca_col, cl_col], inplace=True)
    if df.empty: logging.warning(f"{ticker}: 유동비율 계산 데이터 부족."); return None
    df['Current Ratio'] = df[ca_col] / df[cl_col]; df['Current Ratio'] = df['Current Ratio'].round(2)
    res = df[['Current Ratio']].reset_index().rename(columns={'index':'Date'}); res['Date'] = res['Date'].dt.strftime('%Y-%m-%d'); logging.info(f"{ticker}: {len(res)}개 유동비율 계산 완료."); return res.to_dict('records')
    except Exception as e: logging.error(f"{ticker}: 유동비율 계산 오류: {e}"); return None

# --- 분석 및 시각화 함수들 ---
def plot_stock_chart(ticker, start_date=None, end_date=None, period="1y"):
    # ... (이전과 동일) ...
    df = get_stock_data(ticker, start_date=start_date, end_date=end_date, period=period);
    if df is None or df.empty: logging.error(f"{ticker} 차트 실패: 데이터 없음"); return None
    try: fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3]); fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='가격'), row=1, col=1); fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='거래량', marker_color='rgba(0,0,100,0.6)'), row=2, col=1); fig.update_layout(title=f'{ticker} 주가/거래량 차트', yaxis_title='가격', yaxis2_title='거래량', xaxis_rangeslider_visible=False, hovermode='x unified', margin=dict(l=20, r=20, t=40, b=20)); fig.update_yaxes(title_text="가격", row=1, col=1); fig.update_yaxes(title_text="거래량", row=2, col=1); logging.info(f"{ticker} 차트 생성 완료"); return fig
    except Exception as e: logging.error(f"{ticker} 차트 생성 오류: {e}"); return None

def get_news_sentiment(ticker, api_key):
    # ... (이전과 동일) ...
    if not api_key: logging.warning("NEWS_API_KEY 없음."); return ["뉴스 API 키 미설정."]
    url = f"https://newsapi.org/v2/everything?q={ticker}&pageSize=20&language=en&sortBy=publishedAt&apiKey={api_key}"; try: response = requests.get(url, timeout=10); response.raise_for_status(); articles = response.json().get('articles', []);
    if not articles: logging.info(f"{ticker}: 관련 뉴스 없음."); return ["관련 뉴스 없음."]; output, total_pol, count = [], 0, 0
    for i, article in enumerate(articles, 1): title = article.get('title', 'N/A'); text = article.get('description') or article.get('content') or title or "";
    if text and text != "[Removed]": try: blob = TextBlob(text); pol = blob.sentiment.polarity; output.append(f"{i}. {title} | 감정: {pol:.2f}"); total_pol += pol; count += 1; except Exception as text_e: logging.warning(f"뉴스 처리 오류({title}): {text_e}"); output.append(f"{i}. {title} | 감정 분석 오류")
    else: output.append(f"{i}. {title} | 내용 없음")
    avg_pol = total_pol / count if count > 0 else 0; logging.info(f"{ticker} 뉴스 분석 완료 (평균: {avg_pol:.2f})"); output.insert(0, f"총 {count}개 분석 | 평균 감성: {avg_pol:.2f}"); return output
    except requests.exceptions.RequestException as e: logging.error(f"뉴스 API 요청 실패: {e}"); return [f"뉴스 API 요청 실패: {e}"]
    except Exception as e: logging.error(f"뉴스 분석 오류: {e}"); return ["뉴스 분석 중 오류 발생."]

# --- ⭐ run_prophet_forecast (UnboundLocalError 방지 및 CV 오류 처리 개선) ---
def run_prophet_forecast(ticker, start_date, end_date=None, forecast_days=30, fred_key=None):
    """Prophet 예측, 교차 검증 수행 후 결과 반환"""
    if end_date is None: end_date = datetime.today().strftime("%Y-%m-%d")

    # --- ⭐ df_stock 로딩 부분을 try-except로 감싸고 초기화 ---
    df_stock = None # None으로 초기화
    try:
        df_stock = get_stock_data(ticker, start_date=start_date, end_date=end_date)
        if df_stock is None or df_stock.empty:
             logging.error(f"{ticker}: 주가 데이터 없음(Prophet).")
             return None, None, None # 함수 종료
        df_stock_processed = df_stock.reset_index()[["Date", "Close"]] # 새 변수 사용
        df_stock_processed["Date"] = pd.to_datetime(df_stock_processed["Date"])
    except Exception as get_data_err:
        logging.error(f"{ticker}: 초기 주가 데이터 로딩 중 오류: {get_data_err}")
        return None, None, None # 함수 종료
    # --- ⭐ 수정된 부분 끝 ---

    df_macro = get_macro_data(start_date=start_date, end_date=end_date, fred_key=fred_key)
    # --- 데이터 병합 및 전처리 ---
    if not df_macro.empty:
        try:
            # df_stock_processed 사용
            df_stock_processed['Date'] = pd.to_datetime(df_stock_processed['Date']); df_macro['Date'] = pd.to_datetime(df_macro['Date'])
            df_merged = pd.merge(df_stock_processed, df_macro, on="Date", how="left"); logging.info(f"{ticker}: 주가/매크로 병합.")
            macro_cols = ["VIX", "US10Y", "US13W", "DXY", "FedFunds"]
            for col in macro_cols:
                if col in df_merged.columns:
                    df_merged[col] = pd.to_numeric(df_merged[col], errors='coerce')
                    df_merged[col] = df_merged[col].ffill().bfill()
        except Exception as merge_err:
             logging.error(f"{ticker}: 데이터 병합 오류: {merge_err}"); df_merged = df_stock_processed; logging.warning(f"{ticker}: 주가 데이터만 사용.")
    else: logging.warning(f"{ticker}: 매크로 데이터 없어 주가만 사용."); df_merged = df_stock_processed

    if df_merged['Close'].isnull().any(): df_merged = df_merged.dropna(subset=['Close'])
    if df_merged.empty or len(df_merged) < 30: logging.error(f"Prophet 실패: 최종 데이터 부족 ({len(df_merged)})."); return None, None, None
    logging.info(f"Prophet 학습 데이터 준비: {len(df_merged)} 행"); os.makedirs(DATA_FOLDER, exist_ok=True); data_csv_path = os.path.join(DATA_FOLDER, f"{ticker}_merged_for_prophet.csv")
    try: df_merged.to_csv(data_csv_path, index=False); logging.info(f"학습 데이터 저장: {data_csv_path}")
    except Exception as e: logging.error(f"학습 데이터 저장 실패: {e}")
    # --- Prophet 모델링 ---
    df_prophet = df_merged.rename(columns={"Date": "ds", "Close": "y"}); df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])
    m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False, changepoint_prior_scale=0.05); regressors = []
    macro_cols_available = [col for col in macro_cols if col in df_prophet.columns and pd.api.types.is_numeric_dtype(df_prophet[col]) and df_prophet[col].isnull().sum() == 0]
    if macro_cols_available:
        for col in macro_cols_available: m.add_regressor(col); regressors.append(col); logging.info(f"{ticker}: Regressor 추가: {col}")
    else: logging.info(f"{ticker}: 유효 Regressor 없음.")
    forecast_dict, fig_fcst, cv_path = None, None, None
    try: # 학습 및 예측 전체 감싸기
        m.fit(df_prophet[['ds', 'y'] + regressors]); logging.info(f"{ticker}: Prophet 학습 완료."); os.makedirs(FORECAST_FOLDER, exist_ok=True)
        # --- CV 블록 (오류처리 포함) ---
        try:
            initial_days, period_days, horizon_days = '365 days', '90 days', f'{forecast_days} days'
            if len(df_prophet) > int(initial_days.split()[0]) + int(horizon_days.split()[0]):
                 logging.info(f"Prophet CV 시작..."); df_cv = cross_validation(m, initial=initial_days, period=period_days, horizon=horizon_days, parallel=None)
                 logging.info("Prophet CV 완료."); df_p = performance_metrics(df_cv); logging.info(f"Prophet 성능:\n{df_p.head().to_string()}")
                 fig_cv = plot_cross_validation_metric(df_cv, metric='mape'); plt.title(f'{ticker} CV MAPE'); cv_path = os.path.join(FORECAST_FOLDER, f"{ticker}_cv_mape_plot.png")
                 fig_cv.savefig(cv_path); plt.close(fig_cv); logging.info(f"CV MAPE 차트 저장: {cv_path}")
            else: logging.warning(f"{ticker}: 데이터 기간 부족 CV 건너<0xEB>뜀."); cv_path = None
        except Exception as cv_e: logging.error(f"Prophet CV 오류: {cv_e}"); cv_path = None

        # --- 미래 예측 ---
        logging.info("미래 예측 시작..."); future = m.make_future_dataframe(periods=forecast_days)
        if regressors:
            # --- ⭐ 미래 Regressor 처리 개선 (원본 df_merged 사용 명확화) ---
            temp_m = df_merged.rename(columns={'Date':'ds'}) # Merge 위해 ds 컬럼 사용
            temp_m['ds'] = pd.to_datetime(temp_m['ds']) # 타입 일치 확인
            future['ds'] = pd.to_datetime(future['ds']) # 타입 일치 확인

            # 필요한 regressor 컬럼들의 타입 확인 및 변환
            for col in regressors:
                if col in temp_m.columns:
                    temp_m[col] = pd.to_numeric(temp_m[col], errors='coerce')

            future = future.merge(temp_m[['ds'] + regressors], on='ds', how='left') # ds 기준 merge

            for col in regressors:
                if col in future.columns:
                    # NaN 아닌 마지막 값을 원본 df_merged에서 찾기
                    non_na_series = df_merged[col].dropna()
                    last_val = non_na_series.iloc[-1] if not non_na_series.empty else 0
                    if non_na_series.empty: logging.warning(f"Regressor '{col}' 과거 값 모두 NaN.")
                    # ffill 먼저 실행 후, 그래도 남은 NaN을 last_val로 채움
                    future[col] = future[col].ffill().fillna(last_val)
            # --- ⭐ 수정된 부분 끝 ---
        forecast = m.predict(future); logging.info("미래 예측 완료.")
        csv_fn = os.path.join(FORECAST_FOLDER, f"{ticker}_forecast_data.csv"); forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy().assign(ds=lambda dfx: dfx['ds'].dt.strftime('%Y-%m-%d')).to_csv(csv_fn, index=False); logging.info(f"예측 데이터 저장: {csv_fn}")
        fig_fcst = plot_plotly(m, forecast); fig_fcst.update_layout(title=f'{ticker} Price Forecast', margin=dict(l=20,r=20,t=40,b=20)); logging.info(f"예측 Figure 생성.")
        forecast_dict = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(forecast_days).to_dict('records'); [rec.update({'ds': rec['ds'].strftime('%Y-%m-%d')}) for rec in forecast_dict]
        return forecast_dict, fig_fcst, cv_path # cv_path 반환 유지
    except Exception as e: logging.error(f"Prophet 학습/예측 오류: {e}"); logging.error(traceback.format_exc()); return None, None, None


# --- 메인 분석 함수 ---

# 7. 통합 분석 - 함수 호출 결과 None 처리 강화
def analyze_stock(ticker, news_key, fred_key, analysis_period_years=2, forecast_days=30, num_trend_periods=4):
    """모든 데이터를 종합하여 주식 분석 결과를 반환합니다."""
    logging.info(f"--- {ticker} 주식 분석 시작 ---"); output_results = {}
    try: end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0); start_date = end_date - relativedelta(years=analysis_period_years); start_date_str = start_date.strftime("%Y-%m-%d"); end_date_str = end_date.strftime("%Y-%m-%d"); logging.info(f"분석 기간: {start_date_str} ~ {end_date_str}")
    except Exception as e: logging.error(f"날짜 설정 오류: {e}"); return {"error": f"날짜 설정 오류: {e}"}

    # 각 단계 결과 None 체크 및 기본값 할당 강화
    df_stock_full = get_stock_data(ticker, start_date=start_date_str, end_date=end_date_str)
    if df_stock_full is not None and not df_stock_full.empty: output_results['current_price'] = f"{df_stock_full['Close'].iloc[-1]:.2f}" if not df_stock_full['Close'].empty else "N/A"; output_results['data_points'] = len(df_stock_full)
    else: output_results['current_price'] = "N/A"; output_results['data_points'] = 0; logging.warning(f"{ticker} 주가 정보 실패.")
    output_results['analysis_period_start'] = start_date_str; output_results['analysis_period_end'] = end_date_str

    # 각 분석 함수 호출 시 None일 경우 기본값 설정 (or 연산자 사용)
    output_results['stock_chart_fig'] = plot_stock_chart(ticker, start_date=start_date_str, end_date=end_date_str) or None
    output_results['fundamentals'] = get_fundamental_data(ticker) or {"시가총액": "N/A", "PER": "N/A", "EPS": "N/A", "배당수익률": "N/A", "베타": "N/A", "업종": "N/A", "산업": "N/A", "요약": "N/A"}
    output_results['operating_margin_trend'] = get_operating_margin_trend(ticker, num_periods=num_trend_periods) or []
    output_results['roe_trend'] = get_roe_trend(ticker, num_periods=num_trend_periods) or []
    output_results['debt_to_equity_trend'] = get_debt_to_equity_trend(ticker, num_periods=num_trend_periods) or []
    output_results['current_ratio_trend'] = get_current_ratio_trend(ticker, num_periods=num_trend_periods) or []
    output_results['news_sentiment'] = get_news_sentiment(ticker, news_key) or ["뉴스 분석 실패"]
    fg_value, fg_class = get_fear_greed_index()
    output_results['fear_greed_index'] = {'value': fg_value, 'classification': fg_class} if fg_value is not None else "N/A"

    # Prophet 예측 실행 조건 확인
    if output_results['data_points'] > 30 and output_results['current_price'] != "N/A":
        forecast_result = run_prophet_forecast(ticker, start_date=start_date_str, end_date=end_date_str, forecast_days=forecast_days, fred_key=fred_key)
        if forecast_result and isinstance(forecast_result, tuple) and len(forecast_result) == 3: # 반환값 튜플 확인
            output_results['prophet_forecast'] = forecast_result[0] or "예측 실패"
            output_results['forecast_fig'] = forecast_result[1] # None일 수 있음
            output_results['cv_plot_path'] = forecast_result[2] # None일 수 있음
        else: # run_prophet_forecast 자체가 None 또는 잘못된 튜플 반환 시
            output_results['prophet_forecast'] = "예측 실행 오류"; output_results['forecast_fig'] = None; output_results['cv_plot_path'] = None
            logging.error(f"{ticker}: run_prophet_forecast 함수가 예상치 못한 값을 반환함.")
    else:
        msg = f"데이터 부족 ({output_results['data_points']})" if output_results['data_points'] <= 30 else "주가 정보 없음"; output_results['prophet_forecast'] = f"{msg}으로 예측 불가"; output_results['forecast_fig'] = None; output_results['cv_plot_path'] = None; logging.warning(f"{msg} Prophet 예측 건너<0xEB>뜀.")

    logging.info(f"--- {ticker} 주식 분석 완료 ---"); return output_results


# --- 메인 실행 부분 (테스트용 - 최종 검토 버전) ---
if __name__ == "__main__":
    print(f"stock_analysis.py 직접 실행 (테스트 목적, Base directory: {BASE_DIR}).")
    target_ticker = "MSFT"
    news_key = os.getenv("NEWS_API_KEY"); fred_key = os.getenv("FRED_API_KEY")
    if not news_key or not fred_key: print("경고: API 키 없음."); test_results = None
    else: test_results = analyze_stock(ticker=target_ticker, news_key=news_key, fred_key=fred_key, analysis_period_years=1, forecast_days=15, num_trend_periods=5)
    print("\n--- 테스트 실행 결과 요약 ---")
    if test_results and isinstance(test_results, dict) and "error" not in test_results:
        for key, value in test_results.items():
            if 'fig' in key and value is not None: print(f"- {key.replace('_',' ').title()}: Plotly Figure 생성됨")
            elif key == 'fundamentals' and isinstance(value, dict): print(f"- Fundamentals:");
            for f_key, f_value in value.items(): print(f"    - {f_key}: {f_value}")
            elif '_trend' in key and isinstance(value, list): print(f"- {key.replace('_',' ').title()}: {len(value)} 분기");
            for item in value: print(f"    - {item}")
            elif key == 'prophet_forecast': print(f"- Prophet Forecast: {type(value)} {f'({len(value)}일)' if isinstance(value, list) else ''}")
            elif key == 'news_sentiment': print(f"- News Sentiment: {len(value) if isinstance(value, list) else 0} 항목")
            else: print(f"- {key.replace('_',' ').title()}: {value}")
    elif test_results and "error" in test_results: print(f"분석 중 오류 발생: {test_results['error']}")
    else: print("테스트 분석 실패 (결과 없음 또는 알 수 없는 타입).")
    print("\n--- 테스트 실행 종료 ---")