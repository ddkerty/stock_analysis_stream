# stock_analysis.py (기술적 지표 Regressor 추가 - 오류 수정 버전)

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
import pandas_ta as ta # ⭐ pandas-ta 라이브러리 import 추가

# --- 기본 설정, 경로, API 키 로드 ---
warnings.simplefilter(action='ignore', category=FutureWarning)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()
    logging.info(f"__file__ 없음. CWD 사용: {BASE_DIR}")

CHARTS_FOLDER = os.path.join(BASE_DIR, "charts")
DATA_FOLDER = os.path.join(BASE_DIR, "data")
FORECAST_FOLDER = os.path.join(BASE_DIR, "forecast")

try:
    dotenv_path = os.path.join(BASE_DIR, '.env')
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path=dotenv_path)
        logging.info(f".env 로드 성공: {dotenv_path}")
    else:
        # .env 파일이 없는 것은 오류가 아닐 수 있음
        logging.info(f".env 파일 없음 (정상일 수 있음): {dotenv_path}") # Warning에서 Info로 변경
except Exception as e:
    logging.error(f".env 로드 오류: {e}")

NEWS_API_KEY = os.getenv("NEWS_API_KEY")
FRED_API_KEY = os.getenv("FRED_API_KEY")

if not NEWS_API_KEY: logging.warning("NEWS_API_KEY 없음.")
if not FRED_API_KEY: logging.warning("FRED_API_KEY 없음.")
if NEWS_API_KEY and FRED_API_KEY: logging.info("API 키 로드 시도 완료.")

# --- 데이터 가져오기 함수들 ---
def get_fear_greed_index():
    """공포-탐욕 지수 가져오기 (안정성 개선)"""
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
                    logging.info(f"F&G 성공: {value} ({classification})")
                except (ValueError, TypeError):
                    logging.warning(f"F&G 값 변환 오류: {value_str}")
            else:
                logging.warning("F&G 데이터 구조 오류.")
        else:
            logging.warning("F&G 데이터 형식 오류.")
    except requests.exceptions.RequestException as e:
        logging.error(f"F&G API 요청 오류: {e}")
    except Exception as e:
        logging.error(f"F&G 처리 오류: {e}")
    return value, classification

def get_stock_data(ticker, start_date=None, end_date=None, period="1y"):
    """주가 데이터 가져오기 (OHLCV 포함, NaN 처리 개선)"""
    try:
        stock = yf.Ticker(ticker)
        if start_date and end_date:
            data = stock.history(start=start_date, end=end_date, auto_adjust=False)
        else:
            data = stock.history(period=period, auto_adjust=False)
        logging.info(f"{ticker} 주가 가져오기 시도 완료.")

        if data.empty:
            logging.warning(f"{ticker} 주가 데이터 비어있음.")
            return None
        if isinstance(data.index, pd.DatetimeIndex):
            data.index = data.index.tz_localize(None)

        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            logging.warning(f"{ticker}: 필수 컬럼 {missing_cols} 누락 -> NaN으로 채움.")
            for col in missing_cols:
                data[col] = np.nan # 컬럼이 없으면 생성하고 NaN 할당

        # 필수 컬럼 숫자형 변환 및 오류 시 NaN 처리
        for col in required_cols:
            if col in data.columns: # 컬럼 존재 확인 후 변환
                 data[col] = pd.to_numeric(data[col], errors='coerce')

        # 기술적 지표 계산 위해 필요한 컬럼 NaN 있으면 해당 행 제거 (Close 제외)
        # Close의 NaN은 이후 Prophet 단계에서 처리됨
        essential_ta_cols = ['Open', 'High', 'Low', 'Volume']
        initial_len = len(data)
        data.dropna(subset=[col for col in essential_ta_cols if col in data.columns], inplace=True)
        if len(data) < initial_len:
             logging.warning(f"{ticker}: TA 계산 위한 필수 컬럼(O,H,L,V) NaN 값으로 {initial_len - len(data)} 행 제거")

        return data

    except Exception as e:
        logging.error(f"티커 '{ticker}' 주가 데이터 가져오기 실패: {e}")
        return None

def get_macro_data(start_date, end_date=None, fred_key=None):
    """매크로 지표 데이터 가져오기 (빈 DF 컬럼 명시)"""
    if end_date is None:
        end_date = datetime.today().strftime("%Y-%m-%d")
    yf_tickers = {"^VIX": "VIX", "^TNX": "US10Y", "^IRX": "US13W", "DX-Y.NYB": "DXY"}
    fred_series = {"FEDFUNDS": "FedFunds"}
    expected_cols = ['Date'] + list(yf_tickers.values()) + list(fred_series.values())
    df_macro = pd.DataFrame()
    all_yf_data = []

    for tk, label in yf_tickers.items():
        try:
            tmp = yf.download(tk, start=start_date, end=end_date, progress=False, timeout=15)
            if not tmp.empty:
                tmp = tmp[['Close']].rename(columns={"Close": label})
                tmp.index = pd.to_datetime(tmp.index).tz_localize(None)
                all_yf_data.append(tmp)
                logging.info(f"{label} 성공")
            else:
                logging.warning(f"{label} 비어있음.")
        except Exception as e:
            logging.error(f"{label} 실패: {e}")

    if all_yf_data:
        df_macro = pd.concat(all_yf_data, axis=1)
        if isinstance(df_macro.columns, pd.MultiIndex):
            df_macro.columns = df_macro.columns.get_level_values(-1)

    if fred_key:
        try:
            fred = Fred(api_key=fred_key)
            fred_data = []
            for series_id, label in fred_series.items():
                s = fred.get_series(series_id, start_date=start_date, end_date=end_date).rename(label)
                s.index = pd.to_datetime(s.index).tz_localize(None)
                fred_data.append(s)
            if fred_data:
                df_fred = pd.concat(fred_data, axis=1)
                if not df_macro.empty:
                    df_macro = df_macro.merge(df_fred, left_index=True, right_index=True, how='outer')
                else:
                    df_macro = df_fred
                logging.info("FRED 병합/가져오기 성공")
        except Exception as e:
            logging.error(f"FRED 실패: {e}")
    else:
        logging.warning("FRED 키 없어 스킵.")

    if not df_macro.empty:
        for col in expected_cols:
            if col != 'Date' and col not in df_macro.columns:
                df_macro[col] = pd.NA
                logging.warning(f"매크로 '{col}' 없어 NaN 추가.")
        for col in df_macro.columns:
            if col != 'Date':
                df_macro[col] = pd.to_numeric(df_macro[col], errors='coerce')
        df_macro = df_macro.sort_index().ffill().bfill()
        df_macro = df_macro.reset_index().rename(columns={'index': 'Date'})
        df_macro["Date"] = pd.to_datetime(df_macro["Date"])
        logging.info("매크로 처리 완료.")
        return df_macro[expected_cols]
    else:
        logging.warning("매크로 가져오기 최종 실패.")
        return pd.DataFrame(columns=expected_cols)

def format_market_cap(mc):
    """시가총액 숫자 포맷팅"""
    if isinstance(mc, (int, float)) and mc > 0:
        if mc >= 1e12: return f"${mc / 1e12:.2f} T"
        elif mc >= 1e9: return f"${mc / 1e9:.2f} B"
        elif mc >= 1e6: return f"${mc / 1e6:.2f} M"
        else: return f"${mc:,.0f}"
    return "N/A"

def find_financial_statement_item(index, keywords, exact_match_keywords=None, case_sensitive=False):
    """재무제표 인덱스 항목 찾기 (안정성 강화 버전)"""
    if not isinstance(index, pd.Index): return None
    flags = 0 if case_sensitive else re.IGNORECASE

    if exact_match_keywords:
        for exact_key in exact_match_keywords:
            if exact_key in index:
                logging.debug(f"정확 매칭: '{exact_key}' for {keywords}")
                return exact_key

    pattern_keywords = [re.escape(k) for k in keywords]
    pattern = r'\b' + r'\b.*\b'.join(pattern_keywords) + r'\b'
    matches = []
    for item in index:
        if isinstance(item, str):
            try:
                if re.search(pattern, item, flags=flags):
                    matches.append(item)
            except Exception as e:
                logging.warning(f"항목명 검색 정규식 오류({keywords}, item='{item}'): {e}")
                continue

    if matches:
        best_match = min(matches, key=len)
        logging.debug(f"포함 매칭: '{best_match}' for {keywords}")
        return best_match

    logging.warning(f"재무 항목 찾기 최종 실패: {keywords}")
    return None

def get_fundamental_data(ticker):
    """yfinance .info 사용하여 주요 기본 지표 가져오기."""
    logging.info(f"{ticker}: .info 가져오기...")
    fundamentals = {key: "N/A" for key in ["시가총액", "PER", "EPS", "배당수익률", "베타", "업종", "산업", "요약"]}
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        if not info or info.get('regularMarketPrice') is None:
            logging.warning(f"'{ticker}' 유효 .info 없음.")
            return fundamentals

        fundamentals["시가총액"] = format_market_cap(info.get("marketCap"))
        fwd_pe = info.get('forwardPE')
        trl_pe = info.get('trailingPE')
        if isinstance(fwd_pe, (int, float)):
            fundamentals["PER"] = f"{fwd_pe:.2f} (Fwd)"
        elif isinstance(trl_pe, (int, float)):
            fundamentals["PER"] = f"{trl_pe:.2f} (Trl)"

        eps_val = info.get('trailingEps')
        fundamentals["EPS"] = f"{eps_val:.2f}" if isinstance(eps_val, (int, float)) else "N/A"

        div_yield = info.get('dividendYield')
        fundamentals["배당수익률"] = f"{div_yield * 100:.2f}%" if isinstance(div_yield, (int, float)) and div_yield > 0 else "N/A"

        beta_val = info.get('beta')
        fundamentals["베타"] = f"{beta_val:.2f}" if isinstance(beta_val, (int, float)) else "N/A"

        fundamentals["업종"] = info.get("sector", "N/A")
        fundamentals["산업"] = info.get("industry", "N/A")
        fundamentals["요약"] = info.get("longBusinessSummary", "N/A")

        logging.info(f"{ticker} .info 성공.")
        return fundamentals
    except Exception as e:
        logging.error(f"{ticker} .info 실패: {e}")
        return fundamentals

def get_operating_margin_trend(ticker, num_periods=4):
    """최근 분기별 영업이익률 추세 계산."""
    logging.info(f"{ticker}: 영업이익률 추세 ({num_periods}분기)...")
    try:
        stock = yf.Ticker(ticker)
        qf = stock.quarterly_financials
        if qf.empty:
            logging.warning(f"{ticker}: 분기 재무 없음.")
            return None
        revenue_col = find_financial_statement_item(qf.index, ['Total', 'Revenue'], ['Total Revenue', 'Revenue'])
        op_income_col = find_financial_statement_item(qf.index, ['Operating', 'Income'], ['Operating Income', 'Operating Income Loss'])
        if not revenue_col or not op_income_col:
            logging.warning(f"{ticker}: 매출/영업이익 항목 못찾음.")
            return None

        qf_recent = qf.iloc[:, :num_periods]
        df = qf_recent.loc[[revenue_col, op_income_col]].T.sort_index()
        df.index = pd.to_datetime(df.index)

        df[revenue_col] = pd.to_numeric(df[revenue_col], errors='coerce')
        df[op_income_col] = pd.to_numeric(df[op_income_col], errors='coerce')
        df[revenue_col] = df[revenue_col].replace(0, np.nan)
        df.dropna(subset=[revenue_col, op_income_col], inplace=True)

        if df.empty:
            logging.warning(f"{ticker}: 영업이익률 계산 데이터 부족.")
            return None

        df['Operating Margin (%)'] = (df[op_income_col] / df[revenue_col]) * 100
        df['Operating Margin (%)'] = df['Operating Margin (%)'].round(2)

        res = df[['Operating Margin (%)']].reset_index().rename(columns={'index':'Date'})
        res['Date'] = res['Date'].dt.strftime('%Y-%m-%d')
        logging.info(f"{ticker}: {len(res)}개 영업이익률 계산 완료.")
        return res.to_dict('records')
    except Exception as e:
        logging.error(f"{ticker}: 영업이익률 계산 오류: {e}")
        return None

def get_roe_trend(ticker, num_periods=4):
    """최근 분기별 ROE(%) 추세 계산."""
    logging.info(f"{ticker}: ROE 추세 ({num_periods}분기)...")
    try:
        stock = yf.Ticker(ticker)
        qf = stock.quarterly_financials
        qbs = stock.quarterly_balance_sheet
        if qf.empty or qbs.empty:
            logging.warning(f"{ticker}: 분기 재무/대차대조표 없음.")
            return None

        ni_col = find_financial_statement_item(qf.index, ['Net', 'Income'], ['Net Income', 'Net Income Common Stockholders', 'Net Income Applicable To Common Stockholders'])
        eq_col = find_financial_statement_item(qbs.index, ['Stockholder', 'Equity'], ['Total Stockholder Equity', 'Stockholders Equity']) or find_financial_statement_item(qbs.index, ['Total', 'Equity'])
        if not ni_col or not eq_col:
            logging.warning(f"{ticker}: 순이익/자본 항목 못찾음.")
            return None

        qf_r = qf.loc[[ni_col]].iloc[:, :num_periods].T
        qbs_r = qbs.loc[[eq_col]].iloc[:, :num_periods].T
        df = pd.merge(qf_r, qbs_r, left_index=True, right_index=True, how='outer').sort_index()
        df.index = pd.to_datetime(df.index)

        df[ni_col] = pd.to_numeric(df[ni_col], errors='coerce')
        df[eq_col] = pd.to_numeric(df[eq_col], errors='coerce')
        df[eq_col] = df[eq_col].apply(lambda x: x if pd.notna(x) and x > 0 else np.nan)
        df.dropna(subset=[ni_col, eq_col], inplace=True)

        if df.empty:
            logging.warning(f"{ticker}: ROE 계산 데이터 부족.")
            return None

        df['ROE (%)'] = (df[ni_col] / df[eq_col]) * 100
        df['ROE (%)'] = df['ROE (%)'].round(2)

        res = df[['ROE (%)']].reset_index().rename(columns={'index':'Date'})
        res['Date'] = res['Date'].dt.strftime('%Y-%m-%d')
        logging.info(f"{ticker}: {len(res)}개 ROE 계산 완료.")
        return res.to_dict('records')
    except Exception as e:
        logging.error(f"{ticker}: ROE 계산 오류: {e}")
        return None

def get_debt_to_equity_trend(ticker, num_periods=4):
    """최근 분기별 부채비율(D/E Ratio) 추세 계산."""
    logging.info(f"{ticker}: 부채비율 추세 ({num_periods}분기)...")
    try:
        stock = yf.Ticker(ticker)
        qbs = stock.quarterly_balance_sheet
        if qbs.empty:
            logging.warning(f"{ticker}: 분기 대차대조표 없음.")
            return None

        eq_col = find_financial_statement_item(qbs.index, ['Stockholder', 'Equity'], ['Total Stockholder Equity', 'Stockholders Equity']) or find_financial_statement_item(qbs.index, ['Total', 'Equity'])
        if not eq_col:
            logging.warning(f"{ticker}: 자본 항목 못찾음.")
            return None

        td_col = find_financial_statement_item(qbs.index, ['Total', 'Debt'])
        sd_col = find_financial_statement_item(qbs.index, ['Current', 'Debt'])
        ld_col = find_financial_statement_item(qbs.index, ['Long', 'Term', 'Debt'])

        req_cols = [eq_col]
        use_td = False
        calc_d = False

        if td_col:
            req_cols.append(td_col)
            use_td = True
            logging.info(f"{ticker}: Total Debt 사용.")
        elif sd_col and ld_col:
            req_cols.extend([sd_col, ld_col])
            calc_d = True
            logging.info(f"{ticker}: 단기+장기 부채 합산.")
        else:
            logging.warning(f"{ticker}: 총부채 관련 항목 못찾음.")
            return None

        req_cols = list(set(req_cols)) # 중복 제거
        qbs_r = qbs.loc[req_cols].iloc[:, :num_periods].T
        df = qbs_r.copy()
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        for col in req_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df[eq_col] = df[eq_col].apply(lambda x: x if pd.notna(x) and x != 0 else np.nan) # 자본 0도 NaN 처리

        if use_td:
            df['Calc Debt'] = df[td_col]
        elif calc_d:
            df['Calc Debt'] = df[sd_col].fillna(0) + df[ld_col].fillna(0)
        else:
            # 이 경우는 위에서 처리되었으므로 이론적으로 도달하지 않음
            return None

        df.dropna(subset=['Calc Debt', eq_col], inplace=True) # 자본이 0(NaN)이거나 계산된 부채가 NaN이면 계산 불가

        if df.empty:
            logging.warning(f"{ticker}: 부채비율 계산 데이터 부족.")
            return None

        df['D/E Ratio'] = df['Calc Debt'] / df[eq_col]
        df['D/E Ratio'] = df['D/E Ratio'].round(2)

        res = df[['D/E Ratio']].reset_index().rename(columns={'index':'Date'})
        res['Date'] = res['Date'].dt.strftime('%Y-%m-%d')
        logging.info(f"{ticker}: {len(res)}개 부채비율 계산 완료.")
        return res.to_dict('records')
    except Exception as e:
        logging.error(f"{ticker}: 부채비율 계산 오류: {e}")
        return None

def get_current_ratio_trend(ticker, num_periods=4):
    """최근 분기별 유동비율 추세 계산."""
    logging.info(f"{ticker}: 유동비율 추세 ({num_periods}분기)...")
    try:
        stock = yf.Ticker(ticker)
        qbs = stock.quarterly_balance_sheet
        if qbs.empty:
            logging.warning(f"{ticker}: 분기 대차대조표 없음.")
            return None

        ca_col = find_financial_statement_item(qbs.index, ['Total', 'Current', 'Assets'])
        cl_col = find_financial_statement_item(qbs.index, ['Total', 'Current', 'Liabilities'])

        if not ca_col or not cl_col:
            logging.warning(f"{ticker}: 유동자산/부채 항목 못찾음.")
            return None
        if ca_col == cl_col: # 동일 컬럼명으로 식별될 경우 방지
            logging.error(f"{ticker}: 유동자산/부채 항목 동일 식별('{ca_col}').")
            return None

        req_cols = [ca_col, cl_col]
        qbs_r = qbs.loc[req_cols].iloc[:, :num_periods].T
        df = qbs_r.copy()
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        df[ca_col] = pd.to_numeric(df[ca_col], errors='coerce')
        df[cl_col] = pd.to_numeric(df[cl_col], errors='coerce')

        df[cl_col] = df[cl_col].apply(lambda x: x if pd.notna(x) and x > 0 else np.nan) # 유동부채 0이거나 NaN이면 계산 불가
        df.dropna(subset=[ca_col, cl_col], inplace=True)

        if df.empty:
            logging.warning(f"{ticker}: 유동비율 계산 데이터 부족.")
            return None

        df['Current Ratio'] = df[ca_col] / df[cl_col]
        df['Current Ratio'] = df['Current Ratio'].round(2)

        res = df[['Current Ratio']].reset_index().rename(columns={'index':'Date'})
        res['Date'] = res['Date'].dt.strftime('%Y-%m-%d')
        logging.info(f"{ticker}: {len(res)}개 유동비율 계산 완료.")
        return res.to_dict('records')
    except Exception as e:
        logging.error(f"{ticker}: 유동비율 계산 오류: {e}")
        return None

# --- 분석 및 시각화 함수들 ---
def plot_stock_chart(ticker, start_date=None, end_date=None, period="1y"):
    """주가 차트 Figure 객체 반환."""
    df = get_stock_data(ticker, start_date=start_date, end_date=end_date, period=period)
    if df is None or df.empty:
        logging.error(f"{ticker} 차트 실패: 데이터 없음")
        return None
    try:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='가격'), row=1, col=1)
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='거래량', marker_color='rgba(0,0,100,0.6)'), row=2, col=1)
        fig.update_layout(title=f'{ticker} 주가/거래량 차트', yaxis_title='가격', yaxis2_title='거래량', xaxis_rangeslider_visible=False, hovermode='x unified', margin=dict(l=20, r=20, t=40, b=20))
        fig.update_yaxes(title_text="가격", row=1, col=1)
        fig.update_yaxes(title_text="거래량", row=2, col=1)
        logging.info(f"{ticker} 차트 생성 완료")
        return fig
    except Exception as e:
        logging.error(f"{ticker} 차트 생성 오류: {e}")
        return None

def get_news_sentiment(ticker, api_key):
    """뉴스 감정 분석."""
    if not api_key:
        logging.warning("NEWS_API_KEY 없음.")
        return ["뉴스 API 키 미설정."]
    url = f"https://newsapi.org/v2/everything?q={ticker}&pageSize=20&language=en&sortBy=publishedAt&apiKey={api_key}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        articles = response.json().get('articles', [])
        if not articles:
            logging.info(f"{ticker}: 관련 뉴스 없음.")
            return ["관련 뉴스 없음."]

        output, total_pol, count = [], 0, 0
        for i, article in enumerate(articles, 1):
            title = article.get('title', 'N/A')
            # description이나 content가 None일 경우 빈 문자열로 처리
            description = article.get('description', '') or ""
            content = article.get('content', '') or ""
            text = description or content or title or "" # 최종적으로 title이라도 사용

            if text and text != "[Removed]":
                try:
                    blob = TextBlob(text)
                    pol = blob.sentiment.polarity
                    output.append(f"{i}. {title} | 감정: {pol:.2f}")
                    total_pol += pol
                    count += 1
                except Exception as text_e:
                    logging.warning(f"뉴스 처리 오류({title}): {text_e}")
                    output.append(f"{i}. {title} | 감정 분석 오류")
            else:
                output.append(f"{i}. {title} | 내용 없음")

        avg_pol = total_pol / count if count > 0 else 0
        logging.info(f"{ticker} 뉴스 분석 완료 (평균: {avg_pol:.2f})")
        output.insert(0, f"총 {count}개 분석 | 평균 감성: {avg_pol:.2f}")
        return output
    except requests.exceptions.RequestException as e:
        logging.error(f"뉴스 API 요청 실패: {e}")
        return [f"뉴스 API 요청 실패: {e}"]
    except Exception as e:
        logging.error(f"뉴스 분석 오류: {e}")
        return ["뉴스 분석 중 오류 발생."]

# --- ⭐ run_prophet_forecast (기술적 지표 Regressor 추가 - 최종 수정) ---
def run_prophet_forecast(ticker, start_date, end_date=None, forecast_days=30, fred_key=None):
    """Prophet 예측 (기술적 지표 Regressor 추가)"""
    logging.info(f"{ticker}: Prophet 예측 시작 (기술 지표 포함)...")
    if end_date is None:
        end_date = datetime.today().strftime("%Y-%m-%d")

    # 1. 초기 주가 데이터 로딩 (OHLCV 포함 및 NaN 처리 개선)
    df_stock = None
    try:
        # get_stock_data 함수에서 OHLCV 로드 및 필수 컬럼 NaN 처리 수행
        df_stock = get_stock_data(ticker, start_date=start_date, end_date=end_date)
        if df_stock is None or df_stock.empty:
             # get_stock_data 내부에서 이미 로깅하므로 여기서는 반환만
             return None, None, None
        # Prophet은 'Close' 컬럼의 NaN을 내부적으로 처리 가능 (제거 불필요)
        # if df_stock['Close'].isnull().any():
        #    logging.warning(f"{ticker}: 'Close' 컬럼에 NaN 값 존재. Prophet이 처리합니다.")
        logging.info(f"{ticker}: 초기 주가 데이터 로딩 및 전처리 완료 (Shape: {df_stock.shape}).")
    except Exception as get_data_err:
        logging.error(f"{ticker}: 초기 주가 로딩/처리 중 오류: {get_data_err}")
        return None, None, None

    # 2. 매크로 데이터 로딩 및 병합
    # df_stock의 인덱스(Date)를 컬럼으로 변환하여 merge 준비
    df_stock_for_merge = df_stock.reset_index()
    df_stock_for_merge["Date"] = pd.to_datetime(df_stock_for_merge["Date"])

    df_macro = get_macro_data(start_date=start_date, end_date=end_date, fred_key=fred_key)
    macro_cols = ["VIX", "US10Y", "US13W", "DXY", "FedFunds"] # 나중에 사용될 매크로 컬럼명

    if not df_macro.empty:
        try:
            df_macro['Date'] = pd.to_datetime(df_macro['Date']) # Date 타입 통일
            # 주가 데이터와 매크로 데이터 병합 (left join)
            df_merged = pd.merge(df_stock_for_merge, df_macro, on="Date", how="left")
            logging.info(f"{ticker}: 주가/매크로 병합 완료.")
            # 매크로 변수의 NaN 값을 ffill 후 bfill로 채움
            for col in macro_cols:
                if col in df_merged.columns:
                    df_merged[col] = pd.to_numeric(df_merged[col], errors='coerce').ffill().bfill()
            logging.info(f"{ticker}: 매크로 변수 NaN 처리 완료.")
        except Exception as merge_err:
            logging.error(f"{ticker}: 데이터 병합 오류: {merge_err}")
            df_merged = df_stock_for_merge # 병합 실패 시 주가 데이터만 사용
            logging.warning(f"{ticker}: 주가 데이터만 사용하여 진행.")
    else:
        logging.warning(f"{ticker}: 매크로 데이터 없음. 주가 데이터만 사용.")
        df_merged = df_stock_for_merge

    # --- ⭐ 3. 기술적 지표 계산 (pandas_ta 사용) ---
    logging.info(f"{ticker}: 기술적 지표 계산 시작...")
    tech_indicators = [] # 사용할 기술 지표 컬럼 리스트 초기화
    try:
        # RSI 계산 (14일 기준)
        df_merged.ta.rsi(length=14, append=True) # 'RSI_14' 컬럼 추가
        # MACD 계산 (기본값 12, 26, 9)
        df_merged.ta.macd(fast=12, slow=26, signal=9, append=True) # 'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9' 추가
        logging.info(f"{ticker}: RSI, MACD 계산 완료.")

        # 사용할 기술적 지표 컬럼명 정의 (예: RSI와 MACD 시그널)
        tech_indicators_to_use = ['RSI_14', 'MACDs_12_26_9']

        # 해당 컬럼들이 실제로 생성되었는지 확인
        for ti in tech_indicators_to_use:
            if ti in df_merged.columns:
                 tech_indicators.append(ti)
            else:
                 logging.warning(f"{ticker}: 기술 지표 '{ti}'가 생성되지 않았습니다.")

        # 기술적 지표 생성 초기의 NaN 값 처리 (해당 행 제거하지 않고 Prophet이 처리하도록 둠)
        # -> Prophet은 regressor의 NaN을 허용하지 않음. ffill/bfill 처리 필요
        if tech_indicators:
             initial_nan_count = df_merged[tech_indicators].isnull().sum().sum()
             if initial_nan_count > 0:
                 logging.info(f"{ticker}: 기술 지표 초기 NaN 값 {initial_nan_count}개를 ffill/bfill로 처리합니다.")
                 for col in tech_indicators:
                     df_merged[col] = df_merged[col].ffill().bfill()
             # 최종 확인
             final_nan_count = df_merged[tech_indicators].isnull().sum().sum()
             if final_nan_count > 0:
                  logging.error(f"{ticker}: 기술 지표 NaN 처리 후에도 {final_nan_count}개의 NaN이 남아있습니다. Regressor에서 제외합니다.")
                  # NaN이 남은 컬럼은 regressor 리스트에서 제거
                  cols_with_nan = df_merged[tech_indicators].isnull().any()
                  tech_indicators = [col for col in tech_indicators if not cols_with_nan[col]]


    except Exception as ta_err:
        logging.error(f"{ticker}: 기술적 지표 계산 중 오류: {ta_err}")
        tech_indicators = [] # 오류 발생 시 기술 지표 사용 안 함

    # -----------------------------------------

    # 4. 최종 데이터 검증 및 Prophet 준비
    # Prophet은 최소 2개 이상의 데이터 포인트 필요, 실제로는 더 많은 데이터 권장
    if df_merged.empty or len(df_merged) < 30:
        logging.error(f"Prophet 실패: 최종 데이터 부족({len(df_merged)}).")
        return None, None, None

    logging.info(f"Prophet 학습 데이터 준비 완료 (Shape: {df_merged.shape})")
    os.makedirs(DATA_FOLDER, exist_ok=True)
    data_csv = os.path.join(DATA_FOLDER, f"{ticker}_prophet_tech_input.csv") # 입력 데이터 저장
    try:
        df_merged.to_csv(data_csv, index=False)
        logging.info(f"Prophet 학습 데이터 저장 완료: {data_csv}")
    except Exception as e:
        logging.error(f"학습 데이터 저장 실패: {e}")

    # --- Prophet 모델링 (Regressor 추가) ---
    # Prophet 입력 형식에 맞게 컬럼명 변경 ('Date' -> 'ds', 'Close' -> 'y')
    df_prophet = df_merged.rename(columns={"Date": "ds", "Close": "y"})
    df_prophet['ds'] = pd.to_datetime(df_prophet['ds']) # ds 컬럼 타입을 datetime으로 확실히 지정

    # Prophet 모델 초기화
    m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False, changepoint_prior_scale=0.05)

    # 사용 가능한 모든 Regressor 목록 정의 및 모델에 추가
    all_regressors = []
    # 매크로 변수 추가 (NaN 없는지 최종 확인)
    macro_cols_available = [col for col in macro_cols if col in df_prophet.columns and pd.api.types.is_numeric_dtype(df_prophet[col]) and df_prophet[col].isnull().sum() == 0]
    if macro_cols_available:
        for col in macro_cols_available:
             m.add_regressor(col)
        all_regressors.extend(macro_cols_available)
        logging.info(f"{ticker}: 매크로 Regressors 추가됨: {macro_cols_available}")
    else:
        logging.info(f"{ticker}: 유효한 매크로 Regressor 없음.")

    # 기술적 지표 변수 추가 (NaN 없는지 최종 확인 - 위에서 처리함)
    tech_indicators_available = [col for col in tech_indicators if col in df_prophet.columns and pd.api.types.is_numeric_dtype(df_prophet[col]) and df_prophet[col].isnull().sum() == 0]
    if tech_indicators_available:
        for col in tech_indicators_available:
             m.add_regressor(col)
        all_regressors.extend(tech_indicators_available)
        logging.info(f"{ticker}: 기술 지표 Regressors 추가됨: {tech_indicators_available}")
    else:
        logging.info(f"{ticker}: 유효한 기술 지표 Regressor 없음.")

    # 5. 학습, 예측, CV 실행
    forecast_dict, fig_fcst, cv_path = None, None, None
    try:
        # 모델 학습 (ds, y 및 모든 추가된 regressors 사용)
        m.fit(df_prophet[['ds', 'y'] + all_regressors])
        logging.info(f"{ticker}: Prophet 모델 학습 완료.")
        os.makedirs(FORECAST_FOLDER, exist_ok=True) # 예측 결과 저장 폴더 확인/생성

        # 교차 검증(CV) 실행
        try:
            # CV 파라미터 설정
            initial_cv = f'{min(365*2, int(len(df_prophet)*0.5))} days' # 초기 학습 기간 (데이터 기간의 50% or 최대 2년)
            period_cv = f'{max(30, int(len(df_prophet)*0.1))} days' # 다음 학습 기간 (데이터 기간의 10% or 최소 30일)
            horizon_cv = f'{forecast_days} days' # 예측 기간

            # 데이터 기간이 CV 실행에 충분한지 확인
            # (initial + horizon + period) 보다 길어야 의미 있음
            if len(df_prophet) > (pd.Timedelta(initial_cv).days + pd.Timedelta(horizon_cv).days + pd.Timedelta(period_cv).days):
                logging.info(f"Prophet CV 시작 (initial='{initial_cv}', period='{period_cv}', horizon='{horizon_cv}')...")
                # parallel=None 설정 (필요시 'processes' 등으로 변경 가능)
                df_cv = cross_validation(m, initial=initial_cv, period=period_cv, horizon=horizon_cv, parallel=None)
                logging.info("CV 완료.")
                # 성능 지표 계산
                df_p = performance_metrics(df_cv)
                logging.info(f"Prophet CV 성능 지표:\n{df_p.head().to_string()}")
                # MAPE 시각화 및 저장
                fig_cv = plot_cross_validation_metric(df_cv, metric='mape')
                plt.title(f'{ticker} CV MAPE (Tech Regressors)')
                cv_path = os.path.join(FORECAST_FOLDER, f"{ticker}_cv_mape_tech_plot.png")
                fig_cv.savefig(cv_path)
                plt.close(fig_cv) # 메모리 관리 위해 plot 닫기
                logging.info(f"CV MAPE 차트 저장 완료: {cv_path}")
            else:
                logging.warning(f"{ticker}: 데이터 기간({len(df_prophet)}일)이 부족하여 CV를 건너<0xEB>니다.") # 인코딩 오류 수정: 건너뜁니다
                cv_path = None
        except Exception as cv_e:
            logging.error(f"Prophet CV 중 오류 발생: {cv_e}")
            cv_path = None # CV 실패 시 경로 None 설정

        # 미래 예측 수행
        logging.info("미래 예측 시작...")
        # 예측할 미래 날짜 데이터프레임 생성
        future = m.make_future_dataframe(periods=forecast_days)

        # Regressor가 있다면 미래 값 채우기
        if all_regressors:
            # 과거 데이터에서 Regressor 값 가져오기 (ds 컬럼 포함)
            temp_m = df_prophet[['ds'] + all_regressors].copy()
            future = future.merge(temp_m, on='ds', how='left') # future 날짜에 맞춰 과거 값 병합

            # 미래 기간의 Regressor 값 채우기 (과거 마지막 값으로 채우기 - ffill 후 fillna)
            for col in all_regressors:
                if col in future.columns:
                    # NaN 아닌 마지막 값 찾기 (과거 데이터 기준)
                    non_na_past = df_prophet[col].dropna()
                    last_val = non_na_past.iloc[-1] if not non_na_past.empty else 0
                    if non_na_past.empty:
                         logging.warning(f"Regressor '{col}'의 과거 값이 모두 NaN입니다. 0으로 채웁니다.")

                    # ffill() : 과거 값으로 채움, fillna(last_val) : 미래 값(NaN)을 마지막 값으로 채움
                    future[col] = future[col].ffill().fillna(last_val)

        # 예측 실행
        forecast = m.predict(future)
        logging.info("미래 예측 완료.")

        # 결과 저장 및 반환값 준비
        csv_fn = os.path.join(FORECAST_FOLDER, f"{ticker}_forecast_tech_output.csv")
        # 예측 결과 중 필요한 컬럼만 선택하여 저장
        forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy().assign(ds=lambda dfx: dfx['ds'].dt.strftime('%Y-%m-%d')).to_csv(csv_fn, index=False)
        logging.info(f"예측 결과 데이터 저장 완료: {csv_fn}")

        # 예측 결과 시각화 Figure 생성
        fig_fcst = plot_plotly(m, forecast)
        fig_fcst.update_layout(title=f'{ticker} Price Forecast (Tech Regressors)', margin=dict(l=20,r=20,t=40,b=20))
        logging.info(f"예측 결과 Figure 생성 완료.")

        # 반환할 예측 결과 딕셔너리 생성 (미래 예측 기간만)
        forecast_dict = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(forecast_days).to_dict('records')
        # 날짜 형식 변경 ('YYYY-MM-DD')
        for rec in forecast_dict:
            rec['ds'] = rec['ds'].strftime('%Y-%m-%d')

        return forecast_dict, fig_fcst, cv_path # 최종 결과 반환
    except Exception as e:
        logging.error(f"Prophet 학습/예측 단계에서 오류 발생: {e}")
        logging.error(traceback.format_exc()) # 상세 오류 로그 출력
        return None, None, None # 오류 시 None 반환

# --- 메인 분석 함수 ---
def analyze_stock(ticker, news_key, fred_key, analysis_period_years=2, forecast_days=30, num_trend_periods=4):
    """모든 데이터를 종합하여 주식 분석 결과를 반환합니다."""
    logging.info(f"--- {ticker} 주식 분석 시작 ---")
    output_results = {}
    try:
        end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        start_date = end_date - relativedelta(years=analysis_period_years)
        start_date_str = start_date.strftime("%Y-%m-%d")
        end_date_str = end_date.strftime("%Y-%m-%d")
        logging.info(f"분석 기간: {start_date_str} ~ {end_date_str}")
    except Exception as e:
        logging.error(f"날짜 설정 오류: {e}")
        return {"error": f"날짜 설정 오류: {e}"}

    # 주가 데이터 로드 (OHLCV 포함)
    df_stock_full = get_stock_data(ticker, start_date=start_date_str, end_date=end_date_str)
    stock_data_valid = df_stock_full is not None and not df_stock_full.empty

    # 현재 가격 및 데이터 포인트 수 기록
    if stock_data_valid:
        # 마지막 날짜의 종가 사용
        output_results['current_price'] = f"{df_stock_full['Close'].iloc[-1]:.2f}" if not df_stock_full['Close'].empty and pd.notna(df_stock_full['Close'].iloc[-1]) else "N/A"
        output_results['data_points'] = len(df_stock_full)
    else:
        output_results['current_price'] = "N/A"
        output_results['data_points'] = 0
        logging.warning(f"{ticker}: 분석 기간 내 유효한 주가 정보 없음.")

    output_results['analysis_period_start'] = start_date_str
    output_results['analysis_period_end'] = end_date_str

    # 각종 분석 수행 및 결과 저장 (기본값 설정으로 None 방지)
    output_results['stock_chart_fig'] = plot_stock_chart(ticker, start_date=start_date_str, end_date=end_date_str) or None
    output_results['fundamentals'] = get_fundamental_data(ticker) or {key: "N/A" for key in ["시가총액", "PER", "EPS", "배당수익률", "베타", "업종", "산업", "요약"]}
    output_results['operating_margin_trend'] = get_operating_margin_trend(ticker, num_periods=num_trend_periods) or []
    output_results['roe_trend'] = get_roe_trend(ticker, num_periods=num_trend_periods) or []
    output_results['debt_to_equity_trend'] = get_debt_to_equity_trend(ticker, num_periods=num_trend_periods) or []
    output_results['current_ratio_trend'] = get_current_ratio_trend(ticker, num_periods=num_trend_periods) or []
    output_results['news_sentiment'] = get_news_sentiment(ticker, news_key) or ["뉴스 분석 실패"]
    fg_value, fg_class = get_fear_greed_index()
    output_results['fear_greed_index'] = {'value': fg_value, 'classification': fg_class} if fg_value is not None else "N/A"

    # Prophet 예측 수행 (주가 데이터 유효하고 충분할 경우)
    if stock_data_valid and output_results['data_points'] > 30 : # 데이터 포인트 30개 초과 시 예측 시도
        forecast_result = run_prophet_forecast(ticker, start_date=start_date_str, end_date=end_date_str, forecast_days=forecast_days, fred_key=fred_key)
        # 예측 결과가 정상적인 튜플 형태인지 확인
        if forecast_result and isinstance(forecast_result, tuple) and len(forecast_result) == 3:
            output_results['prophet_forecast'] = forecast_result[0] or "예측 실패" # 예측 딕셔너리
            output_results['forecast_fig'] = forecast_result[1] # 예측 그래프 Figure
            output_results['cv_plot_path'] = forecast_result[2] # CV 그래프 경로
        else:
            output_results['prophet_forecast'] = "예측 실행 오류"
            output_results['forecast_fig'] = None
            output_results['cv_plot_path'] = None
            logging.error(f"{ticker}: run_prophet_forecast 함수가 비정상적인 값을 반환했습니다.")
    else:
        # 데이터 부족 또는 주가 정보 없음 메시지
        msg = f"데이터 부족({output_results['data_points']})" if output_results['data_points'] <= 30 else "주가 정보 없음"
        output_results['prophet_forecast'] = f"{msg}으로 예측 불가"
        output_results['forecast_fig'] = None
        output_results['cv_plot_path'] = None
        logging.warning(f"{ticker}: {msg} - Prophet 예측을 건너<0xEB>니다.") # 인코딩 오류 수정: 건너<0xEB>니다

    logging.info(f"--- {ticker} 주식 분석 완료 ---")
    return output_results

# --- 메인 실행 부분 (테스트용 - 최종 검토 버전) ---
if __name__ == "__main__":
    print(f"stock_analysis.py 직접 실행 (테스트 목적, Base directory: {BASE_DIR}).")
    target_ticker = "MSFT" # 테스트할 티커
    news_key = os.getenv("NEWS_API_KEY")
    fred_key = os.getenv("FRED_API_KEY")

    if not news_key or not fred_key:
        print("경고: NEWS_API_KEY 또는 FRED_API_KEY 환경 변수가 설정되지 않았습니다. 일부 기능이 제한될 수 있습니다.")
        # 키가 없어도 기본적인 분석은 가능하도록 None 대신 빈 문자열 전달 고려 가능
        # news_key = news_key or ""
        # fred_key = fred_key or ""
        test_results = analyze_stock(ticker=target_ticker, news_key=news_key, fred_key=fred_key, analysis_period_years=1, forecast_days=15, num_trend_periods=5)
    else:
        print("API 키 로드됨. 모든 분석 기능 실행.")
        test_results = analyze_stock(ticker=target_ticker, news_key=news_key, fred_key=fred_key, analysis_period_years=1, forecast_days=15, num_trend_periods=5)

    print("\n--- 테스트 실행 결과 요약 ---")
    if test_results and isinstance(test_results, dict) and "error" not in test_results:
        for key, value in test_results.items():
            if 'fig' in key and value is not None:
                print(f"- {key.replace('_',' ').title()}: Plotly Figure 객체 생성됨 (표시 안 함)")
            elif key == 'fundamentals' and isinstance(value, dict):
                print("- Fundamentals:")
                for f_key, f_value in value.items():
                    # 요약(longBusinessSummary)은 너무 길 수 있으므로 일부만 출력
                    if f_key == '요약' and isinstance(f_value, str) and len(f_value) > 100:
                         print(f"    - {f_key}: {f_value[:100]}...")
                    else:
                         print(f"    - {f_key}: {f_value}")
            elif '_trend' in key and isinstance(value, list):
                print(f"- {key.replace('_',' ').title()} ({len(value)} 분기):")
                for item in value[:3]: # 최대 3개 항목만 출력
                    print(f"    - {item}")
                if len(value) > 3: print("     ...")
            elif key == 'prophet_forecast':
                forecast_status = "예측 실패 또는 오류"
                if isinstance(value, list) and value:
                     forecast_status = f"{len(value)}일 예측 생성됨 (첫 날: {value[0]})"
                elif isinstance(value, str):
                     forecast_status = value # "예측 불가" 등의 메시지
                print(f"- Prophet Forecast: {forecast_status}")
            elif key == 'news_sentiment':
                news_status = "뉴스 분석 실패 또는 오류"
                if isinstance(value, list) and value:
                     news_status = f"{len(value)-1}개 뉴스 분석됨 (헤더: {value[0]})"
                elif isinstance(value, list) and not value: # 빈 리스트
                     news_status = "분석된 뉴스 없음"
                print(f"- News Sentiment: {news_status}")
            elif key == 'cv_plot_path':
                 print(f"- Cv Plot Path: {value if value else '생성 안 됨'}")
            else:
                 # 나머지 키들은 간단히 출력
                 print(f"- {key.replace('_',' ').title()}: {value}")
    elif test_results and "error" in test_results:
        print(f"분석 중 오류 발생: {test_results['error']}")
    else:
        print("테스트 분석 실패 (결과 없음 또는 알 수 없는 오류).")

    print("\n--- 테스트 실행 종료 ---")