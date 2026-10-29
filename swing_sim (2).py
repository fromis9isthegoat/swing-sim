"""
단기 스윙 모의투자 (국내 + 미국, 가상 자본금 1억 원)
실행: python swing_sim.py [KR|US]   (인자 없으면 전체)
결과: state.json, trade_log.csv
"""
import csv
import json
import os
import sys
from datetime import datetime, timezone

import pandas as pd
import yfinance as yf

CAPITAL = 100_000_000
RISK_PER_TRADE = 0.01
MAX_POSITION = 0.20
MAX_POSITIONS = 8
FEE = 0.001
HOLD_DAYS = 14
RSI_MAX = 45               # 눌림목 기준 (높일수록 매수가 잦아짐)
STATE, LOG = "state.json", "trade_log.csv"

MARKETS = {
    "KR": {"index": "^KS11", "fx": False, "symbols": {
        "005930.KS": "삼성전자", "000660.KS": "SK하이닉스", "035420.KS": "NAVER",
        "005380.KS": "현대차", "051910.KS": "LG화학", "035720.KS": "카카오"}},
    "US": {"index": "^GSPC", "fx": True, "symbols": {
        "AAPL": "Apple", "MSFT": "Microsoft", "NVDA": "NVIDIA", "AMZN": "Amazon",
        "GOOGL": "Alphabet", "TSLA": "Tesla", "META": "Meta"}},
}


def fetch(sym):
    df = yf.download(sym, period="8mo", interval="1d", progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna()
    c, h, l = df["Close"], df["High"], df["Low"]
    df["ema20"] = c.ewm(span=20, adjust=False).mean()
    df["ema50"] = c.ewm(span=50, adjust=False).mean()
    d = c.diff()
    gain = d.clip(lower=0).rolling(14).mean()
    loss = (-d.clip(upper=0)).rolling(14).mean()
    df["rsi"] = 100 - 100 / (1 + gain / loss)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()
    return df


def usdkrw():
    try:
        x = yf.download("KRW=X", period="5d", progress=False, auto_adjust=True)["Close"].dropna().squeeze()
        return float(x.iloc[-1])
    except Exception:
        return 1400.0


def load():
    if os.path.exists(STATE):
        return json.load(open(STATE, encoding="utf-8"))
    return {"cash": CAPITAL, "positions": {}, "last_day": {}, "prices": {}}


def save(s):
    json.dump(s, open(STATE, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


def log(action, sym, name, qty, price, reason):
    new = not os.path.exists(LOG)
    with open(LOG, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["시각", "구분", "종목", "수량", "가격", "이유"])
        w.writerow([datetime.now().strftime("%Y-%m-%d %H:%M"), action, f"{name}({sym})", qty, round(price, 2), reason])
    print(f"[{action}] {name} {qty}주 @ {price:,.2f} | {reason}")


def regime(index):
    r = fetch(index).iloc[-1]
    bull = r.Close > r.ema50 and r.ema20 > r.ema50
    why = f"지수 종가 {r.Close:,.0f} / 20일선 {r.ema20:,.0f} / 50일선 {r.ema50:,.0f}"
    return bull, ("상승 국면: " if bull else "약세 국면: ") + why


def step():
    active = [a for a in sys.argv[1:] if a in MARKETS] or list(MARKETS)
    s, fx = load(), usdkrw()
    today = datetime.now().strftime("%Y-%m-%d")
    data, regimes = {}, {}
    for m in active:
        regimes[m] = regime(MARKETS[m]["index"])
        for sym in MARKETS[m]["symbols"]:
            try:
                data[sym] = fetch(sym)
            except Exception as e:
                print("데이터 오류", sym, e)

    def rate(m):
        return fx if MARKETS[m]["fx"] else 1.0

    for sym, df in data.items():
        s["prices"][sym] = float(df.iloc[-1].Close)
    equity = s["cash"] + sum(
        p["qty"] * s["prices"].get(k, p["entry"]) * rate(p["mkt"]) for k, p in s["positions"].items())

    for m in active:
        bull, rwhy = regimes[m]
        for sym, name in MARKETS[m]["symbols"].items():
            if sym not in data or s["last_day"].get(sym) == today:
                continue
            r = data[sym].iloc[-1]
            price, atr, fxr = float(r.Close), float(r.atr), rate(m)
            if sym in s["positions"]:
                p = s["positions"][sym]
                held = (datetime.now() - datetime.fromisoformat(p["date"])).days
                reason = None
                if price <= p["stop"]:
                    reason = f"손절: 가격 {price:,.2f}이(가) 손절선 {p['stop']:,.2f} 하회"
                elif price >= p["target"]:
                    reason = f"익절: 가격 {price:,.2f}이(가) 목표가 {p['target']:,.2f} 도달"
                elif not bull:
                    reason = f"시장 약세 전환으로 청산 ({rwhy})"
                elif price < r.ema50:
                    reason = f"추세 이탈: 종가가 50일선 {r.ema50:,.2f} 아래"
                elif held >= HOLD_DAYS:
                    reason = f"시간 손절: {held}일 보유, 기대한 반등 없음"
                if reason:
                    s["cash"] += p["qty"] * price * fxr * (1 - FEE)
                    pnl = (price / p["entry"] - 1) * 100
                    log("매도", sym, name, p["qty"], price, f"{reason} (수익률 {pnl:+.1f}%)")
                    del s["positions"][sym]
                    s["last_day"][sym] = today
            elif bull and len(s["positions"]) < MAX_POSITIONS:
                prev = float(data[sym].iloc[-2].Close)
                uptrend = r.ema20 > r.ema50 and price > r.ema50
                pullback = r.rsi < RSI_MAX
                rebound = price > prev
                if uptrend and pullback and rebound and atr > 0:
                    stop_dist = 2 * atr
                    qty = int(min(equity * RISK_PER_TRADE / (stop_dist * fxr),
                                  equity * MAX_POSITION / (price * fxr),
                                  s["cash"] / (price * fxr * (1 + FEE))))
                    if qty >= 1:
                        s["cash"] -= qty * price * fxr * (1 + FEE)
                        s["positions"][sym] = {"qty": qty, "entry": price, "stop": price - stop_dist,
                                               "target": price + 3 * atr, "date": datetime.now().isoformat(),
                                               "mkt": m}
                        log("매수", sym, name, qty, price,
                            f"{rwhy} | 상승추세(20일선>50일선), 눌림목 RSI {r.rsi:.0f}<{RSI_MAX}, 전일 대비 반등 | "
                            f"손절 {price - stop_dist:,.2f}(2ATR), 목표 {price + 3 * atr:,.2f}(3ATR), "
                            f"1회 위험 {RISK_PER_TRADE:.0%}")
                        s["last_day"][sym] = today

    equity = s["cash"] + sum(
        p["qty"] * s["prices"].get(k, p["entry"]) * rate(p["mkt"]) for k, p in s["positions"].items())
    print(f"\n=== {datetime.now():%Y-%m-%d %H:%M} | 총자산 {equity:,.0f}원 "
          f"({(equity / CAPITAL - 1) * 100:+.2f}%) | 현금 {s['cash']:,.0f}원 | 보유 {len(s['positions'])}종목 ===")
    for m in active:
        print(f"[{m}] {regimes[m][1]}")
    s["fx"] = fx
    s["updated"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    save(s)


if __name__ == "__main__":
    step()
