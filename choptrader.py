import os
import time
import csv
import argparse
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd
from dotenv import load_dotenv

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest,
    LimitOrderRequest,
    GetOrdersRequest,
    TakeProfitRequest,
    StopLossRequest,
)
from alpaca.trading.enums import (
    OrderSide,
    TimeInForce,
    QueryOrderStatus,
    OrderClass,
)

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed


# ============================================================
# AURA TRADER V2.1
# Adaptive Relative Strength Rotation Scalper
# With --after-hours support for 4:00 PM–8:00 PM ET
# ============================================================
# Regular market:
#   - Uses bracket orders: market buy + take-profit + stop-loss
#
# After-hours mode:
#   - Use: python aura_trader.py --paper --after-hours
#   - Allows 4:00 PM–8:00 PM ET
#   - Uses extended-hours LIMIT orders only
#   - Uses synthetic exits watched by the bot loop
#
# Paper trading only.
# ============================================================


load_dotenv()

API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

if not API_KEY or not SECRET_KEY:
    raise ValueError("Missing ALPACA_API_KEY or ALPACA_SECRET_KEY in .env file")


# ============================================================
# Universe
# ============================================================

LONG_SYMBOLS = [
    "SPY", "QQQ", "IWM", "DIA", "TQQQ", "SOXL", "SPXL", "TNA",

    "NVDA", "TSLA", "AMD", "AAPL", "MSFT", "META", "AMZN", "GOOGL", "GOOG",
    "AVGO", "SMCI", "ARM", "MU", "INTC", "QCOM", "MRVL", "AMAT", "LRCX", "KLAC",

    "PLTR", "COIN", "MSTR", "NFLX", "UBER", "SHOP", "SNOW", "CRWD", "PANW",
    "NET", "DDOG", "RBLX", "HOOD", "SQ", "PYPL", "AFRM", "ROKU", "DKNG",

    "RIVN", "LCID", "NIO", "XPEV", "BABA", "PDD", "JD",

    "JPM", "BAC", "C", "GS", "MS", "WFC",

    "XOM", "CVX", "OXY", "SLB", "BA", "GE", "CAT", "DE",

    "LLY", "NVO", "UNH", "MRK", "PFE", "ABBV",

    "WMT", "COST", "TGT", "HD", "LOW", "DIS", "NKE", "SBUX",
]

INVERSE_SYMBOLS = [
    "SQQQ", "SOXS", "SPXS", "TZA",
]

REGIME_SYMBOLS = ["SPY", "QQQ"]

ALL_SYMBOLS = sorted(list(set(LONG_SYMBOLS + INVERSE_SYMBOLS + REGIME_SYMBOLS)))


# ============================================================
# Strategy / risk settings
# ============================================================

MAX_OPEN_POSITIONS = 4

MAX_TOTAL_EXPOSURE = 0.95
MAX_POSITION_ALLOCATION = 0.35
MIN_POSITION_ALLOCATION = 0.05

MIN_SCORE_TO_TRADE = 6.0
SCORE_POWER = 1.35

ORDER_COOLDOWN_SECONDS = 180
MIN_PRICE = 3.00
MIN_BUYING_POWER_FLOOR = 1000

LOOKBACK_DAYS = 2
MIN_BARS_REQUIRED = 35
LOOP_SECONDS = 15

REGULAR_MAX_CANDLE_AGE_SECONDS = 180
AFTER_HOURS_MAX_CANDLE_AGE_SECONDS = 600

DAILY_KILL_SWITCH = 0.03

USE_BRACKET_ORDERS_REGULAR_HOURS = True

MIN_TAKE_PROFIT_PCT = 0.0035
MAX_TAKE_PROFIT_PCT = 0.0120
MIN_STOP_LOSS_PCT = 0.0025
MAX_STOP_LOSS_PCT = 0.0080

ATR_TAKE_PROFIT_MULT = 1.00
ATR_STOP_LOSS_MULT = 0.65

AFTER_HOURS_BUY_LIMIT_BUFFER = 0.002
AFTER_HOURS_SELL_LIMIT_BUFFER = 0.002

ALLOW_CHOPPY_MEAN_REVERSION = False

LOG_FILE = "trades.csv"


# ============================================================
# Clients / local state
# ============================================================

trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)
data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)

last_order_time_by_symbol = {}
local_position_entry_time = {}


# ============================================================
# Logging
# ============================================================

def init_log():
    try:
        with open(LOG_FILE, "x", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp",
                "session",
                "regime",
                "symbol",
                "price",
                "signal",
                "qty",
                "equity",
                "buying_power",
                "action",
                "reason",
                "score",
                "allocation",
                "r1",
                "r3",
                "r5",
                "relative_strength",
                "unrealized_plpc",
            ])
    except FileExistsError:
        pass


def log_event(
    session,
    regime,
    symbol,
    price,
    signal,
    qty,
    equity,
    buying_power,
    action,
    reason,
    score="",
    allocation="",
    r1="",
    r3="",
    r5="",
    relative_strength="",
    unrealized_plpc="",
):
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().isoformat(),
            session,
            regime,
            symbol,
            price,
            signal,
            qty,
            equity,
            buying_power,
            action,
            reason,
            score,
            allocation,
            r1,
            r3,
            r5,
            relative_strength,
            unrealized_plpc,
        ])


# ============================================================
# Session helpers
# ============================================================

def regular_market_is_open():
    try:
        clock = trading_client.get_clock()
        return bool(clock.is_open)
    except Exception as e:
        print(f"Clock check failed: {e}")
        return False


def after_hours_session_is_open():
    """
    Allows 4:00 PM–8:00 PM ET, Monday–Friday.

    This simple time filter does not know holidays.
    Alpaca may still reject orders on holidays or special closures.
    """
    now_et = datetime.now(ZoneInfo("America/New_York"))

    if now_et.weekday() >= 5:
        return False

    after_start = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
    after_end = now_et.replace(hour=20, minute=0, second=0, microsecond=0)

    return after_start <= now_et <= after_end


def get_session_state(after_hours_enabled):
    regular_open = regular_market_is_open()
    after_open = after_hours_session_is_open() if after_hours_enabled else False

    if regular_open:
        return "regular", True

    if after_open:
        return "after_hours", True

    return "closed", False


def current_max_candle_age_seconds(session):
    if session == "after_hours":
        return AFTER_HOURS_MAX_CANDLE_AGE_SECONDS

    return REGULAR_MAX_CANDLE_AGE_SECONDS


# ============================================================
# Alpaca helpers
# ============================================================

def get_positions_dict():
    positions = trading_client.get_all_positions()
    return {p.symbol: p for p in positions}


def get_open_orders():
    try:
        request = GetOrdersRequest(status=QueryOrderStatus.OPEN)
        orders = trading_client.get_orders(filter=request)
        return list(orders)
    except Exception as e:
        print(f"Open order check failed: {e}")
        return []


def get_pending_order_symbols(open_orders):
    symbols = set()

    for order in open_orders:
        try:
            symbols.add(order.symbol)
        except Exception:
            pass

    return symbols


def has_open_order_for_symbol(symbol):
    open_orders = get_open_orders()

    for order in open_orders:
        try:
            if order.symbol == symbol:
                return True
        except Exception:
            pass

    return False


def symbol_on_cooldown(symbol):
    last_time = last_order_time_by_symbol.get(symbol)

    if last_time is None:
        return False

    return (time.time() - last_time) < ORDER_COOLDOWN_SECONDS


def round_price(price):
    return round(float(price), 2)


def submit_regular_bracket_buy(symbol, qty, estimated_entry_price, atr_value):
    price = float(estimated_entry_price)
    atr = float(atr_value) if atr_value and atr_value > 0 else price * 0.004

    atr_pct = atr / price

    take_profit_pct = ATR_TAKE_PROFIT_MULT * atr_pct
    stop_loss_pct = ATR_STOP_LOSS_MULT * atr_pct

    take_profit_pct = max(MIN_TAKE_PROFIT_PCT, min(MAX_TAKE_PROFIT_PCT, take_profit_pct))
    stop_loss_pct = max(MIN_STOP_LOSS_PCT, min(MAX_STOP_LOSS_PCT, stop_loss_pct))

    take_profit_price = round_price(price * (1 + take_profit_pct))
    stop_loss_price = round_price(price * (1 - stop_loss_pct))

    if stop_loss_price >= price:
        stop_loss_price = round_price(price * (1 - MIN_STOP_LOSS_PCT))

    if take_profit_price <= price:
        take_profit_price = round_price(price * (1 + MIN_TAKE_PROFIT_PCT))

    order = MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=OrderSide.BUY,
        time_in_force=TimeInForce.DAY,
        order_class=OrderClass.BRACKET,
        take_profit=TakeProfitRequest(limit_price=take_profit_price),
        stop_loss=StopLossRequest(stop_price=stop_loss_price),
    )

    result = trading_client.submit_order(order)

    last_order_time_by_symbol[symbol] = time.time()
    local_position_entry_time[symbol] = time.time()

    print(
        f"REGULAR_BRACKET_BUY | {symbol} | qty={qty} | entry≈{price:.2f} | "
        f"tp={take_profit_price:.2f} | stop={stop_loss_price:.2f}"
    )

    return result


def submit_regular_market_buy(symbol, qty):
    order = MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=OrderSide.BUY,
        time_in_force=TimeInForce.DAY,
    )

    result = trading_client.submit_order(order)

    last_order_time_by_symbol[symbol] = time.time()
    local_position_entry_time[symbol] = time.time()

    print(f"REGULAR_MARKET_BUY | {symbol} | qty={qty}")

    return result


def submit_regular_market_sell(symbol, qty):
    order = MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=OrderSide.SELL,
        time_in_force=TimeInForce.DAY,
    )

    result = trading_client.submit_order(order)

    last_order_time_by_symbol[symbol] = time.time()

    print(f"REGULAR_MARKET_SELL | {symbol} | qty={qty}")

    return result


def extended_limit_buy_price(reference_price):
    return round_price(float(reference_price) * (1 + AFTER_HOURS_BUY_LIMIT_BUFFER))


def extended_limit_sell_price(reference_price):
    return round_price(float(reference_price) * (1 - AFTER_HOURS_SELL_LIMIT_BUFFER))


def submit_after_hours_limit_buy(symbol, qty, reference_price):
    limit_price = extended_limit_buy_price(reference_price)

    order = LimitOrderRequest(
        symbol=symbol,
        qty=qty,
        side=OrderSide.BUY,
        time_in_force=TimeInForce.DAY,
        limit_price=limit_price,
        extended_hours=True,
    )

    result = trading_client.submit_order(order)

    last_order_time_by_symbol[symbol] = time.time()
    local_position_entry_time[symbol] = time.time()

    print(f"AFTER_HOURS_LIMIT_BUY | {symbol} | qty={qty} | limit={limit_price:.2f}")

    return result


def submit_after_hours_limit_sell(symbol, qty, reference_price):
    limit_price = extended_limit_sell_price(reference_price)

    order = LimitOrderRequest(
        symbol=symbol,
        qty=qty,
        side=OrderSide.SELL,
        time_in_force=TimeInForce.DAY,
        limit_price=limit_price,
        extended_hours=True,
    )

    result = trading_client.submit_order(order)

    last_order_time_by_symbol[symbol] = time.time()

    print(f"AFTER_HOURS_LIMIT_SELL | {symbol} | qty={qty} | limit={limit_price:.2f}")

    return result


def cancel_open_orders_for_symbol(symbol):
    open_orders = get_open_orders()

    for order in open_orders:
        try:
            if order.symbol == symbol:
                trading_client.cancel_order_by_id(order.id)
                print(f"CANCEL_ORDER | {symbol} | id={order.id}")
        except Exception as e:
            print(f"Cancel order failed for {symbol}: {e}")


# ============================================================
# Data helpers
# ============================================================

def fetch_all_bars(symbols):
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=LOOKBACK_DAYS)

    request = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame.Minute,
        start=start,
        end=end,
        feed=DataFeed.IEX,
    )

    bars = data_client.get_stock_bars(request).df

    if bars is None or bars.empty:
        return {}

    result = {}

    if isinstance(bars.index, pd.MultiIndex):
        symbol_level = 0
        available_symbols = bars.index.get_level_values(symbol_level).unique()

        for symbol in available_symbols:
            try:
                df = bars.xs(symbol, level=symbol_level).copy()
                df = df.sort_index()
                result[symbol] = df.tail(240)
            except Exception:
                pass
    else:
        result[symbols[0]] = bars.sort_index().tail(240)

    return result


def latest_bar_age_seconds(df):
    try:
        ts = df.index[-1]

        if isinstance(ts, tuple):
            ts = ts[-1]

        ts = pd.Timestamp(ts)

        if ts.tzinfo is None:
            ts = ts.tz_localize(timezone.utc)
        else:
            ts = ts.tz_convert(timezone.utc)

        now = pd.Timestamp.now(tz=timezone.utc)

        return float((now - ts).total_seconds())
    except Exception:
        return 999999.0


def bar_is_fresh(df, session):
    age = latest_bar_age_seconds(df)
    return age <= current_max_candle_age_seconds(session)


# ============================================================
# Indicators
# ============================================================

def add_indicators(df):
    df = df.copy()

    df["ema3"] = df["close"].ewm(span=3, adjust=False).mean()
    df["ema8"] = df["close"].ewm(span=8, adjust=False).mean()
    df["ema21"] = df["close"].ewm(span=21, adjust=False).mean()

    df["r1"] = df["close"].pct_change(1)
    df["r3"] = df["close"].pct_change(3)
    df["r5"] = df["close"].pct_change(5)
    df["r15"] = df["close"].pct_change(15)

    df["volume_avg20"] = df["volume"].rolling(20).mean()
    df["volume_ratio"] = df["volume"] / df["volume_avg20"]

    df["high20"] = df["high"].rolling(20).max().shift(1)
    df["low20"] = df["low"].rolling(20).min().shift(1)

    typical_price = (df["high"] + df["low"] + df["close"]) / 3

    df["rolling_vwap20"] = (
        (typical_price * df["volume"]).rolling(20).sum()
        / df["volume"].rolling(20).sum()
    )

    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()

    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

    df["atr14"] = true_range.rolling(14).mean()

    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, 0.000001)

    df["rsi14"] = 100 - (100 / (1 + rs))

    return df.dropna()


# ============================================================
# Regime detection
# ============================================================

def get_regime(bars_by_symbol, session):
    spy_df = bars_by_symbol.get("SPY")
    qqq_df = bars_by_symbol.get("QQQ")

    if spy_df is None or qqq_df is None:
        return "unknown", "missing_SPY_or_QQQ"

    if len(spy_df) < MIN_BARS_REQUIRED or len(qqq_df) < MIN_BARS_REQUIRED:
        return "unknown", "insufficient_regime_data"

    if not bar_is_fresh(spy_df, session) or not bar_is_fresh(qqq_df, session):
        return "stale", "regime_bars_stale"

    spy = add_indicators(spy_df)
    qqq = add_indicators(qqq_df)

    if spy.empty or qqq.empty:
        return "unknown", "indicator_failure"

    s = spy.iloc[-1]
    q = qqq.iloc[-1]

    spy_bull = s["ema8"] > s["ema21"] and s["close"] > s["ema8"]
    qqq_bull = q["ema8"] > q["ema21"] and q["close"] > q["ema8"]

    spy_bear = s["ema8"] < s["ema21"] and s["close"] < s["ema8"]
    qqq_bear = q["ema8"] < q["ema21"] and q["close"] < q["ema8"]

    spy_flat = abs(float(s["ema8"] - s["ema21"])) / float(s["close"]) < 0.0008
    qqq_flat = abs(float(q["ema8"] - q["ema21"])) / float(q["close"]) < 0.0008

    if spy_bull and qqq_bull:
        return "bullish", "SPY_QQQ_bullish"

    if spy_bear and qqq_bear:
        return "bearish", "SPY_QQQ_bearish"

    if spy_flat or qqq_flat:
        return "choppy", "flat_EMA_structure"

    return "choppy", "mixed_SPY_QQQ_signals"


# ============================================================
# Candidate scoring
# ============================================================

def get_benchmark_returns(bars_by_symbol):
    result = {
        "spy_r3": 0.0,
        "spy_r5": 0.0,
        "qqq_r3": 0.0,
        "qqq_r5": 0.0,
    }

    try:
        spy = add_indicators(bars_by_symbol["SPY"])
        qqq = add_indicators(bars_by_symbol["QQQ"])

        result["spy_r3"] = float(spy.iloc[-1]["r3"])
        result["spy_r5"] = float(spy.iloc[-1]["r5"])
        result["qqq_r3"] = float(qqq.iloc[-1]["r3"])
        result["qqq_r5"] = float(qqq.iloc[-1]["r5"])
    except Exception:
        pass

    return result


def score_momentum_symbol(symbol, df, benchmark_returns, regime, session):
    if df is None or len(df) < MIN_BARS_REQUIRED:
        return None

    if not bar_is_fresh(df, session):
        return None

    df = add_indicators(df)

    if df.empty or len(df) < MIN_BARS_REQUIRED:
        return None

    latest = df.iloc[-1]
    prev = df.iloc[-2]

    price = float(latest["close"])

    if price < MIN_PRICE:
        return None

    ema3 = float(latest["ema3"])
    ema8 = float(latest["ema8"])
    ema21 = float(latest["ema21"])
    vwap = float(latest["rolling_vwap20"])
    atr = float(latest["atr14"]) if pd.notna(latest["atr14"]) else price * 0.004

    r1 = float(latest["r1"])
    r3 = float(latest["r3"])
    r5 = float(latest["r5"])
    r15 = float(latest["r15"]) if pd.notna(latest["r15"]) else 0.0

    volume_ratio = float(latest["volume_ratio"]) if pd.notna(latest["volume_ratio"]) else 0.0

    spy_r5 = benchmark_returns.get("spy_r5", 0.0)
    qqq_r5 = benchmark_returns.get("qqq_r5", 0.0)

    relative_strength = r5 - max(spy_r5, qqq_r5)

    above_ema8 = price > ema8
    trend_stack = ema3 > ema8 > ema21
    green_bar = latest["close"] > prev["close"]
    above_vwap = price > vwap

    recent_pullback = False

    try:
        recent = df.tail(6)
        touched_ema_or_vwap = (
            (recent["low"] <= recent["ema8"]).any()
            or (recent["low"] <= recent["rolling_vwap20"]).any()
        )
        reclaimed = price > ema8 and price > vwap and green_bar
        recent_pullback = bool(touched_ema_or_vwap and reclaimed)
    except Exception:
        pass

    breakout = False

    try:
        breakout = price > float(latest["high20"])
    except Exception:
        pass

    momentum_valid = (
        trend_stack
        and above_ema8
        and green_bar
        and r1 > 0.00005
        and r3 > 0.00020
        and relative_strength > -0.0005
    )

    pullback_valid = (
        ema8 > ema21
        and above_vwap
        and recent_pullback
        and r1 > 0
        and relative_strength > -0.001
    )

    if not momentum_valid and not pullback_valid:
        return {
            "symbol": symbol,
            "valid": False,
            "price": price,
            "score": 0.0,
            "reason": "no_valid_momentum_or_pullback",
            "r1": r1,
            "r3": r3,
            "r5": r5,
            "relative_strength": relative_strength,
            "atr": atr,
        }

    score = 0.0

    if trend_stack:
        score += 2.0

    if above_ema8:
        score += 1.0

    if above_vwap:
        score += 1.0

    if green_bar:
        score += 0.75

    if relative_strength > 0:
        score += 2.0 + min(relative_strength * 1000, 6.0)

    if volume_ratio > 1.1:
        score += min(volume_ratio, 3.0)

    if breakout:
        score += 1.5

    if recent_pullback:
        score += 2.0

    score += max(0, r1) * 900
    score += max(0, r3) * 700
    score += max(0, r5) * 450
    score += max(0, r15) * 200

    atr_pct = atr / price

    if atr_pct > 0.025:
        score -= 2.0

    if regime == "bearish" and symbol not in INVERSE_SYMBOLS:
        return None

    reason = "relative_strength_momentum"

    if pullback_valid:
        reason = "vwap_ema_pullback_reclaim"

    if breakout:
        reason += "_breakout"

    return {
        "symbol": symbol,
        "valid": True,
        "price": price,
        "score": round(max(score, 0), 4),
        "reason": reason,
        "r1": r1,
        "r3": r3,
        "r5": r5,
        "relative_strength": relative_strength,
        "atr": atr,
    }


def score_mean_reversion_symbol(symbol, df, benchmark_returns, session):
    if not ALLOW_CHOPPY_MEAN_REVERSION:
        return None

    if df is None or len(df) < MIN_BARS_REQUIRED:
        return None

    if not bar_is_fresh(df, session):
        return None

    df = add_indicators(df)

    if df.empty:
        return None

    latest = df.iloc[-1]
    prev = df.iloc[-2]

    price = float(latest["close"])
    vwap = float(latest["rolling_vwap20"])
    rsi = float(latest["rsi14"])
    atr = float(latest["atr14"]) if pd.notna(latest["atr14"]) else price * 0.004

    r1 = float(latest["r1"])
    r3 = float(latest["r3"])
    r5 = float(latest["r5"])

    green_bar = latest["close"] > prev["close"]
    oversold = rsi < 35
    below_vwap = price < vwap
    not_collapsing = price > float(latest["low20"]) if pd.notna(latest["low20"]) else True
    reclaim_attempt = green_bar and r1 > 0

    valid = oversold and below_vwap and not_collapsing and reclaim_attempt

    if not valid:
        return None

    distance_to_vwap = (vwap - price) / price

    score = 4.0
    score += min(distance_to_vwap * 700, 5.0)
    score += max(0, 35 - rsi) * 0.15

    return {
        "symbol": symbol,
        "valid": True,
        "price": price,
        "score": round(score, 4),
        "reason": "choppy_mean_reversion",
        "r1": r1,
        "r3": r3,
        "r5": r5,
        "relative_strength": 0.0,
        "atr": atr,
    }


# ============================================================
# Exposure / sizing
# ============================================================

def get_current_exposure_dollars(positions):
    exposure = 0.0

    for position in positions.values():
        try:
            exposure += abs(float(position.market_value))
        except Exception:
            pass

    return exposure


CONFLICT_GROUPS = [
    {"TQQQ", "SQQQ"},
    {"SOXL", "SOXS"},
    {"SPXL", "SPXS"},
    {"TNA", "TZA"},
]


def conflicts_with_existing(symbol, selected_symbols, held_symbols, pending_symbols):
    existing = set(selected_symbols) | set(held_symbols) | set(pending_symbols)

    for group in CONFLICT_GROUPS:
        if symbol in group and len(group.intersection(existing)) > 0:
            return True

    return False


def build_position_plan(candidates, positions, open_orders, equity, buying_power):
    held_symbols = set(positions.keys())
    pending_symbols = get_pending_order_symbols(open_orders)

    selected_symbols = []

    pending_not_held = pending_symbols.difference(held_symbols)
    slots_left = MAX_OPEN_POSITIONS - len(held_symbols) - len(pending_not_held)

    if slots_left <= 0:
        return []

    current_exposure = get_current_exposure_dollars(positions)
    max_total_exposure_dollars = equity * MAX_TOTAL_EXPOSURE
    available_exposure = max_total_exposure_dollars - current_exposure

    usable_capital = min(available_exposure, buying_power * 0.95)

    if usable_capital <= 0:
        return []

    eligible = []

    for candidate in candidates:
        symbol = candidate["symbol"]
        score = float(candidate["score"])

        if score < MIN_SCORE_TO_TRADE:
            continue

        if symbol in held_symbols:
            continue

        if symbol in pending_symbols:
            continue

        if symbol_on_cooldown(symbol):
            continue

        if conflicts_with_existing(symbol, selected_symbols, held_symbols, pending_symbols):
            continue

        eligible.append(candidate)

    if not eligible:
        return []

    eligible = sorted(eligible, key=lambda x: x["score"], reverse=True)

    selected = []

    for candidate in eligible:
        symbol = candidate["symbol"]

        if len(selected) >= slots_left:
            break

        already_selected = [c["symbol"] for c in selected]

        if conflicts_with_existing(symbol, already_selected, held_symbols, pending_symbols):
            continue

        selected.append(candidate)

    if not selected:
        return []

    adjusted_scores = []

    for candidate in selected:
        score = float(candidate["score"])
        adjusted = max(score - MIN_SCORE_TO_TRADE, 0.01) ** SCORE_POWER
        adjusted_scores.append(adjusted)

    total_adjusted_score = sum(adjusted_scores)

    if total_adjusted_score <= 0:
        return []

    plan = []

    for candidate, adjusted_score in zip(selected, adjusted_scores):
        symbol = candidate["symbol"]
        price = float(candidate["price"])
        score = float(candidate["score"])

        raw_weight = adjusted_score / total_adjusted_score

        max_capital = equity * MAX_POSITION_ALLOCATION
        min_capital = equity * MIN_POSITION_ALLOCATION

        target_capital = usable_capital * raw_weight
        target_capital = min(target_capital, max_capital)

        if target_capital < min_capital:
            continue

        qty = int(target_capital // price)

        if qty <= 0:
            continue

        plan.append({
            "symbol": symbol,
            "price": price,
            "qty": qty,
            "score": score,
            "capital": qty * price,
            "allocation": (qty * price) / equity,
            "weight": raw_weight,
            "reason": candidate["reason"],
            "r1": candidate["r1"],
            "r3": candidate["r3"],
            "r5": candidate["r5"],
            "relative_strength": candidate["relative_strength"],
            "atr": candidate["atr"],
        })

    return plan


# ============================================================
# Position management
# ============================================================

def manage_positions(session, regime, positions, bars_by_symbol, equity, buying_power, dry_run):
    if not positions:
        return

    print("\nPosition management:")

    for symbol, position in positions.items():
        try:
            qty = abs(int(float(position.qty)))
        except Exception:
            qty = 0

        try:
            unrealized_plpc = float(position.unrealized_plpc)
        except Exception:
            unrealized_plpc = 0.0

        latest_price = None

        if symbol in bars_by_symbol:
            try:
                latest_price = float(bars_by_symbol[symbol].iloc[-1]["close"])
            except Exception:
                latest_price = None

        latest_price_display = latest_price if latest_price is not None else ""

        print(
            f"  POSITION | {symbol:5s} | qty={qty} | "
            f"unrealized={unrealized_plpc * 100:.3f}%"
        )

        log_event(
            session=session,
            regime=regime,
            symbol=symbol,
            price=latest_price_display,
            signal="POSITION",
            qty=qty,
            equity=equity,
            buying_power=buying_power,
            action="HOLDING",
            reason="position_monitoring",
            unrealized_plpc=unrealized_plpc,
        )

        if qty <= 0 or latest_price is None:
            continue

        # Regular-hours exits are managed by bracket orders.
        # After-hours bracket orders are not used, so exits are synthetic.
        if session == "after_hours":
            if has_open_order_for_symbol(symbol):
                print(f"  EXIT_WAIT | {symbol} | open order already exists")
                continue

            if unrealized_plpc >= MIN_TAKE_PROFIT_PCT:
                print(f"  AFTER_HOURS_TAKE_PROFIT | {symbol}")

                if dry_run:
                    action = "DRY_RUN_AFTER_HOURS_TAKE_PROFIT"
                else:
                    submit_after_hours_limit_sell(symbol, qty, latest_price)
                    action = "AFTER_HOURS_TAKE_PROFIT_LIMIT_SELL_SENT"

                log_event(
                    session=session,
                    regime=regime,
                    symbol=symbol,
                    price=latest_price,
                    signal="SELL",
                    qty=qty,
                    equity=equity,
                    buying_power=buying_power,
                    action=action,
                    reason="synthetic_after_hours_take_profit",
                    unrealized_plpc=unrealized_plpc,
                )

            elif unrealized_plpc <= -MIN_STOP_LOSS_PCT:
                print(f"  AFTER_HOURS_STOP_LOSS | {symbol}")

                if dry_run:
                    action = "DRY_RUN_AFTER_HOURS_STOP_LOSS"
                else:
                    submit_after_hours_limit_sell(symbol, qty, latest_price)
                    action = "AFTER_HOURS_STOP_LOSS_LIMIT_SELL_SENT"

                log_event(
                    session=session,
                    regime=regime,
                    symbol=symbol,
                    price=latest_price,
                    signal="SELL",
                    qty=qty,
                    equity=equity,
                    buying_power=buying_power,
                    action=action,
                    reason="synthetic_after_hours_stop_loss",
                    unrealized_plpc=unrealized_plpc,
                )


# ============================================================
# Printing
# ============================================================

def print_top_candidates(candidates, limit=10):
    ranked = sorted(candidates, key=lambda x: x["score"], reverse=True)
    top = ranked[:limit]

    print("\nTop candidates:")

    for c in top:
        print(
            f"  {c['symbol']:5s} | "
            f"score={c['score']:7.3f} | "
            f"price={c['price']:8.2f} | "
            f"r1={c['r1'] * 100:7.3f}% | "
            f"r3={c['r3'] * 100:7.3f}% | "
            f"r5={c['r5'] * 100:7.3f}% | "
            f"rs={c['relative_strength'] * 100:7.3f}% | "
            f"{c['reason']}"
        )


def print_position_plan(plan):
    print("\nPosition plan:")

    for item in plan:
        print(
            f"  {item['symbol']:5s} | "
            f"qty={item['qty']:6d} | "
            f"capital=${item['capital']:,.2f} | "
            f"alloc={item['allocation'] * 100:5.1f}% | "
            f"score={item['score']:7.3f} | "
            f"weight={item['weight'] * 100:5.1f}%"
        )


# ============================================================
# Main bot
# ============================================================

def run_bot(dry_run=True, after_hours=False, ignore_market_hours=False, ignore_freshness=False):
    init_log()

    account = trading_client.get_account()
    starting_equity = float(account.equity)

    print("============================================================")
    print("AuraTrader V2.1 - Adaptive Relative Strength Rotation Scalper")
    print("============================================================")
    print(f"Mode: {'DRY RUN' if dry_run else 'PAPER ORDERS'}")
    print(f"After-hours enabled: {after_hours}")
    print(f"Starting equity: ${starting_equity:,.2f}")
    print(f"Symbols scanned: {len(ALL_SYMBOLS)}")
    print(f"Max open positions: {MAX_OPEN_POSITIONS}")
    print(f"Max total exposure: {MAX_TOTAL_EXPOSURE * 100:.0f}%")
    print(f"Max one position: {MAX_POSITION_ALLOCATION * 100:.0f}%")
    print(f"Min score to trade: {MIN_SCORE_TO_TRADE}")
    print(f"Score power: {SCORE_POWER}")
    print(f"Regular-hours bracket orders: {USE_BRACKET_ORDERS_REGULAR_HOURS}")
    print(f"Loop seconds: {LOOP_SECONDS}")
    print("Press CTRL+C to stop.")
    print("============================================================")

    while True:
        try:
            account = trading_client.get_account()
            equity = float(account.equity)
            buying_power = float(account.buying_power)

            session, session_open = get_session_state(after_hours_enabled=after_hours)

            print("\n" + "=" * 90)
            print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            print(f"Session={session.upper()} | Equity=${equity:,.2f} | Buying Power=${buying_power:,.2f}")

            if equity <= starting_equity * (1 - DAILY_KILL_SWITCH):
                print("KILL SWITCH ACTIVE. Account drawdown limit hit.")
                break

            if not session_open and not ignore_market_hours:
                print("MARKET CLOSED | Not scanning or sending new orders.")
                if after_hours:
                    print("After-hours flag is active, but current time is not 4:00 PM–8:00 PM ET.")
                time.sleep(60)
                continue

            if session == "after_hours":
                print("AFTER-HOURS MODE | Using extended-hours limit orders and synthetic exits.")

            positions = get_positions_dict()
            open_orders = get_open_orders()

            print(f"Open Positions={len(positions)} | Open Orders={len(open_orders)}")

            if open_orders:
                for order in open_orders:
                    try:
                        print(
                            f"  OPEN ORDER | {order.symbol} | "
                            f"side={order.side} | qty={order.qty} | status={order.status}"
                        )
                    except Exception:
                        pass

            bars_by_symbol = fetch_all_bars(ALL_SYMBOLS)

            if not bars_by_symbol:
                print("No market data returned.")
                time.sleep(LOOP_SECONDS)
                continue

            if not ignore_freshness:
                freshness_ok = all(
                    bar_is_fresh(bars_by_symbol[s], session)
                    for s in REGIME_SYMBOLS
                    if s in bars_by_symbol
                )

                if not freshness_ok:
                    max_age = current_max_candle_age_seconds(session)
                    print(f"STALE DATA | Regime candles older than {max_age} seconds. No trading.")
                    time.sleep(LOOP_SECONDS)
                    continue

            regime, regime_reason = get_regime(bars_by_symbol, session)

            print(f"Regime={regime.upper()} | reason={regime_reason}")

            if regime in ["unknown", "stale"]:
                print("No trades: regime unavailable or stale.")
                time.sleep(LOOP_SECONDS)
                continue

            manage_positions(
                session=session,
                regime=regime,
                positions=positions,
                bars_by_symbol=bars_by_symbol,
                equity=equity,
                buying_power=buying_power,
                dry_run=dry_run,
            )

            positions = get_positions_dict()
            open_orders = get_open_orders()

            slots_left = MAX_OPEN_POSITIONS - len(positions) - len(
                get_pending_order_symbols(open_orders).difference(set(positions.keys()))
            )

            if slots_left <= 0:
                print("No new buys: max position slots filled or pending.")
                time.sleep(LOOP_SECONDS)
                continue

            current_exposure = get_current_exposure_dollars(positions)

            if current_exposure >= equity * MAX_TOTAL_EXPOSURE:
                print("No new buys: max total exposure reached.")
                time.sleep(LOOP_SECONDS)
                continue

            if buying_power < MIN_BUYING_POWER_FLOOR:
                print(f"No new buys: buying power too low: ${buying_power:,.2f}")
                time.sleep(LOOP_SECONDS)
                continue

            if regime == "bullish":
                allowed_symbols = LONG_SYMBOLS
                mode = "bull_momentum_relative_strength"

            elif regime == "bearish":
                allowed_symbols = INVERSE_SYMBOLS
                mode = "bear_inverse_etf_momentum"

            elif regime == "choppy":
                if ALLOW_CHOPPY_MEAN_REVERSION:
                    allowed_symbols = LONG_SYMBOLS
                    mode = "choppy_mean_reversion"
                else:
                    print("CHOPPY MARKET | No trades. Mean reversion disabled.")
                    time.sleep(LOOP_SECONDS)
                    continue

            else:
                print("No trades: unrecognized regime.")
                time.sleep(LOOP_SECONDS)
                continue

            print(f"Mode={mode} | allowed symbols={len(allowed_symbols)}")

            benchmark_returns = get_benchmark_returns(bars_by_symbol)

            candidates = []

            for symbol in allowed_symbols:
                try:
                    df = bars_by_symbol.get(symbol)

                    if df is None:
                        continue

                    if not ignore_freshness and not bar_is_fresh(df, session):
                        continue

                    if regime == "choppy":
                        result = score_mean_reversion_symbol(
                            symbol=symbol,
                            df=df,
                            benchmark_returns=benchmark_returns,
                            session=session,
                        )
                    else:
                        result = score_momentum_symbol(
                            symbol=symbol,
                            df=df,
                            benchmark_returns=benchmark_returns,
                            regime=regime,
                            session=session,
                        )

                    if result is None:
                        continue

                    if result["valid"]:
                        candidates.append(result)

                    log_event(
                        session=session,
                        regime=regime,
                        symbol=symbol,
                        price=result.get("price", ""),
                        signal="BUY_CANDIDATE" if result.get("valid") else "HOLD",
                        qty=0,
                        equity=equity,
                        buying_power=buying_power,
                        action="SCANNED",
                        reason=result.get("reason", ""),
                        score=result.get("score", ""),
                        allocation="",
                        r1=result.get("r1", ""),
                        r3=result.get("r3", ""),
                        r5=result.get("r5", ""),
                        relative_strength=result.get("relative_strength", ""),
                        unrealized_plpc="",
                    )

                except Exception as e:
                    print(f"{symbol} | scan error | {e}")

            if not candidates:
                print("No valid candidates this loop.")
                time.sleep(LOOP_SECONDS)
                continue

            print_top_candidates(candidates)

            position_plan = build_position_plan(
                candidates=candidates,
                positions=positions,
                open_orders=open_orders,
                equity=equity,
                buying_power=buying_power,
            )

            if not position_plan:
                print("No position plan generated. Scores may be too low, slots full, cooldown active, or exposure limited.")
                time.sleep(LOOP_SECONDS)
                continue

            print_position_plan(position_plan)

            for item in position_plan:
                symbol = item["symbol"]
                qty = item["qty"]
                price = item["price"]
                atr = item["atr"]

                if qty <= 0:
                    print(f"SKIP | {symbol} | qty=0")
                    continue

                if symbol in positions:
                    print(f"SKIP | {symbol} | already held")
                    continue

                if has_open_order_for_symbol(symbol):
                    print(f"SKIP | {symbol} | pending order exists")
                    continue

                if symbol_on_cooldown(symbol):
                    print(f"SKIP | {symbol} | cooldown active")
                    continue

                if dry_run:
                    action = "DRY_RUN_BUY"
                    last_order_time_by_symbol[symbol] = time.time()
                    local_position_entry_time[symbol] = time.time()

                else:
                    if session == "after_hours":
                        submit_after_hours_limit_buy(symbol, qty, price)
                        action = "AFTER_HOURS_LIMIT_BUY_SENT"

                    elif USE_BRACKET_ORDERS_REGULAR_HOURS:
                        submit_regular_bracket_buy(
                            symbol=symbol,
                            qty=qty,
                            estimated_entry_price=price,
                            atr_value=atr,
                        )
                        action = "REGULAR_BRACKET_BUY_SENT"

                    else:
                        submit_regular_market_buy(symbol, qty)
                        action = "REGULAR_MARKET_BUY_SENT"

                print(
                    f"{action} | {symbol} | qty={qty} | price≈{price:.2f} | "
                    f"capital=${item['capital']:,.2f} | "
                    f"alloc={item['allocation'] * 100:.1f}% | "
                    f"score={item['score']:.3f} | reason={item['reason']}"
                )

                log_event(
                    session=session,
                    regime=regime,
                    symbol=symbol,
                    price=price,
                    signal="BUY",
                    qty=qty,
                    equity=equity,
                    buying_power=buying_power,
                    action=action,
                    reason=item["reason"],
                    score=item["score"],
                    allocation=item["allocation"],
                    r1=item["r1"],
                    r3=item["r3"],
                    r5=item["r5"],
                    relative_strength=item["relative_strength"],
                    unrealized_plpc="",
                )

            time.sleep(LOOP_SECONDS)

        except KeyboardInterrupt:
            print("\nBot stopped by user.")
            break

        except Exception as e:
            print(f"LOOP ERROR: {e}")
            time.sleep(LOOP_SECONDS)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan and log only. No orders.",
    )

    parser.add_argument(
        "--paper",
        action="store_true",
        help="Send paper orders to Alpaca.",
    )

    parser.add_argument(
        "--after-hours",
        action="store_true",
        help="Allow 4:00 PM–8:00 PM ET after-hours trading using extended-hours limit orders.",
    )

    parser.add_argument(
        "--ignore-market-hours",
        action="store_true",
        help="Bypass market-hours gate. Use for testing only.",
    )

    parser.add_argument(
        "--ignore-freshness",
        action="store_true",
        help="Bypass stale-candle gate. Use for testing only.",
    )

    args = parser.parse_args()

    if args.paper:
        run_bot(
            dry_run=False,
            after_hours=args.after_hours,
            ignore_market_hours=args.ignore_market_hours,
            ignore_freshness=args.ignore_freshness,
        )
    else:
        run_bot(
            dry_run=True,
            after_hours=args.after_hours,
            ignore_market_hours=args.ignore_market_hours,
            ignore_freshness=args.ignore_freshness,
        )