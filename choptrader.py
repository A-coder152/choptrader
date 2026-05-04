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

try:
    from alpaca.data.requests import StockLatestQuoteRequest
except Exception:
    StockLatestQuoteRequest = None

from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed


# ============================================================
# AURA TRADER V2.4
# Opening Range + VWAP Continuation Scalper
#
# Strategy:
#   - Long-only
#   - No inverse ETFs
#   - No choppy/bearish trading
#   - Trades high relative strength names only
#   - Requires opening-range breakout
#   - Requires price above VWAP and EMA trend
#   - Uses next-bar historical backtest
#   - Uses stop, take-profit, breakeven, trailing, time stop
# ============================================================


load_dotenv()

API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

if not API_KEY or not SECRET_KEY:
    raise ValueError("Missing ALPACA_API_KEY or ALPACA_SECRET_KEY in .env file")


# ============================================================
# Universe
# ============================================================

REGIME_SYMBOLS = ["SPY", "QQQ"]

TRADE_SYMBOLS = [
    "SPY", "QQQ", "TQQQ", "SOXL", "SPXL",

    "NVDA", "TSLA", "AMD", "AAPL", "MSFT", "META", "AMZN", "GOOGL", "GOOG",
    "AVGO", "MU", "QCOM", "MRVL", "AMAT", "LRCX", "KLAC",

    "PLTR", "COIN", "MSTR", "NFLX", "UBER", "SHOP", "SNOW", "CRWD", "PANW",
    "NET", "DDOG", "HOOD", "SQ", "AFRM", "ROKU", "DKNG",

    "BABA", "PDD", "JPM", "BAC", "GS", "MS", "WFC",

    "XOM", "CVX", "OXY", "BA", "GE", "CAT", "DE",

    "LLY", "NVO", "UNH", "MRK", "ABBV",

    "WMT", "COST", "HD", "LOW", "DIS", "NKE", "SBUX",
]

INVERSE_SYMBOLS = []

LEVERAGED_ETFS = {"TQQQ", "SOXL", "SPXL"}
ALL_SYMBOLS = sorted(list(set(TRADE_SYMBOLS + REGIME_SYMBOLS)))


# ============================================================
# Strategy / risk settings
# ============================================================

MAX_OPEN_POSITIONS = 2

MAX_TOTAL_EXPOSURE = 0.40
MAX_POSITION_ALLOCATION = 0.20
MIN_POSITION_ALLOCATION = 0.05
MAX_LEVERAGED_ETF_ALLOCATION = 0.06

MIN_SCORE_TO_TRADE = 13.0
SCORE_POWER = 1.20

ORDER_COOLDOWN_SECONDS = 600
MIN_PRICE = 3.00
MIN_BUYING_POWER_FLOOR = 1000

LOOKBACK_DAYS = 3
MIN_BARS_REQUIRED = 60
LOOP_SECONDS = 15

REGULAR_MAX_CANDLE_AGE_SECONDS = 180
AFTER_HOURS_MAX_CANDLE_AGE_SECONDS = 600

DAILY_KILL_SWITCH = 0.025

USE_BRACKET_ORDERS_REGULAR_HOURS = True

MIN_TAKE_PROFIT_PCT = 0.0070
MAX_TAKE_PROFIT_PCT = 0.0130
MIN_STOP_LOSS_PCT = 0.0030
MAX_STOP_LOSS_PCT = 0.0050

ATR_TAKE_PROFIT_MULT = 1.25
ATR_STOP_LOSS_MULT = 0.55

BREAKEVEN_TRIGGER_PCT = 0.0040
BREAKEVEN_LOCK_PCT = 0.0005

TRAIL_TRIGGER_PCT = 0.0070
TRAIL_GIVEBACK_PCT = 0.0035

MAX_HOLD_SECONDS = 45 * 60

OPENING_RANGE_MINUTES = 15
TRADE_START_ET = (9, 46)
TRADE_END_ET = (11, 15)
FORCE_FLATTEN_AFTER_ET = (15, 55)

MIN_VOLUME_RATIO_TO_TRADE = 1.50
MIN_RELATIVE_STRENGTH_TO_TRADE = 0.0012
MIN_R3_TO_TRADE = 0.0005
MIN_R5_TO_TRADE = 0.0012

OPENING_RANGE_BREAKOUT_BUFFER = 0.0002
MAX_BREAKOUT_EXTENSION_PCT = 0.0060
MAX_EXTENSION_FROM_VWAP_ATR = 1.15
MAX_RSI_MOMENTUM = 72

MAX_ENTRY_CHASE_PCT = 0.0015

USE_SPREAD_FILTER = True
MAX_SPREAD_PCT_REGULAR = 0.0008
MAX_SPREAD_PCT_AFTER_HOURS = 0.0030

AFTER_HOURS_NEW_ENTRIES = False
AFTER_HOURS_BUY_LIMIT_BUFFER = 0.002
AFTER_HOURS_SELL_LIMIT_BUFFER = 0.002

BACKTEST_DEFAULT_DAYS = 30
BACKTEST_DEFAULT_STARTING_EQUITY = 100000.00
BACKTEST_DEFAULT_SLIPPAGE_BPS = 5.0

LOG_SCHEMA_VERSION = "v2_4"
LOG_FILE = f"trades_{LOG_SCHEMA_VERSION}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"


# ============================================================
# Clients / local state
# ============================================================

trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)
data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)

last_order_time_by_symbol = {}
local_position_entry_time = {}
local_position_high_water = {}


# ============================================================
# Time helpers
# ============================================================

def now_et():
    return datetime.now(ZoneInfo("America/New_York"))


def timestamp_to_et(ts):
    t = pd.Timestamp(ts)

    if t.tzinfo is None:
        t = t.tz_localize(timezone.utc)
    else:
        t = t.tz_convert(timezone.utc)

    return t.tz_convert(ZoneInfo("America/New_York"))


def past_time_et(hour, minute):
    n = now_et()
    cutoff = n.replace(hour=hour, minute=minute, second=0, microsecond=0)
    return n >= cutoff


def within_entry_window_now():
    n = now_et()

    start = n.replace(
        hour=TRADE_START_ET[0],
        minute=TRADE_START_ET[1],
        second=0,
        microsecond=0,
    )

    end = n.replace(
        hour=TRADE_END_ET[0],
        minute=TRADE_END_ET[1],
        second=0,
        microsecond=0,
    )

    return start <= n < end


def same_et_date(ts, et_date):
    return timestamp_to_et(ts).date() == et_date


# ============================================================
# Logging
# ============================================================

def init_log():
    try:
        with open(LOG_FILE, "x", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "schema_version",
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
            LOG_SCHEMA_VERSION,
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
# Market session helpers
# ============================================================

def regular_market_is_open():
    try:
        clock = trading_client.get_clock()
        return bool(clock.is_open)
    except Exception as e:
        print(f"Clock check failed: {e}")
        return False


def after_hours_session_is_open():
    n = now_et()

    if n.weekday() >= 5:
        return False

    after_start = n.replace(hour=16, minute=0, second=0, microsecond=0)
    after_end = n.replace(hour=20, minute=0, second=0, microsecond=0)

    return after_start <= n <= after_end


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


def cancel_open_orders_for_symbol(symbol):
    open_orders = get_open_orders()

    for order in open_orders:
        try:
            if order.symbol == symbol:
                trading_client.cancel_order_by_id(order.id)
                print(f"CANCEL_ORDER | {symbol} | id={order.id}")
        except Exception as e:
            print(f"Cancel order failed for {symbol}: {e}")


def cancel_all_open_orders():
    open_orders = get_open_orders()

    for order in open_orders:
        try:
            trading_client.cancel_order_by_id(order.id)
            print(f"CANCEL_ORDER | {order.symbol} | id={order.id}")
        except Exception as e:
            print(f"Cancel order failed: {e}")


def compute_bracket_prices(entry_price, atr_value):
    price = float(entry_price)
    atr = float(atr_value) if atr_value and atr_value > 0 else price * 0.004

    atr_pct = atr / price

    take_profit_pct = ATR_TAKE_PROFIT_MULT * atr_pct
    stop_loss_pct = ATR_STOP_LOSS_MULT * atr_pct

    take_profit_pct = max(MIN_TAKE_PROFIT_PCT, min(MAX_TAKE_PROFIT_PCT, take_profit_pct))
    stop_loss_pct = max(MIN_STOP_LOSS_PCT, min(MAX_STOP_LOSS_PCT, stop_loss_pct))

    take_profit_price = price * (1 + take_profit_pct)
    stop_loss_price = price * (1 - stop_loss_pct)

    return take_profit_price, stop_loss_price


def submit_regular_bracket_buy(symbol, qty, estimated_entry_price, atr_value):
    take_profit_price, stop_loss_price = compute_bracket_prices(
        estimated_entry_price,
        atr_value,
    )

    order = MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=OrderSide.BUY,
        time_in_force=TimeInForce.DAY,
        order_class=OrderClass.BRACKET,
        take_profit=TakeProfitRequest(limit_price=round_price(take_profit_price)),
        stop_loss=StopLossRequest(stop_price=round_price(stop_loss_price)),
    )

    result = trading_client.submit_order(order)

    last_order_time_by_symbol[symbol] = time.time()
    local_position_entry_time[symbol] = time.time()
    local_position_high_water[symbol] = float(estimated_entry_price)

    print(
        f"REGULAR_BRACKET_BUY | {symbol} | qty={qty} | entry≈{estimated_entry_price:.2f} | "
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
    local_position_high_water[symbol] = float(reference_price)

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


# ============================================================
# Quote / spread filter
# ============================================================

def get_latest_quote(symbol):
    if StockLatestQuoteRequest is None:
        return None, "quote_request_class_unavailable"

    try:
        request = StockLatestQuoteRequest(symbol_or_symbols=[symbol], feed=DataFeed.IEX)
        quotes = data_client.get_stock_latest_quote(request)

        if isinstance(quotes, dict):
            quote = quotes.get(symbol)
        else:
            quote = getattr(quotes, symbol, None)

        if quote is None:
            return None, "quote_missing"

        return quote, "quote_ok"

    except Exception as e:
        return None, f"quote_error:{e}"


def quote_is_tradeable(symbol, session):
    if not USE_SPREAD_FILTER:
        return True, "spread_filter_disabled"

    quote, reason = get_latest_quote(symbol)

    if quote is None:
        return False, reason

    try:
        bid = float(quote.bid_price)
        ask = float(quote.ask_price)
    except Exception:
        return False, "quote_bid_ask_unavailable"

    if bid <= 0 or ask <= 0 or ask <= bid:
        return False, f"bad_quote_bid={bid}_ask={ask}"

    mid = (bid + ask) / 2.0
    spread_pct = (ask - bid) / mid

    max_spread = (
        MAX_SPREAD_PCT_AFTER_HOURS
        if session == "after_hours"
        else MAX_SPREAD_PCT_REGULAR
    )

    if spread_pct > max_spread:
        return False, f"spread_too_wide:{spread_pct:.5f}"

    return True, f"spread_ok:{spread_pct:.5f}"


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
                result[symbol] = df.tail(420)
            except Exception:
                pass
    else:
        result[symbols[0]] = bars.sort_index().tail(420)

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
    df = df.sort_index()

    df["ema3"] = df["close"].ewm(span=3, adjust=False).mean()
    df["ema8"] = df["close"].ewm(span=8, adjust=False).mean()
    df["ema21"] = df["close"].ewm(span=21, adjust=False).mean()

    df["r1"] = df["close"].pct_change(1)
    df["r3"] = df["close"].pct_change(3)
    df["r5"] = df["close"].pct_change(5)
    df["r15"] = df["close"].pct_change(15)

    df["volume_avg20"] = df["volume"].rolling(20).mean()
    df["volume_ratio"] = df["volume"] / df["volume_avg20"]

    typical_price = (df["high"] + df["low"] + df["close"]) / 3

    et_dates = [timestamp_to_et(ts).date() for ts in df.index]
    df["_session_date"] = et_dates
    df["_pv"] = typical_price * df["volume"]

    df["_cum_pv"] = df.groupby("_session_date")["_pv"].cumsum()
    df["_cum_vol"] = df.groupby("_session_date")["volume"].cumsum()
    df["session_vwap"] = df["_cum_pv"] / df["_cum_vol"].replace(0, pd.NA)

    df["rolling_vwap20"] = (
        (typical_price * df["volume"]).rolling(20).sum()
        / df["volume"].rolling(20).sum()
    )

    df["high20"] = df["high"].rolling(20).max().shift(1)
    df["low20"] = df["low"].rolling(20).min().shift(1)

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


def get_opening_range_levels(df):
    if df is None or df.empty:
        return None

    latest_ts = df.index[-1]
    latest_et = timestamp_to_et(latest_ts)
    latest_date = latest_et.date()

    if (latest_et.hour, latest_et.minute) < (9, 30 + OPENING_RANGE_MINUTES):
        return None

    if "_session_date" not in df.columns:
        return None

    session_df = df[df["_session_date"] == latest_date]

    if session_df.empty:
        return None

    opening_rows = []

    for ts, row in session_df.iterrows():
        et = timestamp_to_et(ts)
        hm = (et.hour, et.minute)

        if (9, 30) <= hm < (9, 30 + OPENING_RANGE_MINUTES):
            opening_rows.append(row)

    if len(opening_rows) < 5:
        return None

    opening_df = pd.DataFrame(opening_rows)

    opening_high = float(opening_df["high"].max())
    opening_low = float(opening_df["low"].min())

    if opening_high <= 0 or opening_low <= 0 or opening_high <= opening_low:
        return None

    return {
        "opening_high": opening_high,
        "opening_low": opening_low,
        "opening_range_pct": (opening_high - opening_low) / opening_low,
    }


# ============================================================
# Regime detection
# ============================================================

def get_regime_from_precomputed(spy_df, qqq_df):
    if spy_df is None or qqq_df is None:
        return "unknown", "missing_SPY_or_QQQ"

    if len(spy_df) < MIN_BARS_REQUIRED or len(qqq_df) < MIN_BARS_REQUIRED:
        return "unknown", "insufficient_regime_data"

    s = spy_df.iloc[-1]
    q = qqq_df.iloc[-1]

    spy_bull = s["ema8"] > s["ema21"] and s["close"] > s["ema8"] and s["close"] > s["session_vwap"]
    qqq_bull = q["ema8"] > q["ema21"] and q["close"] > q["ema8"] and q["close"] > q["session_vwap"]

    spy_bear = s["ema8"] < s["ema21"] and s["close"] < s["ema8"] and s["close"] < s["session_vwap"]
    qqq_bear = q["ema8"] < q["ema21"] and q["close"] < q["ema8"] and q["close"] < q["session_vwap"]

    if spy_bull and qqq_bull:
        return "bullish", "SPY_QQQ_bullish_vwap_confirmed"

    if spy_bear and qqq_bear:
        return "bearish", "SPY_QQQ_bearish_vwap_confirmed"

    return "choppy", "mixed_or_unconfirmed_regime"


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

    return get_regime_from_precomputed(spy, qqq)


# ============================================================
# Candidate scoring
# ============================================================

def invalid_candidate(symbol, price, reason, r1, r3, r5, relative_strength, atr):
    return {
        "symbol": symbol,
        "valid": False,
        "price": price,
        "score": 0.0,
        "reason": reason,
        "r1": r1,
        "r3": r3,
        "r5": r5,
        "relative_strength": relative_strength,
        "atr": atr,
    }


def get_benchmark_returns_from_precomputed(spy_df, qqq_df):
    result = {
        "spy_r3": 0.0,
        "spy_r5": 0.0,
        "qqq_r3": 0.0,
        "qqq_r5": 0.0,
    }

    try:
        if spy_df is not None and not spy_df.empty:
            result["spy_r3"] = float(spy_df.iloc[-1]["r3"])
            result["spy_r5"] = float(spy_df.iloc[-1]["r5"])

        if qqq_df is not None and not qqq_df.empty:
            result["qqq_r3"] = float(qqq_df.iloc[-1]["r3"])
            result["qqq_r5"] = float(qqq_df.iloc[-1]["r5"])
    except Exception:
        pass

    return result


def get_benchmark_returns(bars_by_symbol):
    try:
        spy = add_indicators(bars_by_symbol["SPY"])
        qqq = add_indicators(bars_by_symbol["QQQ"])
        return get_benchmark_returns_from_precomputed(spy, qqq)
    except Exception:
        return {
            "spy_r3": 0.0,
            "spy_r5": 0.0,
            "qqq_r3": 0.0,
            "qqq_r5": 0.0,
        }


def score_opening_range_symbol_from_indicator_df(symbol, df, benchmark_returns, regime):
    if regime != "bullish":
        return None

    if df is None or len(df) < MIN_BARS_REQUIRED:
        return None

    levels = get_opening_range_levels(df)

    if levels is None:
        return None

    latest = df.iloc[-1]
    prev = df.iloc[-2]

    price = float(latest["close"])

    if price < MIN_PRICE:
        return None

    opening_high = float(levels["opening_high"])
    opening_low = float(levels["opening_low"])
    opening_range_pct = float(levels["opening_range_pct"])

    if opening_range_pct < 0.0015:
        return None

    if opening_range_pct > 0.035:
        return None

    ema3 = float(latest["ema3"])
    ema8 = float(latest["ema8"])
    ema21 = float(latest["ema21"])
    session_vwap = float(latest["session_vwap"])
    atr = float(latest["atr14"]) if pd.notna(latest["atr14"]) else price * 0.004
    rsi = float(latest["rsi14"]) if pd.notna(latest["rsi14"]) else 50.0

    r1 = float(latest["r1"])
    r3 = float(latest["r3"])
    r5 = float(latest["r5"])
    r15 = float(latest["r15"]) if pd.notna(latest["r15"]) else 0.0

    volume_ratio = float(latest["volume_ratio"]) if pd.notna(latest["volume_ratio"]) else 0.0

    spy_r5 = benchmark_returns.get("spy_r5", 0.0)
    qqq_r5 = benchmark_returns.get("qqq_r5", 0.0)
    relative_strength = r5 - max(spy_r5, qqq_r5)

    breakout_level = opening_high * (1 + OPENING_RANGE_BREAKOUT_BUFFER)

    if price <= breakout_level:
        return invalid_candidate(
            symbol,
            price,
            "not_above_opening_range",
            r1,
            r3,
            r5,
            relative_strength,
            atr,
        )

    breakout_extension_pct = (price / opening_high) - 1

    if breakout_extension_pct > MAX_BREAKOUT_EXTENSION_PCT:
        return invalid_candidate(
            symbol,
            price,
            "breakout_too_extended",
            r1,
            r3,
            r5,
            relative_strength,
            atr,
        )

    extension_from_vwap_atr = abs(price - session_vwap) / atr if atr > 0 else 999

    if extension_from_vwap_atr > MAX_EXTENSION_FROM_VWAP_ATR:
        return invalid_candidate(
            symbol,
            price,
            "too_extended_from_session_vwap",
            r1,
            r3,
            r5,
            relative_strength,
            atr,
        )

    if volume_ratio < MIN_VOLUME_RATIO_TO_TRADE:
        return invalid_candidate(
            symbol,
            price,
            "volume_too_weak",
            r1,
            r3,
            r5,
            relative_strength,
            atr,
        )

    if relative_strength < MIN_RELATIVE_STRENGTH_TO_TRADE:
        return invalid_candidate(
            symbol,
            price,
            "relative_strength_too_weak",
            r1,
            r3,
            r5,
            relative_strength,
            atr,
        )

    if r3 < MIN_R3_TO_TRADE:
        return invalid_candidate(
            symbol,
            price,
            "r3_too_weak",
            r1,
            r3,
            r5,
            relative_strength,
            atr,
        )

    if r5 < MIN_R5_TO_TRADE:
        return invalid_candidate(
            symbol,
            price,
            "r5_too_weak",
            r1,
            r3,
            r5,
            relative_strength,
            atr,
        )

    if rsi > MAX_RSI_MOMENTUM:
        return invalid_candidate(
            symbol,
            price,
            "momentum_overbought",
            r1,
            r3,
            r5,
            relative_strength,
            atr,
        )

    trend_stack = ema3 > ema8 > ema21
    above_vwap = price > session_vwap
    green_bar = latest["close"] > prev["close"]
    above_opening_mid = price > ((opening_high + opening_low) / 2)

    if not trend_stack:
        return invalid_candidate(
            symbol,
            price,
            "trend_stack_failed",
            r1,
            r3,
            r5,
            relative_strength,
            atr,
        )

    if not above_vwap:
        return invalid_candidate(
            symbol,
            price,
            "below_session_vwap",
            r1,
            r3,
            r5,
            relative_strength,
            atr,
        )

    if not green_bar:
        return invalid_candidate(
            symbol,
            price,
            "not_green_confirmation_bar",
            r1,
            r3,
            r5,
            relative_strength,
            atr,
        )

    score = 0.0

    score += 4.0
    score += min(volume_ratio, 4.0)
    score += min(relative_strength * 1500, 5.0)
    score += max(0, r3) * 900
    score += max(0, r5) * 700
    score += max(0, r15) * 300

    if above_opening_mid:
        score += 1.0

    if price > opening_high:
        score += 2.0

    if price > session_vwap:
        score += 1.5

    if breakout_extension_pct <= 0.003:
        score += 1.0

    if symbol in LEVERAGED_ETFS:
        score -= 1.0

    return {
        "symbol": symbol,
        "valid": True,
        "price": price,
        "score": round(max(score, 0), 4),
        "reason": "opening_range_vwap_continuation",
        "r1": r1,
        "r3": r3,
        "r5": r5,
        "relative_strength": relative_strength,
        "atr": atr,
    }


def score_symbol(symbol, df, benchmark_returns, regime, session):
    if df is None or len(df) < MIN_BARS_REQUIRED:
        return None

    if not bar_is_fresh(df, session):
        return None

    df = add_indicators(df)

    if df.empty or len(df) < MIN_BARS_REQUIRED:
        return None

    return score_opening_range_symbol_from_indicator_df(
        symbol=symbol,
        df=df,
        benchmark_returns=benchmark_returns,
        regime=regime,
    )


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


def conflicts_with_existing(symbol, selected_symbols, held_symbols, pending_symbols):
    existing = set(selected_symbols) | set(held_symbols) | set(pending_symbols)

    if symbol in existing:
        return True

    if symbol in LEVERAGED_ETFS and len(LEVERAGED_ETFS.intersection(existing)) > 0:
        return True

    return False


def build_position_plan(candidates, positions, open_orders, equity, buying_power):
    held_symbols = set(positions.keys())
    pending_symbols = get_pending_order_symbols(open_orders)

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

        if conflicts_with_existing(symbol, [], held_symbols, pending_symbols):
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

        if symbol in LEVERAGED_ETFS:
            max_capital = min(max_capital, equity * MAX_LEVERAGED_ETF_ALLOCATION)

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


# END BLOCK 1 OF 2

# ============================================================
# Position management
# ============================================================

def get_position_qty(position):
    try:
        return abs(int(float(position.qty)))
    except Exception:
        return 0


def get_position_unrealized_plpc(position):
    try:
        return float(position.unrealized_plpc)
    except Exception:
        return 0.0


def get_position_avg_entry_price(position):
    try:
        return float(position.avg_entry_price)
    except Exception:
        return None


def get_latest_price_from_bars(symbol, bars_by_symbol):
    if symbol not in bars_by_symbol:
        return None

    try:
        return float(bars_by_symbol[symbol].iloc[-1]["close"])
    except Exception:
        return None


def sell_position_now(symbol, qty, latest_price, session, dry_run, action_label):
    if dry_run:
        print(f"{action_label} | DRY_RUN | {symbol} | qty={qty}")
        return f"DRY_RUN_{action_label}"

    cancel_open_orders_for_symbol(symbol)

    if session == "after_hours":
        submit_after_hours_limit_sell(symbol, qty, latest_price)
        return f"{action_label}_AFTER_HOURS_LIMIT_SELL_SENT"

    submit_regular_market_sell(symbol, qty)
    return f"{action_label}_REGULAR_MARKET_SELL_SENT"


def flatten_all_positions(session, regime, bars_by_symbol, equity, buying_power, dry_run, reason):
    positions = get_positions_dict()

    if not positions:
        print("EOD FLATTEN | No positions to close.")
        return

    print("EOD FLATTEN | Closing all positions.")

    if not dry_run:
        cancel_all_open_orders()

    for symbol, position in positions.items():
        qty = get_position_qty(position)

        if qty <= 0:
            continue

        latest_price = get_latest_price_from_bars(symbol, bars_by_symbol)

        if latest_price is None:
            try:
                latest_price = float(position.current_price)
            except Exception:
                latest_price = 0.0

        if dry_run:
            action = "DRY_RUN_EOD_FLATTEN"
        else:
            if session == "after_hours":
                submit_after_hours_limit_sell(symbol, qty, latest_price)
                action = "EOD_FLATTEN_AFTER_HOURS_LIMIT_SELL_SENT"
            else:
                submit_regular_market_sell(symbol, qty)
                action = "EOD_FLATTEN_REGULAR_MARKET_SELL_SENT"

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
            reason=reason,
            unrealized_plpc=get_position_unrealized_plpc(position),
        )


def manage_positions(session, regime, positions, bars_by_symbol, equity, buying_power, dry_run):
    if not positions:
        return

    print("\nPosition management:")

    for symbol, position in positions.items():
        qty = get_position_qty(position)
        unrealized_plpc = get_position_unrealized_plpc(position)
        avg_entry_price = get_position_avg_entry_price(position)
        latest_price = get_latest_price_from_bars(symbol, bars_by_symbol)

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

        if avg_entry_price is None or avg_entry_price <= 0:
            avg_entry_price = latest_price

        high_water = local_position_high_water.get(symbol, avg_entry_price)
        high_water = max(float(high_water), float(latest_price))
        local_position_high_water[symbol] = high_water

        if regime != "bullish":
            print(f"  REGIME_EXIT | {symbol} | regime={regime}")

            action = sell_position_now(
                symbol=symbol,
                qty=qty,
                latest_price=latest_price,
                session=session,
                dry_run=dry_run,
                action_label="REGIME_EXIT",
            )

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
                reason="regime_not_bullish",
                unrealized_plpc=unrealized_plpc,
            )

            continue

        if unrealized_plpc >= TRAIL_TRIGGER_PCT:
            trailing_stop_price = high_water * (1 - TRAIL_GIVEBACK_PCT)

            if latest_price <= trailing_stop_price:
                print(f"  TRAILING_EXIT | {symbol} | high={high_water:.2f} | trail={trailing_stop_price:.2f}")

                action = sell_position_now(
                    symbol=symbol,
                    qty=qty,
                    latest_price=latest_price,
                    session=session,
                    dry_run=dry_run,
                    action_label="TRAILING_EXIT",
                )

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
                    reason="trailing_profit_protection",
                    unrealized_plpc=unrealized_plpc,
                )

                continue

        if unrealized_plpc >= BREAKEVEN_TRIGGER_PCT:
            breakeven_lock_price = avg_entry_price * (1 + BREAKEVEN_LOCK_PCT)

            if latest_price <= breakeven_lock_price:
                print(f"  BREAKEVEN_LOCK_EXIT | {symbol} | lock={breakeven_lock_price:.2f}")

                action = sell_position_now(
                    symbol=symbol,
                    qty=qty,
                    latest_price=latest_price,
                    session=session,
                    dry_run=dry_run,
                    action_label="BREAKEVEN_LOCK_EXIT",
                )

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
                    reason="breakeven_lock_triggered",
                    unrealized_plpc=unrealized_plpc,
                )

                continue

        entry_time = local_position_entry_time.get(symbol)

        if entry_time is not None:
            hold_seconds = time.time() - entry_time

            if hold_seconds > MAX_HOLD_SECONDS and unrealized_plpc < MIN_TAKE_PROFIT_PCT:
                print(f"  TIME_STOP | {symbol} | held={hold_seconds:.0f}s")

                action = sell_position_now(
                    symbol=symbol,
                    qty=qty,
                    latest_price=latest_price,
                    session=session,
                    dry_run=dry_run,
                    action_label="TIME_STOP",
                )

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
                    reason="time_stop_signal_decay",
                    unrealized_plpc=unrealized_plpc,
                )

                continue

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
            f"weight={item['weight'] * 100:5.1f}"
        )


# ============================================================
# Backtesting helpers
# ============================================================

def parse_backtest_datetime(value, end_of_day=False):
    if value is None:
        return None

    value = value.strip()
    dt = datetime.fromisoformat(value)

    if len(value) <= 10:
        if end_of_day:
            dt = dt.replace(hour=16, minute=0, second=0, microsecond=0)
        else:
            dt = dt.replace(hour=9, minute=30, second=0, microsecond=0)

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ZoneInfo("America/New_York"))

    return dt.astimezone(timezone.utc)


def backtest_feed_from_name(feed_name):
    feed_name = str(feed_name).upper()

    try:
        return getattr(DataFeed, feed_name)
    except Exception:
        print(f"Unknown feed '{feed_name}', using IEX.")
        return DataFeed.IEX


def bt_is_regular_session(ts):
    et = timestamp_to_et(ts)

    if et.weekday() >= 5:
        return False

    after_open = (et.hour, et.minute) >= (9, 30)
    before_close = (et.hour, et.minute) < (16, 0)

    return after_open and before_close


def bt_after_or_equal_et(ts, hour_minute):
    et = timestamp_to_et(ts)
    return (et.hour, et.minute) >= hour_minute


def bt_within_entry_window(ts):
    et = timestamp_to_et(ts)
    hm = (et.hour, et.minute)
    return TRADE_START_ET <= hm < TRADE_END_ET


def bt_apply_buy_slippage(price, slippage_bps):
    return float(price) * (1 + float(slippage_bps) / 10000.0)


def bt_apply_sell_slippage(price, slippage_bps):
    return float(price) * (1 - float(slippage_bps) / 10000.0)


def fetch_backtest_bars(symbols, start_utc, end_utc, feed):
    request = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame.Minute,
        start=start_utc,
        end=end_utc,
        feed=feed,
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
                df.index = pd.to_datetime(df.index)

                if df.index.tz is None:
                    df.index = df.index.tz_localize(timezone.utc)
                else:
                    df.index = df.index.tz_convert(timezone.utc)

                result[symbol] = df
            except Exception as e:
                print(f"Backtest data prep failed for {symbol}: {e}")
    else:
        symbol = symbols[0]
        df = bars.sort_index().copy()
        df.index = pd.to_datetime(df.index)

        if df.index.tz is None:
            df.index = df.index.tz_localize(timezone.utc)
        else:
            df.index = df.index.tz_convert(timezone.utc)

        result[symbol] = df

    return result


def prepare_backtest_data(raw_bars_by_symbol):
    prepared = {}

    for symbol, df in raw_bars_by_symbol.items():
        try:
            if df is None or df.empty:
                continue

            ind = add_indicators(df)

            if ind is None or ind.empty:
                continue

            prepared[symbol] = ind
        except Exception as e:
            print(f"Indicator prep failed for {symbol}: {e}")

    return prepared


def bt_get_window(data_by_symbol, symbol, ts, lookback_rows=420):
    df = data_by_symbol.get(symbol)

    if df is None or df.empty:
        return None

    try:
        window = df.loc[:ts].tail(lookback_rows)
    except Exception:
        return None

    if window is None or len(window) < MIN_BARS_REQUIRED:
        return None

    return window


def bt_get_row(data_by_symbol, symbol, ts):
    df = data_by_symbol.get(symbol)

    if df is None or df.empty:
        return None

    try:
        return df.loc[ts]
    except Exception:
        return None


def bt_next_timestamp_for_symbol(data_by_symbol, symbol, current_ts):
    df = data_by_symbol.get(symbol)

    if df is None or df.empty:
        return None

    idx = df.index
    pos = idx.searchsorted(current_ts, side="right")

    while pos < len(idx):
        next_ts = idx[pos]

        if bt_is_regular_session(next_ts):
            return next_ts

        pos += 1

    return None


def get_regime_backtest(data_by_symbol, ts):
    spy_df = bt_get_window(data_by_symbol, "SPY", ts)
    qqq_df = bt_get_window(data_by_symbol, "QQQ", ts)
    return get_regime_from_precomputed(spy_df, qqq_df)


def get_benchmark_returns_backtest(data_by_symbol, ts):
    spy_df = bt_get_window(data_by_symbol, "SPY", ts)
    qqq_df = bt_get_window(data_by_symbol, "QQQ", ts)
    return get_benchmark_returns_from_precomputed(spy_df, qqq_df)


def score_symbol_backtest(symbol, df, benchmark_returns, regime):
    return score_opening_range_symbol_from_indicator_df(
        symbol=symbol,
        df=df,
        benchmark_returns=benchmark_returns,
        regime=regime,
    )


def bt_symbol_on_cooldown(symbol, cooldowns, ts):
    last_ts = cooldowns.get(symbol)

    if last_ts is None:
        return False

    elapsed = (pd.Timestamp(ts) - pd.Timestamp(last_ts)).total_seconds()
    return elapsed < ORDER_COOLDOWN_SECONDS


def bt_current_symbol_price(data_by_symbol, symbol, ts, fallback_price):
    row = bt_get_row(data_by_symbol, symbol, ts)

    if row is None:
        return float(fallback_price)

    try:
        return float(row["close"])
    except Exception:
        return float(fallback_price)


def bt_calculate_equity(cash, positions, data_by_symbol, ts):
    value = float(cash)

    for symbol, pos in positions.items():
        mark = bt_current_symbol_price(data_by_symbol, symbol, ts, pos["entry_price"])
        value += pos["qty"] * mark

    return value


def bt_current_exposure(positions, data_by_symbol, ts):
    exposure = 0.0

    for symbol, pos in positions.items():
        mark = bt_current_symbol_price(data_by_symbol, symbol, ts, pos["entry_price"])
        exposure += abs(pos["qty"] * mark)

    return exposure


def bt_build_position_plan(
    candidates,
    positions,
    pending_symbols,
    equity,
    cash,
    data_by_symbol,
    ts,
    cooldowns,
):
    held_symbols = set(positions.keys())
    pending_symbols = set(pending_symbols)

    slots_left = MAX_OPEN_POSITIONS - len(held_symbols) - len(pending_symbols)

    if slots_left <= 0:
        return []

    current_exposure = bt_current_exposure(positions, data_by_symbol, ts)
    max_total_exposure_dollars = equity * MAX_TOTAL_EXPOSURE
    available_exposure = max_total_exposure_dollars - current_exposure

    usable_capital = min(available_exposure, cash)

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

        if bt_symbol_on_cooldown(symbol, cooldowns, ts):
            continue

        if conflicts_with_existing(symbol, [], held_symbols, pending_symbols):
            continue

        next_ts = bt_next_timestamp_for_symbol(data_by_symbol, symbol, ts)

        if next_ts is None:
            continue

        if bt_after_or_equal_et(next_ts, FORCE_FLATTEN_AFTER_ET):
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

        raw_weight = adjusted_score / total_adjusted_score

        max_capital = equity * MAX_POSITION_ALLOCATION

        if symbol in LEVERAGED_ETFS:
            max_capital = min(max_capital, equity * MAX_LEVERAGED_ETF_ALLOCATION)

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
            "score": float(candidate["score"]),
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


# END BLOCK 2 OF 3

def bt_record_exit(
    trades,
    position,
    symbol,
    exit_ts,
    raw_exit_price,
    exit_reason,
    slippage_bps,
):
    exit_price = bt_apply_sell_slippage(raw_exit_price, slippage_bps)

    qty = int(position["qty"])
    entry_price = float(position["entry_price"])

    gross_entry_value = qty * entry_price
    gross_exit_value = qty * exit_price
    pnl = gross_exit_value - gross_entry_value
    pnl_pct = (exit_price / entry_price) - 1

    holding_seconds = (pd.Timestamp(exit_ts) - pd.Timestamp(position["entry_time"])).total_seconds()

    trade = {
        "symbol": symbol,
        "qty": qty,
        "signal_time": position["signal_time"],
        "entry_time": position["entry_time"],
        "exit_time": exit_ts,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "take_profit": position["take_profit"],
        "stop_loss": position["stop_loss"],
        "pnl": pnl,
        "pnl_pct": pnl_pct,
        "holding_minutes": holding_seconds / 60.0,
        "exit_reason": exit_reason,
        "entry_reason": position["reason"],
        "score": position["score"],
        "r1": position["r1"],
        "r3": position["r3"],
        "r5": position["r5"],
        "relative_strength": position["relative_strength"],
    }

    trades.append(trade)

    return gross_exit_value, trade


def bt_print_report(trades, equity_curve, starting_equity, trades_file, curve_file):
    print("\n" + "=" * 90)
    print("BACKTEST REPORT")
    print("=" * 90)

    if not equity_curve:
        print("No equity curve generated.")
        return

    final_equity = float(equity_curve[-1]["equity"])
    total_return = (final_equity / starting_equity) - 1

    peak = starting_equity
    max_drawdown = 0.0

    for point in equity_curve:
        equity = float(point["equity"])
        peak = max(peak, equity)
        drawdown = (equity / peak) - 1
        max_drawdown = min(max_drawdown, drawdown)

    trade_count = len(trades)

    print(f"Starting equity:       ${starting_equity:,.2f}")
    print(f"Final equity:          ${final_equity:,.2f}")
    print(f"Total return:          {total_return * 100:.2f}%")
    print(f"Max drawdown:          {max_drawdown * 100:.2f}%")
    print(f"Trades:                {trade_count}")

    if trade_count == 0:
        print("No trades were taken.")
        print(f"Equity curve saved:    {curve_file}")
        return

    pnl_values = [float(t["pnl"]) for t in trades]
    pnl_pct_values = [float(t["pnl_pct"]) for t in trades]

    wins = [p for p in pnl_values if p > 0]
    losses = [p for p in pnl_values if p <= 0]

    win_rate = len(wins) / trade_count if trade_count else 0.0
    avg_trade = sum(pnl_values) / trade_count if trade_count else 0.0
    avg_trade_pct = sum(pnl_pct_values) / trade_count if trade_count else 0.0
    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = sum(losses) / len(losses) if losses else 0.0

    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))

    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    else:
        profit_factor = float("inf") if gross_profit > 0 else 0.0

    print(f"Win rate:              {win_rate * 100:.2f}%")
    print(f"Average trade:         ${avg_trade:,.2f}")
    print(f"Average trade pct:     {avg_trade_pct * 100:.3f}%")
    print(f"Average win:           ${avg_win:,.2f}")
    print(f"Average loss:          ${avg_loss:,.2f}")
    print(f"Profit factor:         {profit_factor:.3f}")

    df = pd.DataFrame(trades)

    print("\nExit reason summary:")
    try:
        summary = df.groupby("exit_reason").agg(
            trades=("pnl", "count"),
            pnl=("pnl", "sum"),
            avg_pnl=("pnl", "mean"),
            avg_pnl_pct=("pnl_pct", "mean"),
        )
        print(summary.to_string())
    except Exception:
        print("Could not generate exit reason summary.")

    print("\nSymbol summary:")
    try:
        symbol_summary = df.groupby("symbol").agg(
            trades=("pnl", "count"),
            pnl=("pnl", "sum"),
            avg_pnl=("pnl", "mean"),
            avg_pnl_pct=("pnl_pct", "mean"),
        ).sort_values("pnl", ascending=False)
        print(symbol_summary.to_string())
    except Exception:
        print("Could not generate symbol summary.")

    print(f"\nTrades saved:          {trades_file}")
    print(f"Equity curve saved:    {curve_file}")


def run_backtest(
    start=None,
    end=None,
    days=BACKTEST_DEFAULT_DAYS,
    starting_equity=BACKTEST_DEFAULT_STARTING_EQUITY,
    slippage_bps=BACKTEST_DEFAULT_SLIPPAGE_BPS,
    feed_name="iex",
):
    end_utc = parse_backtest_datetime(end, end_of_day=True)

    if end_utc is None:
        end_utc = datetime.now(timezone.utc)

    start_utc = parse_backtest_datetime(start, end_of_day=False)

    if start_utc is None:
        start_utc = end_utc - timedelta(days=int(days))

    fetch_start_utc = start_utc - timedelta(days=LOOKBACK_DAYS + 2)
    feed = backtest_feed_from_name(feed_name)

    print("============================================================")
    print("AuraTrader V2.4 - Historical Backtest")
    print("============================================================")
    print(f"Backtest start UTC:    {start_utc}")
    print(f"Backtest end UTC:      {end_utc}")
    print(f"Warmup fetch UTC:      {fetch_start_utc}")
    print(f"Starting equity:       ${starting_equity:,.2f}")
    print(f"Slippage:              {slippage_bps:.2f} bps per side")
    print(f"Feed:                  {feed}")
    print(f"Symbols scanned:       {len(ALL_SYMBOLS)}")
    print(f"Max open positions:    {MAX_OPEN_POSITIONS}")
    print(f"Max total exposure:    {MAX_TOTAL_EXPOSURE * 100:.0f}%")
    print(f"Max one position:      {MAX_POSITION_ALLOCATION * 100:.0f}%")
    print(f"Min score to trade:    {MIN_SCORE_TO_TRADE}")
    print(f"Entry window ET:       {TRADE_START_ET[0]:02d}:{TRADE_START_ET[1]:02d}-{TRADE_END_ET[0]:02d}:{TRADE_END_ET[1]:02d}")
    print("============================================================")

    raw_bars = fetch_backtest_bars(
        symbols=ALL_SYMBOLS,
        start_utc=fetch_start_utc,
        end_utc=end_utc,
        feed=feed,
    )

    if not raw_bars:
        print("No historical data returned.")
        return

    data_by_symbol = prepare_backtest_data(raw_bars)

    if "SPY" not in data_by_symbol or "QQQ" not in data_by_symbol:
        print("Backtest cannot run: missing SPY or QQQ data.")
        return

    all_timestamps = sorted(
        set().union(*[set(df.index) for df in data_by_symbol.values()])
    )

    timestamps = [
        ts for ts in all_timestamps
        if ts >= start_utc and ts <= end_utc and bt_is_regular_session(ts)
    ]

    if not timestamps:
        print("No regular-session timestamps found in backtest range.")
        return

    cash = float(starting_equity)
    positions = {}
    pending_entries = []
    cooldowns = {}
    trades = []
    equity_curve = []

    for ts in timestamps:
        regime, regime_reason = get_regime_backtest(data_by_symbol, ts)

        still_pending = []

        for entry in pending_entries:
            if pd.Timestamp(entry["entry_ts"]) > pd.Timestamp(ts):
                still_pending.append(entry)
                continue

            symbol = entry["symbol"]

            if symbol in positions:
                continue

            row = bt_get_row(data_by_symbol, symbol, entry["entry_ts"])

            if row is None:
                continue

            raw_entry_price = float(row["open"])
            signal_price = float(entry["price"])

            if raw_entry_price > signal_price * (1 + MAX_ENTRY_CHASE_PCT):
                continue

            entry_price = bt_apply_buy_slippage(raw_entry_price, slippage_bps)

            if entry_price <= 0:
                continue

            qty = int(entry["qty"])
            max_affordable_qty = int(cash // entry_price)
            qty = min(qty, max_affordable_qty)

            if qty <= 0:
                continue

            take_profit, stop_loss = compute_bracket_prices(entry_price, entry["atr"])

            cost = qty * entry_price
            cash -= cost

            positions[symbol] = {
                "symbol": symbol,
                "qty": qty,
                "signal_time": entry["signal_time"],
                "entry_time": entry["entry_ts"],
                "entry_price": entry_price,
                "take_profit": take_profit,
                "stop_loss": stop_loss,
                "dynamic_stop": stop_loss,
                "high_water": entry_price,
                "atr": entry["atr"],
                "score": entry["score"],
                "reason": entry["reason"],
                "r1": entry["r1"],
                "r3": entry["r3"],
                "r5": entry["r5"],
                "relative_strength": entry["relative_strength"],
            }

        pending_entries = still_pending

        symbols_to_close = []

        for symbol, position in list(positions.items()):
            row = bt_get_row(data_by_symbol, symbol, ts)

            if row is None:
                continue

            high = float(row["high"])
            low = float(row["low"])
            close = float(row["close"])

            entry_price = float(position["entry_price"])

            position["high_water"] = max(float(position["high_water"]), high)

            if position["high_water"] >= entry_price * (1 + BREAKEVEN_TRIGGER_PCT):
                breakeven_stop = entry_price * (1 + BREAKEVEN_LOCK_PCT)
                position["dynamic_stop"] = max(float(position["dynamic_stop"]), breakeven_stop)

            if position["high_water"] >= entry_price * (1 + TRAIL_TRIGGER_PCT):
                trailing_stop = position["high_water"] * (1 - TRAIL_GIVEBACK_PCT)
                position["dynamic_stop"] = max(float(position["dynamic_stop"]), trailing_stop)

            exit_reason = None
            raw_exit_price = None

            if low <= float(position["dynamic_stop"]):
                if float(position["dynamic_stop"]) > float(position["stop_loss"]):
                    exit_reason = "dynamic_stop_hit"
                else:
                    exit_reason = "stop_loss_hit"

                raw_exit_price = float(position["dynamic_stop"])

            elif high >= float(position["take_profit"]):
                exit_reason = "take_profit_hit"
                raw_exit_price = float(position["take_profit"])

            elif bt_after_or_equal_et(ts, FORCE_FLATTEN_AFTER_ET):
                exit_reason = "eod_flatten"
                raw_exit_price = close

            elif regime != "bullish":
                exit_reason = "regime_exit"
                raw_exit_price = close

            else:
                hold_seconds = (pd.Timestamp(ts) - pd.Timestamp(position["entry_time"])).total_seconds()
                unrealized_pct = (close / entry_price) - 1

                if hold_seconds >= MAX_HOLD_SECONDS and unrealized_pct < MIN_TAKE_PROFIT_PCT:
                    exit_reason = "time_stop"
                    raw_exit_price = close

            if exit_reason is not None:
                proceeds, trade = bt_record_exit(
                    trades=trades,
                    position=position,
                    symbol=symbol,
                    exit_ts=ts,
                    raw_exit_price=raw_exit_price,
                    exit_reason=exit_reason,
                    slippage_bps=slippage_bps,
                )

                cash += proceeds
                symbols_to_close.append(symbol)

        for symbol in symbols_to_close:
            positions.pop(symbol, None)

        equity = bt_calculate_equity(cash, positions, data_by_symbol, ts)

        equity_curve.append({
            "timestamp": ts,
            "equity": equity,
            "cash": cash,
            "open_positions": len(positions),
            "regime": regime,
            "regime_reason": regime_reason,
        })

        if not bt_within_entry_window(ts):
            continue

        if regime != "bullish":
            continue

        benchmark_returns = get_benchmark_returns_backtest(data_by_symbol, ts)

        candidates = []

        for symbol in TRADE_SYMBOLS:
            df_window = bt_get_window(data_by_symbol, symbol, ts)

            if df_window is None:
                continue

            result = score_symbol_backtest(
                symbol=symbol,
                df=df_window,
                benchmark_returns=benchmark_returns,
                regime=regime,
            )

            if result is None:
                continue

            if result.get("valid"):
                candidates.append(result)

        if not candidates:
            continue

        pending_symbols = {entry["symbol"] for entry in pending_entries}

        position_plan = bt_build_position_plan(
            candidates=candidates,
            positions=positions,
            pending_symbols=pending_symbols,
            equity=equity,
            cash=cash,
            data_by_symbol=data_by_symbol,
            ts=ts,
            cooldowns=cooldowns,
        )

        if not position_plan:
            continue

        for item in position_plan:
            symbol = item["symbol"]

            if symbol in positions:
                continue

            if symbol in pending_symbols:
                continue

            next_ts = bt_next_timestamp_for_symbol(data_by_symbol, symbol, ts)

            if next_ts is None:
                continue

            if bt_after_or_equal_et(next_ts, FORCE_FLATTEN_AFTER_ET):
                continue

            pending_entries.append({
                "symbol": symbol,
                "signal_time": ts,
                "entry_ts": next_ts,
                "qty": item["qty"],
                "price": item["price"],
                "score": item["score"],
                "reason": item["reason"],
                "r1": item["r1"],
                "r3": item["r3"],
                "r5": item["r5"],
                "relative_strength": item["relative_strength"],
                "atr": item["atr"],
                "allocation": item["allocation"],
            })

            cooldowns[symbol] = ts
            pending_symbols.add(symbol)

    if timestamps:
        final_ts = timestamps[-1]

        for symbol, position in list(positions.items()):
            row = bt_get_row(data_by_symbol, symbol, final_ts)

            if row is None:
                raw_exit_price = position["entry_price"]
            else:
                raw_exit_price = float(row["close"])

            proceeds, trade = bt_record_exit(
                trades=trades,
                position=position,
                symbol=symbol,
                exit_ts=final_ts,
                raw_exit_price=raw_exit_price,
                exit_reason="final_backtest_flatten",
                slippage_bps=slippage_bps,
            )

            cash += proceeds
            positions.pop(symbol, None)

        final_equity = bt_calculate_equity(cash, positions, data_by_symbol, final_ts)

        equity_curve.append({
            "timestamp": final_ts,
            "equity": final_equity,
            "cash": cash,
            "open_positions": len(positions),
            "regime": "final",
            "regime_reason": "final_backtest_flatten",
        })

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    trades_file = f"backtest_trades_{stamp}.csv"
    curve_file = f"backtest_equity_curve_{stamp}.csv"

    if trades:
        pd.DataFrame(trades).to_csv(trades_file, index=False)
    else:
        pd.DataFrame(columns=[
            "symbol",
            "qty",
            "signal_time",
            "entry_time",
            "exit_time",
            "entry_price",
            "exit_price",
            "take_profit",
            "stop_loss",
            "pnl",
            "pnl_pct",
            "holding_minutes",
            "exit_reason",
            "entry_reason",
            "score",
            "r1",
            "r3",
            "r5",
            "relative_strength",
        ]).to_csv(trades_file, index=False)

    pd.DataFrame(equity_curve).to_csv(curve_file, index=False)

    bt_print_report(
        trades=trades,
        equity_curve=equity_curve,
        starting_equity=starting_equity,
        trades_file=trades_file,
        curve_file=curve_file,
    )


# ============================================================
# Main live bot
# ============================================================

def run_bot(dry_run=True, after_hours=False, ignore_market_hours=False, ignore_freshness=False):
    init_log()

    account = trading_client.get_account()
    starting_equity = float(account.equity)

    print("============================================================")
    print("AuraTrader V2.4 - Opening Range VWAP Continuation Scalper")
    print("============================================================")
    print(f"Mode: {'DRY RUN' if dry_run else 'PAPER ORDERS'}")
    print(f"After-hours enabled: {after_hours}")
    print(f"After-hours new entries: {AFTER_HOURS_NEW_ENTRIES}")
    print(f"Starting equity: ${starting_equity:,.2f}")
    print(f"Symbols scanned: {len(ALL_SYMBOLS)}")
    print(f"Max open positions: {MAX_OPEN_POSITIONS}")
    print(f"Max total exposure: {MAX_TOTAL_EXPOSURE * 100:.0f}%")
    print(f"Max one position: {MAX_POSITION_ALLOCATION * 100:.0f}%")
    print(f"Max leveraged ETF position: {MAX_LEVERAGED_ETF_ALLOCATION * 100:.0f}%")
    print(f"Min score to trade: {MIN_SCORE_TO_TRADE}")
    print(f"Entry window ET: {TRADE_START_ET[0]:02d}:{TRADE_START_ET[1]:02d}-{TRADE_END_ET[0]:02d}:{TRADE_END_ET[1]:02d}")
    print(f"Force flatten after ET: {FORCE_FLATTEN_AFTER_ET[0]:02d}:{FORCE_FLATTEN_AFTER_ET[1]:02d}")
    print(f"Max hold seconds: {MAX_HOLD_SECONDS}")
    print(f"Log file: {LOG_FILE}")
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
                    print("After-hours flag is active, but current time is not 4:00 PM-8:00 PM ET.")
                time.sleep(60)
                continue

            if session == "after_hours":
                print("AFTER-HOURS MODE | Managing existing positions with synthetic exits.")

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

            if session == "regular" and past_time_et(*FORCE_FLATTEN_AFTER_ET):
                flatten_all_positions(
                    session=session,
                    regime=regime,
                    bars_by_symbol=bars_by_symbol,
                    equity=equity,
                    buying_power=buying_power,
                    dry_run=dry_run,
                    reason="forced_eod_flatten_no_overnight",
                )
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

            if session == "after_hours" and not AFTER_HOURS_NEW_ENTRIES:
                print("AFTER-HOURS | New entries disabled. Managing existing positions only.")
                time.sleep(LOOP_SECONDS)
                continue

            if session == "regular" and not within_entry_window_now():
                print("No new buys: outside entry window.")
                time.sleep(LOOP_SECONDS)
                continue

            if regime != "bullish":
                print("No new buys: regime not bullish.")
                time.sleep(LOOP_SECONDS)
                continue

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

            print(f"Mode=opening_range_vwap_continuation | allowed symbols={len(TRADE_SYMBOLS)}")

            benchmark_returns = get_benchmark_returns(bars_by_symbol)

            candidates = []

            for symbol in TRADE_SYMBOLS:
                try:
                    df = bars_by_symbol.get(symbol)

                    if df is None:
                        continue

                    if not ignore_freshness and not bar_is_fresh(df, session):
                        continue

                    result = score_symbol(
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

                latest_bar_price = get_latest_price_from_bars(symbol, bars_by_symbol)

                if latest_bar_price is not None and latest_bar_price > float(price) * (1 + MAX_ENTRY_CHASE_PCT):
                    print(f"SKIP | {symbol} | chased_too_far_from_signal")
                    continue

                tradeable_quote, quote_reason = quote_is_tradeable(symbol, session)

                if not tradeable_quote:
                    print(f"SKIP | {symbol} | {quote_reason}")

                    log_event(
                        session=session,
                        regime=regime,
                        symbol=symbol,
                        price=price,
                        signal="BUY_REJECTED",
                        qty=0,
                        equity=equity,
                        buying_power=buying_power,
                        action="QUOTE_FILTER_REJECTED",
                        reason=quote_reason,
                        score=item["score"],
                        allocation=item["allocation"],
                        r1=item["r1"],
                        r3=item["r3"],
                        r5=item["r5"],
                        relative_strength=item["relative_strength"],
                    )

                    continue

                if dry_run:
                    action = "DRY_RUN_BUY"
                    last_order_time_by_symbol[symbol] = time.time()
                    local_position_entry_time[symbol] = time.time()
                    local_position_high_water[symbol] = float(price)

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
                    f"score={item['score']:.3f} | reason={item['reason']} | {quote_reason}"
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
                    reason=f"{item['reason']}|{quote_reason}",
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

# ============================================================
# AURA TRADER V2.6 ADAPTIVE AGGRESSIVE SCALP OVERRIDE
# Paste this directly above the "# CLI" section.
#
# Goal:
#   - Keep high turnover
#   - Keep high exposure
#   - Stop trading symbols that are losing
#   - Stop trading after loss streaks
#   - Trade fast momentum only when recent symbol stats allow it
# ============================================================

LOG_SCHEMA_VERSION = "v2_6_adaptive_aggressive"
LOG_FILE = f"trades_{LOG_SCHEMA_VERSION}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"


# ============================================================
# Aggressive settings, but with adaptive gates
# ============================================================

TRADE_SYMBOLS = [
    "SPY", "QQQ", "IWM", "DIA",
    "TQQQ", "SQQQ", "SOXL", "SOXS", "SPXL", "SPXS", "TNA", "TZA",

    "NVDA", "TSLA", "AMD", "AAPL", "MSFT", "META", "AMZN", "GOOGL", "GOOG",
    "AVGO", "MU", "INTC", "QCOM", "MRVL", "AMAT", "LRCX", "KLAC", "SMCI",

    "PLTR", "COIN", "MSTR", "NFLX", "UBER", "SHOP", "SNOW", "CRWD", "PANW",
    "NET", "DDOG", "RBLX", "HOOD", "SQ", "PYPL", "AFRM", "ROKU", "DKNG",

    "BABA", "PDD", "JPM", "BAC", "GS", "MS", "WFC",

    "XOM", "CVX", "OXY", "SLB", "BA", "GE", "CAT", "DE",

    "LLY", "NVO", "UNH", "MRK", "PFE", "ABBV",

    "WMT", "COST", "TGT", "HD", "LOW", "DIS", "NKE", "SBUX",
]

INVERSE_SYMBOLS = ["SQQQ", "SOXS", "SPXS", "TZA"]
LEVERAGED_ETFS = {"TQQQ", "SQQQ", "SOXL", "SOXS", "SPXL", "SPXS", "TNA", "TZA"}
ALL_SYMBOLS = sorted(list(set(TRADE_SYMBOLS + REGIME_SYMBOLS)))

MAX_OPEN_POSITIONS = 5

MAX_TOTAL_EXPOSURE = 0.90
MAX_POSITION_ALLOCATION = 0.22
MIN_POSITION_ALLOCATION = 0.035
MAX_LEVERAGED_ETF_ALLOCATION = 0.16

MIN_SCORE_TO_TRADE = 8.0
SCORE_POWER = 1.10

ORDER_COOLDOWN_SECONDS = 180

TRADE_START_ET = (9, 35)
TRADE_END_ET = (15, 35)
FORCE_FLATTEN_AFTER_ET = (15, 55)

DAILY_KILL_SWITCH = 0.06

MIN_VOLUME_RATIO_TO_TRADE = 1.15
MIN_RELATIVE_STRENGTH_TO_TRADE = 0.0002
MIN_R3_TO_TRADE = 0.0002
MIN_R5_TO_TRADE = 0.0000

MAX_EXTENSION_FROM_VWAP_ATR = 1.65
MAX_RSI_MOMENTUM = 76

MIN_TAKE_PROFIT_PCT = 0.0035
MAX_TAKE_PROFIT_PCT = 0.0065
MIN_STOP_LOSS_PCT = 0.0018
MAX_STOP_LOSS_PCT = 0.0032

ATR_TAKE_PROFIT_MULT = 0.75
ATR_STOP_LOSS_MULT = 0.35

BREAKEVEN_TRIGGER_PCT = 0.0022
BREAKEVEN_LOCK_PCT = 0.0004

TRAIL_TRIGGER_PCT = 0.0040
TRAIL_GIVEBACK_PCT = 0.0018

MAX_HOLD_SECONDS = 8 * 60
MAX_ENTRY_CHASE_PCT = 0.0015

BACKTEST_DEFAULT_DAYS = 30
BACKTEST_DEFAULT_SLIPPAGE_BPS = 5.0


# ============================================================
# Adaptive gates
# ============================================================

ADAPTIVE_MIN_TRADES_BEFORE_GATE = 4
ADAPTIVE_MIN_WIN_RATE = 0.38
ADAPTIVE_MIN_PROFIT_FACTOR = 0.95
ADAPTIVE_DISABLE_SYMBOL_AFTER_DAILY_LOSSES = 2
ADAPTIVE_DISABLE_SYMBOL_AFTER_TOTAL_LOSSES = 5
ADAPTIVE_GLOBAL_LOSS_STREAK_LIMIT = 6
ADAPTIVE_MAX_DAILY_REALIZED_LOSS_PCT = 0.025

# Seed from your latest 7-day run: these were the only symbols with positive P/L.
# Other symbols are allowed, but they are cut quickly by the adaptive gate.
ADAPTIVE_PREFERRED_SYMBOLS = {
    "NET", "AFRM", "TZA", "AVGO", "HD", "ABBV", "PDD", "XOM", "AMD", "TGT"
}


def make_adaptive_symbol_stats():
    return {
        "trades": 0,
        "wins": 0,
        "losses": 0,
        "pnl": 0.0,
        "gross_profit": 0.0,
        "gross_loss": 0.0,
    }


def make_adaptive_day_stats():
    return {
        "trades": 0,
        "wins": 0,
        "losses": 0,
        "pnl": 0.0,
    }


def adaptive_profit_factor(stats):
    gross_loss = abs(float(stats.get("gross_loss", 0.0)))

    if gross_loss <= 0:
        return 999.0 if float(stats.get("gross_profit", 0.0)) > 0 else 0.0

    return float(stats.get("gross_profit", 0.0)) / gross_loss


def adaptive_win_rate(stats):
    trades = int(stats.get("trades", 0))

    if trades <= 0:
        return 0.0

    return int(stats.get("wins", 0)) / trades


def adaptive_symbol_allowed(symbol, symbol_stats, day_symbol_stats, day_key):
    stats = symbol_stats.setdefault(symbol, make_adaptive_symbol_stats())
    day_stats = day_symbol_stats.setdefault((day_key, symbol), make_adaptive_day_stats())

    if int(day_stats["losses"]) >= ADAPTIVE_DISABLE_SYMBOL_AFTER_DAILY_LOSSES:
        return False, "symbol_disabled_daily_losses"

    if int(stats["losses"]) >= ADAPTIVE_DISABLE_SYMBOL_AFTER_TOTAL_LOSSES and float(stats["pnl"]) < 0:
        return False, "symbol_disabled_total_losses"

    trades = int(stats["trades"])

    if trades < ADAPTIVE_MIN_TRADES_BEFORE_GATE:
        return True, "adaptive_warmup"

    wr = adaptive_win_rate(stats)
    pf = adaptive_profit_factor(stats)

    if float(stats["pnl"]) < 0 and wr < ADAPTIVE_MIN_WIN_RATE:
        return False, "symbol_failed_winrate_gate"

    if float(stats["pnl"]) < 0 and pf < ADAPTIVE_MIN_PROFIT_FACTOR:
        return False, "symbol_failed_profit_factor_gate"

    return True, "adaptive_passed"


def adaptive_update_after_trade(trade, symbol_stats, day_symbol_stats, adaptive_state, starting_equity):
    symbol = trade["symbol"]
    pnl = float(trade["pnl"])
    day_key = timestamp_to_et(trade["exit_time"]).date()

    stats = symbol_stats.setdefault(symbol, make_adaptive_symbol_stats())
    day_stats = day_symbol_stats.setdefault((day_key, symbol), make_adaptive_day_stats())

    stats["trades"] += 1
    day_stats["trades"] += 1

    stats["pnl"] += pnl
    day_stats["pnl"] += pnl

    adaptive_state["daily_realized_pnl_by_date"][day_key] = (
        adaptive_state["daily_realized_pnl_by_date"].get(day_key, 0.0) + pnl
    )

    if pnl > 0:
        stats["wins"] += 1
        stats["gross_profit"] += pnl
        day_stats["wins"] += 1
        adaptive_state["loss_streak"] = 0
    else:
        stats["losses"] += 1
        stats["gross_loss"] += pnl
        day_stats["losses"] += 1
        adaptive_state["loss_streak"] += 1

    daily_pnl = adaptive_state["daily_realized_pnl_by_date"].get(day_key, 0.0)

    if daily_pnl <= -starting_equity * ADAPTIVE_MAX_DAILY_REALIZED_LOSS_PCT:
        adaptive_state["disabled_day"] = day_key

    if adaptive_state["loss_streak"] >= ADAPTIVE_GLOBAL_LOSS_STREAK_LIMIT:
        adaptive_state["disabled_day"] = day_key


def adaptive_global_trading_allowed(ts, adaptive_state):
    day_key = timestamp_to_et(ts).date()

    if adaptive_state.get("disabled_day") == day_key:
        return False, "adaptive_global_disabled_for_day"

    return True, "adaptive_global_allowed"


# ============================================================
# Regime override
# ============================================================

def get_regime_from_precomputed(spy_df, qqq_df):
    if spy_df is None or qqq_df is None:
        return "unknown", "missing_SPY_or_QQQ"

    if len(spy_df) < MIN_BARS_REQUIRED or len(qqq_df) < MIN_BARS_REQUIRED:
        return "unknown", "insufficient_regime_data"

    s = spy_df.iloc[-1]
    q = qqq_df.iloc[-1]

    spy_price = float(s["close"])
    qqq_price = float(q["close"])

    spy_bull = (
        spy_price > float(s["ema8"])
        and spy_price > float(s["session_vwap"])
        and float(s["r3"]) > -0.0005
    )

    qqq_bull = (
        qqq_price > float(q["ema8"])
        and qqq_price > float(q["session_vwap"])
        and float(q["r3"]) > -0.0005
    )

    spy_bear = (
        spy_price < float(s["ema8"])
        and spy_price < float(s["session_vwap"])
        and float(s["r5"]) < -0.001
    )

    qqq_bear = (
        qqq_price < float(q["ema8"])
        and qqq_price < float(q["session_vwap"])
        and float(q["r5"]) < -0.001
    )

    if spy_bull or qqq_bull:
        return "bullish", "risk_on_momentum"

    if spy_bear and qqq_bear:
        return "bearish", "risk_off_inverse_allowed"

    return "choppy", "mixed_market_selective"


def compute_bracket_prices(entry_price, atr_value):
    price = float(entry_price)
    atr = float(atr_value) if atr_value and atr_value > 0 else price * 0.003

    atr_pct = atr / price

    take_profit_pct = ATR_TAKE_PROFIT_MULT * atr_pct
    stop_loss_pct = ATR_STOP_LOSS_MULT * atr_pct

    take_profit_pct = max(MIN_TAKE_PROFIT_PCT, min(MAX_TAKE_PROFIT_PCT, take_profit_pct))
    stop_loss_pct = max(MIN_STOP_LOSS_PCT, min(MAX_STOP_LOSS_PCT, stop_loss_pct))

    take_profit_price = price * (1 + take_profit_pct)
    stop_loss_price = price * (1 - stop_loss_pct)

    return take_profit_price, stop_loss_price


def conflicts_with_existing(symbol, selected_symbols, held_symbols, pending_symbols):
    existing = set(selected_symbols) | set(held_symbols) | set(pending_symbols)
    return symbol in existing


# ============================================================
# Scoring override
# ============================================================

def score_adaptive_scalp_symbol_from_indicator_df(symbol, df, benchmark_returns, regime):
    if df is None or len(df) < MIN_BARS_REQUIRED:
        return None

    latest = df.iloc[-1]
    prev = df.iloc[-2]

    price = float(latest["close"])

    if price < MIN_PRICE:
        return None

    ema3 = float(latest["ema3"])
    ema8 = float(latest["ema8"])
    ema21 = float(latest["ema21"])

    session_vwap = float(latest["session_vwap"])
    rolling_vwap = float(latest["rolling_vwap20"])

    atr = float(latest["atr14"]) if pd.notna(latest["atr14"]) else price * 0.003
    rsi = float(latest["rsi14"]) if pd.notna(latest["rsi14"]) else 50.0

    r1 = float(latest["r1"])
    r3 = float(latest["r3"])
    r5 = float(latest["r5"])
    r15 = float(latest["r15"]) if pd.notna(latest["r15"]) else 0.0

    volume_ratio = float(latest["volume_ratio"]) if pd.notna(latest["volume_ratio"]) else 0.0

    spy_r3 = benchmark_returns.get("spy_r3", 0.0)
    spy_r5 = benchmark_returns.get("spy_r5", 0.0)
    qqq_r3 = benchmark_returns.get("qqq_r3", 0.0)
    qqq_r5 = benchmark_returns.get("qqq_r5", 0.0)

    benchmark_r3 = max(spy_r3, qqq_r3)
    benchmark_r5 = max(spy_r5, qqq_r5)

    is_inverse = symbol in INVERSE_SYMBOLS

    if is_inverse:
        if regime != "bearish":
            return invalid_candidate(symbol, price, "inverse_only_in_bearish_regime", r1, r3, r5, r3, atr)
        relative_strength = r3
    else:
        if regime == "bearish":
            return invalid_candidate(symbol, price, "long_blocked_in_bearish_regime", r1, r3, r5, 0, atr)
        relative_strength = (r3 - benchmark_r3) + 0.50 * (r5 - benchmark_r5)

    green_bar = float(latest["close"]) > float(prev["close"])
    above_ema3 = price > ema3
    fast_stack = ema3 > ema8
    medium_stack = ema8 > ema21
    above_vwap = price > session_vwap or price > rolling_vwap

    extension_from_vwap_atr = abs(price - session_vwap) / atr if atr > 0 else 999

    if volume_ratio < MIN_VOLUME_RATIO_TO_TRADE:
        return invalid_candidate(symbol, price, "volume_too_weak", r1, r3, r5, relative_strength, atr)

    if relative_strength < MIN_RELATIVE_STRENGTH_TO_TRADE:
        return invalid_candidate(symbol, price, "relative_strength_too_weak", r1, r3, r5, relative_strength, atr)

    if r3 < MIN_R3_TO_TRADE:
        return invalid_candidate(symbol, price, "r3_too_weak", r1, r3, r5, relative_strength, atr)

    if r5 < MIN_R5_TO_TRADE and r15 < 0:
        return invalid_candidate(symbol, price, "r5_r15_too_weak", r1, r3, r5, relative_strength, atr)

    if rsi > MAX_RSI_MOMENTUM:
        return invalid_candidate(symbol, price, "rsi_too_hot", r1, r3, r5, relative_strength, atr)

    if extension_from_vwap_atr > MAX_EXTENSION_FROM_VWAP_ATR:
        return invalid_candidate(symbol, price, "too_extended_from_vwap", r1, r3, r5, relative_strength, atr)

    valid = (
        above_ema3
        and fast_stack
        and above_vwap
        and r1 > -0.0002
        and r3 > 0
        and (green_bar or r3 > 0.001)
    )

    if not valid:
        return invalid_candidate(symbol, price, "no_adaptive_scalp_setup", r1, r3, r5, relative_strength, atr)

    score = 0.0

    score += 1.25 if above_ema3 else 0
    score += 1.50 if fast_stack else 0
    score += 0.75 if medium_stack else 0
    score += 1.25 if above_vwap else 0
    score += 0.75 if green_bar else 0

    score += min(volume_ratio, 3.5)
    score += max(0, relative_strength) * 1600
    score += max(0, r1) * 1000
    score += max(0, r3) * 850
    score += max(0, r5) * 550
    score += max(0, r15) * 250

    if symbol in ADAPTIVE_PREFERRED_SYMBOLS:
        score += 1.25

    if symbol in LEVERAGED_ETFS:
        score += 0.25

    if 45 <= rsi <= 68:
        score += 0.75

    if extension_from_vwap_atr <= 1.00:
        score += 0.75

    return {
        "symbol": symbol,
        "valid": True,
        "price": price,
        "score": round(max(score, 0), 4),
        "reason": "v2_6_adaptive_aggressive_scalp",
        "r1": r1,
        "r3": r3,
        "r5": r5,
        "relative_strength": relative_strength,
        "atr": atr,
    }


def score_symbol(symbol, df, benchmark_returns, regime, session):
    if df is None or len(df) < MIN_BARS_REQUIRED:
        return None

    if not bar_is_fresh(df, session):
        return None

    df = add_indicators(df)

    if df.empty or len(df) < MIN_BARS_REQUIRED:
        return None

    return score_adaptive_scalp_symbol_from_indicator_df(
        symbol=symbol,
        df=df,
        benchmark_returns=benchmark_returns,
        regime=regime,
    )


def score_symbol_backtest(symbol, df, benchmark_returns, regime):
    return score_adaptive_scalp_symbol_from_indicator_df(
        symbol=symbol,
        df=df,
        benchmark_returns=benchmark_returns,
        regime=regime,
    )


# ============================================================
# Backtest override with adaptive gates
# ============================================================

def run_backtest(
    start=None,
    end=None,
    days=BACKTEST_DEFAULT_DAYS,
    starting_equity=BACKTEST_DEFAULT_STARTING_EQUITY,
    slippage_bps=BACKTEST_DEFAULT_SLIPPAGE_BPS,
    feed_name="iex",
):
    end_utc = parse_backtest_datetime(end, end_of_day=True)

    if end_utc is None:
        end_utc = datetime.now(timezone.utc)

    start_utc = parse_backtest_datetime(start, end_of_day=False)

    if start_utc is None:
        start_utc = end_utc - timedelta(days=int(days))

    fetch_start_utc = start_utc - timedelta(days=LOOKBACK_DAYS + 2)
    feed = backtest_feed_from_name(feed_name)

    print("============================================================")
    print("AuraTrader V2.6 - Adaptive Aggressive Backtest")
    print("============================================================")
    print(f"Backtest start UTC:    {start_utc}")
    print(f"Backtest end UTC:      {end_utc}")
    print(f"Starting equity:       ${starting_equity:,.2f}")
    print(f"Slippage:              {slippage_bps:.2f} bps per side")
    print(f"Max exposure:          {MAX_TOTAL_EXPOSURE * 100:.0f}%")
    print(f"Max open positions:    {MAX_OPEN_POSITIONS}")
    print(f"Symbols scanned:       {len(ALL_SYMBOLS)}")
    print("============================================================")

    raw_bars = fetch_backtest_bars(
        symbols=ALL_SYMBOLS,
        start_utc=fetch_start_utc,
        end_utc=end_utc,
        feed=feed,
    )

    if not raw_bars:
        print("No historical data returned.")
        return

    data_by_symbol = prepare_backtest_data(raw_bars)

    if "SPY" not in data_by_symbol or "QQQ" not in data_by_symbol:
        print("Backtest cannot run: missing SPY or QQQ data.")
        return

    all_timestamps = sorted(
        set().union(*[set(df.index) for df in data_by_symbol.values()])
    )

    timestamps = [
        ts for ts in all_timestamps
        if ts >= start_utc and ts <= end_utc and bt_is_regular_session(ts)
    ]

    if not timestamps:
        print("No regular-session timestamps found in backtest range.")
        return

    cash = float(starting_equity)
    positions = {}
    pending_entries = []
    cooldowns = {}
    trades = []
    equity_curve = []

    symbol_stats = {}
    day_symbol_stats = {}

    adaptive_state = {
        "loss_streak": 0,
        "disabled_day": None,
        "daily_realized_pnl_by_date": {},
    }

    for ts in timestamps:
        day_key = timestamp_to_et(ts).date()
        regime, regime_reason = get_regime_backtest(data_by_symbol, ts)

        still_pending = []

        for entry in pending_entries:
            if pd.Timestamp(entry["entry_ts"]) > pd.Timestamp(ts):
                still_pending.append(entry)
                continue

            symbol = entry["symbol"]

            allowed, _ = adaptive_symbol_allowed(symbol, symbol_stats, day_symbol_stats, day_key)
            global_allowed, _ = adaptive_global_trading_allowed(ts, adaptive_state)

            if not allowed or not global_allowed:
                continue

            if symbol in positions:
                continue

            row = bt_get_row(data_by_symbol, symbol, entry["entry_ts"])

            if row is None:
                continue

            raw_entry_price = float(row["open"])
            signal_price = float(entry["price"])

            if raw_entry_price > signal_price * (1 + MAX_ENTRY_CHASE_PCT):
                continue

            entry_price = bt_apply_buy_slippage(raw_entry_price, slippage_bps)

            if entry_price <= 0:
                continue

            qty = int(entry["qty"])
            max_affordable_qty = int(cash // entry_price)
            qty = min(qty, max_affordable_qty)

            if qty <= 0:
                continue

            take_profit, stop_loss = compute_bracket_prices(entry_price, entry["atr"])

            cost = qty * entry_price
            cash -= cost

            positions[symbol] = {
                "symbol": symbol,
                "qty": qty,
                "signal_time": entry["signal_time"],
                "entry_time": entry["entry_ts"],
                "entry_price": entry_price,
                "take_profit": take_profit,
                "stop_loss": stop_loss,
                "dynamic_stop": stop_loss,
                "high_water": entry_price,
                "atr": entry["atr"],
                "score": entry["score"],
                "reason": entry["reason"],
                "r1": entry["r1"],
                "r3": entry["r3"],
                "r5": entry["r5"],
                "relative_strength": entry["relative_strength"],
            }

        pending_entries = still_pending

        symbols_to_close = []

        for symbol, position in list(positions.items()):
            row = bt_get_row(data_by_symbol, symbol, ts)

            if row is None:
                continue

            high = float(row["high"])
            low = float(row["low"])
            close = float(row["close"])

            entry_price = float(position["entry_price"])

            position["high_water"] = max(float(position["high_water"]), high)

            if position["high_water"] >= entry_price * (1 + BREAKEVEN_TRIGGER_PCT):
                breakeven_stop = entry_price * (1 + BREAKEVEN_LOCK_PCT)
                position["dynamic_stop"] = max(float(position["dynamic_stop"]), breakeven_stop)

            if position["high_water"] >= entry_price * (1 + TRAIL_TRIGGER_PCT):
                trailing_stop = position["high_water"] * (1 - TRAIL_GIVEBACK_PCT)
                position["dynamic_stop"] = max(float(position["dynamic_stop"]), trailing_stop)

            exit_reason = None
            raw_exit_price = None

            is_inverse = symbol in INVERSE_SYMBOLS

            if low <= float(position["dynamic_stop"]):
                if float(position["dynamic_stop"]) > float(position["stop_loss"]):
                    exit_reason = "dynamic_stop_hit"
                else:
                    exit_reason = "stop_loss_hit"

                raw_exit_price = float(position["dynamic_stop"])

            elif high >= float(position["take_profit"]):
                exit_reason = "take_profit_hit"
                raw_exit_price = float(position["take_profit"])

            elif bt_after_or_equal_et(ts, FORCE_FLATTEN_AFTER_ET):
                exit_reason = "eod_flatten"
                raw_exit_price = close

            elif is_inverse and regime != "bearish":
                exit_reason = "inverse_regime_exit"
                raw_exit_price = close

            elif not is_inverse and regime == "bearish":
                exit_reason = "long_regime_exit"
                raw_exit_price = close

            else:
                hold_seconds = (pd.Timestamp(ts) - pd.Timestamp(position["entry_time"])).total_seconds()
                unrealized_pct = (close / entry_price) - 1

                if hold_seconds >= MAX_HOLD_SECONDS and unrealized_pct < MIN_TAKE_PROFIT_PCT:
                    exit_reason = "time_stop"
                    raw_exit_price = close

            if exit_reason is not None:
                proceeds, trade = bt_record_exit(
                    trades=trades,
                    position=position,
                    symbol=symbol,
                    exit_ts=ts,
                    raw_exit_price=raw_exit_price,
                    exit_reason=exit_reason,
                    slippage_bps=slippage_bps,
                )

                adaptive_update_after_trade(
                    trade=trade,
                    symbol_stats=symbol_stats,
                    day_symbol_stats=day_symbol_stats,
                    adaptive_state=adaptive_state,
                    starting_equity=starting_equity,
                )

                cash += proceeds
                symbols_to_close.append(symbol)

        for symbol in symbols_to_close:
            positions.pop(symbol, None)

        equity = bt_calculate_equity(cash, positions, data_by_symbol, ts)

        equity_curve.append({
            "timestamp": ts,
            "equity": equity,
            "cash": cash,
            "open_positions": len(positions),
            "regime": regime,
            "regime_reason": regime_reason,
        })

        if not bt_within_entry_window(ts):
            continue

        global_allowed, _ = adaptive_global_trading_allowed(ts, adaptive_state)

        if not global_allowed:
            continue

        if regime not in ["bullish", "bearish", "choppy"]:
            continue

        benchmark_returns = get_benchmark_returns_backtest(data_by_symbol, ts)

        candidates = []

        for symbol in TRADE_SYMBOLS:
            allowed, adaptive_reason = adaptive_symbol_allowed(
                symbol=symbol,
                symbol_stats=symbol_stats,
                day_symbol_stats=day_symbol_stats,
                day_key=day_key,
            )

            if not allowed:
                continue

            df_window = bt_get_window(data_by_symbol, symbol, ts)

            if df_window is None:
                continue

            result = score_symbol_backtest(
                symbol=symbol,
                df=df_window,
                benchmark_returns=benchmark_returns,
                regime=regime,
            )

            if result is None:
                continue

            if result.get("valid"):
                result["reason"] = f"{result['reason']}|{adaptive_reason}"
                candidates.append(result)

        if not candidates:
            continue

        pending_symbols = {entry["symbol"] for entry in pending_entries}

        position_plan = bt_build_position_plan(
            candidates=candidates,
            positions=positions,
            pending_symbols=pending_symbols,
            equity=equity,
            cash=cash,
            data_by_symbol=data_by_symbol,
            ts=ts,
            cooldowns=cooldowns,
        )

        if not position_plan:
            continue

        for item in position_plan:
            symbol = item["symbol"]

            if symbol in positions:
                continue

            if symbol in pending_symbols:
                continue

            next_ts = bt_next_timestamp_for_symbol(data_by_symbol, symbol, ts)

            if next_ts is None:
                continue

            if bt_after_or_equal_et(next_ts, FORCE_FLATTEN_AFTER_ET):
                continue

            pending_entries.append({
                "symbol": symbol,
                "signal_time": ts,
                "entry_ts": next_ts,
                "qty": item["qty"],
                "price": item["price"],
                "score": item["score"],
                "reason": item["reason"],
                "r1": item["r1"],
                "r3": item["r3"],
                "r5": item["r5"],
                "relative_strength": item["relative_strength"],
                "atr": item["atr"],
                "allocation": item["allocation"],
            })

            cooldowns[symbol] = ts
            pending_symbols.add(symbol)

    if timestamps:
        final_ts = timestamps[-1]

        for symbol, position in list(positions.items()):
            row = bt_get_row(data_by_symbol, symbol, final_ts)

            if row is None:
                raw_exit_price = position["entry_price"]
            else:
                raw_exit_price = float(row["close"])

            proceeds, trade = bt_record_exit(
                trades=trades,
                position=position,
                symbol=symbol,
                exit_ts=final_ts,
                raw_exit_price=raw_exit_price,
                exit_reason="final_backtest_flatten",
                slippage_bps=slippage_bps,
            )

            adaptive_update_after_trade(
                trade=trade,
                symbol_stats=symbol_stats,
                day_symbol_stats=day_symbol_stats,
                adaptive_state=adaptive_state,
                starting_equity=starting_equity,
            )

            cash += proceeds
            positions.pop(symbol, None)

        final_equity = bt_calculate_equity(cash, positions, data_by_symbol, final_ts)

        equity_curve.append({
            "timestamp": final_ts,
            "equity": final_equity,
            "cash": cash,
            "open_positions": len(positions),
            "regime": "final",
            "regime_reason": "final_backtest_flatten",
        })

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    trades_file = f"backtest_trades_{stamp}.csv"
    curve_file = f"backtest_equity_curve_{stamp}.csv"

    if trades:
        pd.DataFrame(trades).to_csv(trades_file, index=False)
    else:
        pd.DataFrame(columns=[
            "symbol",
            "qty",
            "signal_time",
            "entry_time",
            "exit_time",
            "entry_price",
            "exit_price",
            "take_profit",
            "stop_loss",
            "pnl",
            "pnl_pct",
            "holding_minutes",
            "exit_reason",
            "entry_reason",
            "score",
            "r1",
            "r3",
            "r5",
            "relative_strength",
        ]).to_csv(trades_file, index=False)

    pd.DataFrame(equity_curve).to_csv(curve_file, index=False)

    bt_print_report(
        trades=trades,
        equity_curve=equity_curve,
        starting_equity=starting_equity,
        trades_file=trades_file,
        curve_file=curve_file,
    )

# ============================================================
# AURA TRADER V2.7 HIGH-TURNOVER PULLBACK SCALP OVERRIDE
# Paste below the v2.6 override and above "# CLI".
#
# Goal:
#   - Keep 90% max exposure
#   - Improve win rate
#   - Stop chasing extended candles
#   - Enter VWAP/EMA pullback reclaims instead
#   - Take smaller faster profits
# ============================================================

LOG_SCHEMA_VERSION = "v2_7_pullback_scalp"
LOG_FILE = f"trades_{LOG_SCHEMA_VERSION}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"


# ============================================================
# Universe: remove inverse ETFs and worst churn names
# ============================================================

TRADE_SYMBOLS = [
    "QQQ", "DIA",

    "AAPL", "MSFT", "AMD", "AVGO", "MU", "MRVL", "LRCX", "KLAC",
    "META", "GOOG", "GOOGL", "AMZN", "NFLX",

    "MSTR", "BABA", "PDD", "JPM", "WFC", "GS", "MS",

    "ROKU", "NET", "AFRM", "SHOP", "UBER", "SNOW", "DIS",

    "XOM", "CVX", "HD", "LOW", "TGT", "WMT", "COST",

    "BA", "GE", "DE", "CAT",

    "ABBV", "MRK", "UNH",
]

INVERSE_SYMBOLS = []
LEVERAGED_ETFS = set()
ALL_SYMBOLS = sorted(list(set(TRADE_SYMBOLS + REGIME_SYMBOLS)))


# ============================================================
# Risk / turnover settings
# ============================================================

MAX_OPEN_POSITIONS = 6

MAX_TOTAL_EXPOSURE = 0.90
MAX_POSITION_ALLOCATION = 0.18
MIN_POSITION_ALLOCATION = 0.04
MAX_LEVERAGED_ETF_ALLOCATION = 0.00

MIN_SCORE_TO_TRADE = 7.5
SCORE_POWER = 1.05

ORDER_COOLDOWN_SECONDS = 240

TRADE_START_ET = (9, 40)
TRADE_END_ET = (15, 15)
FORCE_FLATTEN_AFTER_ET = (15, 55)

DAILY_KILL_SWITCH = 0.04

MIN_VOLUME_RATIO_TO_TRADE = 1.05
MIN_RELATIVE_STRENGTH_TO_TRADE = -0.0002
MIN_R3_TO_TRADE = -0.0015
MIN_R5_TO_TRADE = -0.0020

MAX_EXTENSION_FROM_VWAP_ATR = 0.85
MAX_RSI_MOMENTUM = 68

# Smaller faster profit target.
MIN_TAKE_PROFIT_PCT = 0.0018
MAX_TAKE_PROFIT_PCT = 0.0035

# Slightly wider than target, because this version aims for higher win rate.
MIN_STOP_LOSS_PCT = 0.0022
MAX_STOP_LOSS_PCT = 0.0040

ATR_TAKE_PROFIT_MULT = 0.42
ATR_STOP_LOSS_MULT = 0.55

BREAKEVEN_TRIGGER_PCT = 0.0016
BREAKEVEN_LOCK_PCT = 0.0002

TRAIL_TRIGGER_PCT = 0.0024
TRAIL_GIVEBACK_PCT = 0.0012

MAX_HOLD_SECONDS = 10 * 60
MAX_ENTRY_CHASE_PCT = 0.0008

BACKTEST_DEFAULT_DAYS = 30
BACKTEST_DEFAULT_SLIPPAGE_BPS = 5.0


# ============================================================
# Adaptive gates: stricter
# ============================================================

ADAPTIVE_MIN_TRADES_BEFORE_GATE = 3
ADAPTIVE_MIN_WIN_RATE = 0.45
ADAPTIVE_MIN_PROFIT_FACTOR = 1.00

# This is the important change.
ADAPTIVE_DISABLE_SYMBOL_AFTER_DAILY_LOSSES = 1
ADAPTIVE_DISABLE_SYMBOL_AFTER_TOTAL_LOSSES = 3
ADAPTIVE_GLOBAL_LOSS_STREAK_LIMIT = 4
ADAPTIVE_MAX_DAILY_REALIZED_LOSS_PCT = 0.015

ADAPTIVE_PREFERRED_SYMBOLS = {
    "ROKU", "MRVL", "MU", "MSTR", "BABA", "WFC", "DIS", "JPM",
    "NET", "AFRM", "AVGO", "HD", "ABBV", "PDD", "XOM"
}


# ============================================================
# Regime: trade long only when market is not actively bearish
# ============================================================

def get_regime_from_precomputed(spy_df, qqq_df):
    if spy_df is None or qqq_df is None:
        return "unknown", "missing_SPY_or_QQQ"

    if len(spy_df) < MIN_BARS_REQUIRED or len(qqq_df) < MIN_BARS_REQUIRED:
        return "unknown", "insufficient_regime_data"

    s = spy_df.iloc[-1]
    q = qqq_df.iloc[-1]

    spy_price = float(s["close"])
    qqq_price = float(q["close"])

    spy_bear = (
        spy_price < float(s["ema8"])
        and spy_price < float(s["session_vwap"])
        and float(s["r5"]) < -0.002
    )

    qqq_bear = (
        qqq_price < float(q["ema8"])
        and qqq_price < float(q["session_vwap"])
        and float(q["r5"]) < -0.002
    )

    if spy_bear and qqq_bear:
        return "bearish", "both_indexes_bearish_no_longs"

    spy_ok = spy_price >= float(s["session_vwap"]) or float(s["r3"]) > -0.001
    qqq_ok = qqq_price >= float(q["session_vwap"]) or float(q["r3"]) > -0.001

    if spy_ok or qqq_ok:
        return "bullish", "market_allows_long_scalps"

    return "choppy", "mixed_but_tradeable_pullbacks"


def compute_bracket_prices(entry_price, atr_value):
    price = float(entry_price)
    atr = float(atr_value) if atr_value and atr_value > 0 else price * 0.003

    atr_pct = atr / price

    take_profit_pct = ATR_TAKE_PROFIT_MULT * atr_pct
    stop_loss_pct = ATR_STOP_LOSS_MULT * atr_pct

    take_profit_pct = max(MIN_TAKE_PROFIT_PCT, min(MAX_TAKE_PROFIT_PCT, take_profit_pct))
    stop_loss_pct = max(MIN_STOP_LOSS_PCT, min(MAX_STOP_LOSS_PCT, stop_loss_pct))

    take_profit_price = price * (1 + take_profit_pct)
    stop_loss_price = price * (1 - stop_loss_pct)

    return take_profit_price, stop_loss_price


def conflicts_with_existing(symbol, selected_symbols, held_symbols, pending_symbols):
    existing = set(selected_symbols) | set(held_symbols) | set(pending_symbols)
    return symbol in existing


# ============================================================
# New entry logic: VWAP/EMA pullback reclaim
# ============================================================

def score_pullback_scalp_symbol_from_indicator_df(symbol, df, benchmark_returns, regime):
    if regime == "bearish":
        return None

    if df is None or len(df) < MIN_BARS_REQUIRED:
        return None

    latest = df.iloc[-1]
    prev = df.iloc[-2]

    price = float(latest["close"])

    if price < MIN_PRICE:
        return None

    ema3 = float(latest["ema3"])
    ema8 = float(latest["ema8"])
    ema21 = float(latest["ema21"])

    session_vwap = float(latest["session_vwap"])
    rolling_vwap = float(latest["rolling_vwap20"])

    atr = float(latest["atr14"]) if pd.notna(latest["atr14"]) else price * 0.003
    rsi = float(latest["rsi14"]) if pd.notna(latest["rsi14"]) else 50.0

    r1 = float(latest["r1"])
    r3 = float(latest["r3"])
    r5 = float(latest["r5"])
    r15 = float(latest["r15"]) if pd.notna(latest["r15"]) else 0.0

    volume_ratio = float(latest["volume_ratio"]) if pd.notna(latest["volume_ratio"]) else 0.0

    spy_r3 = benchmark_returns.get("spy_r3", 0.0)
    qqq_r3 = benchmark_returns.get("qqq_r3", 0.0)
    benchmark_r3 = max(spy_r3, qqq_r3)

    relative_strength = r3 - benchmark_r3

    green_bar = float(latest["close"]) > float(prev["close"])
    prior_red_or_flat = float(prev["close"]) <= float(df.iloc[-3]["close"])

    reclaimed_ema3 = float(prev["close"]) <= float(prev["ema3"]) and price > ema3
    reclaimed_ema8 = float(prev["low"]) <= float(prev["ema8"]) and price > ema8
    reclaimed_vwap = (
        float(prev["low"]) <= float(prev["session_vwap"])
        and price > session_vwap
    )

    trend_ok = ema3 >= ema8 * 0.999 and ema8 >= ema21 * 0.998
    above_vwap = price > session_vwap or price > rolling_vwap

    distance_to_vwap_atr = abs(price - session_vwap) / atr if atr > 0 else 999
    distance_to_ema8_atr = abs(price - ema8) / atr if atr > 0 else 999

    near_value = (
        distance_to_vwap_atr <= MAX_EXTENSION_FROM_VWAP_ATR
        or distance_to_ema8_atr <= 0.75
    )

    # Avoid chasing already-running candles.
    if r1 > 0.0025:
        return invalid_candidate(symbol, price, "one_bar_chase_too_high", r1, r3, r5, relative_strength, atr)

    if r3 > 0.0060:
        return invalid_candidate(symbol, price, "three_bar_chase_too_high", r1, r3, r5, relative_strength, atr)

    if volume_ratio < MIN_VOLUME_RATIO_TO_TRADE:
        return invalid_candidate(symbol, price, "volume_too_weak", r1, r3, r5, relative_strength, atr)

    if relative_strength < MIN_RELATIVE_STRENGTH_TO_TRADE:
        return invalid_candidate(symbol, price, "relative_strength_too_weak", r1, r3, r5, relative_strength, atr)

    if r3 < MIN_R3_TO_TRADE:
        return invalid_candidate(symbol, price, "r3_too_weak", r1, r3, r5, relative_strength, atr)

    if r5 < MIN_R5_TO_TRADE:
        return invalid_candidate(symbol, price, "r5_too_weak", r1, r3, r5, relative_strength, atr)

    if rsi > MAX_RSI_MOMENTUM:
        return invalid_candidate(symbol, price, "rsi_too_hot", r1, r3, r5, relative_strength, atr)

    if not near_value:
        return invalid_candidate(symbol, price, "not_near_vwap_or_ema8", r1, r3, r5, relative_strength, atr)

    reclaim_signal = reclaimed_ema3 or reclaimed_ema8 or reclaimed_vwap

    valid = (
        green_bar
        and prior_red_or_flat
        and reclaim_signal
        and trend_ok
        and above_vwap
        and r1 > 0
    )

    if not valid:
        return invalid_candidate(symbol, price, "no_pullback_reclaim_setup", r1, r3, r5, relative_strength, atr)

    score = 0.0

    score += 2.0 if green_bar else 0
    score += 1.0 if prior_red_or_flat else 0
    score += 2.0 if reclaimed_ema3 else 0
    score += 2.5 if reclaimed_ema8 else 0
    score += 2.5 if reclaimed_vwap else 0
    score += 1.5 if trend_ok else 0
    score += 1.0 if above_vwap else 0

    score += min(volume_ratio, 3.0)
    score += max(0, relative_strength) * 1200
    score += max(0, r1) * 700
    score += max(0, r3) * 500
    score += max(0, r5) * 300
    score += max(0, r15) * 150

    if symbol in ADAPTIVE_PREFERRED_SYMBOLS:
        score += 1.5

    if 42 <= rsi <= 62:
        score += 1.0

    if distance_to_vwap_atr <= 0.50:
        score += 1.0

    if distance_to_ema8_atr <= 0.50:
        score += 1.0

    return {
        "symbol": symbol,
        "valid": True,
        "price": price,
        "score": round(max(score, 0), 4),
        "reason": "v2_7_vwap_ema_pullback_reclaim_scalp",
        "r1": r1,
        "r3": r3,
        "r5": r5,
        "relative_strength": relative_strength,
        "atr": atr,
    }


def score_symbol(symbol, df, benchmark_returns, regime, session):
    if df is None or len(df) < MIN_BARS_REQUIRED:
        return None

    if not bar_is_fresh(df, session):
        return None

    df = add_indicators(df)

    if df.empty or len(df) < MIN_BARS_REQUIRED:
        return None

    return score_pullback_scalp_symbol_from_indicator_df(
        symbol=symbol,
        df=df,
        benchmark_returns=benchmark_returns,
        regime=regime,
    )


def score_symbol_backtest(symbol, df, benchmark_returns, regime):
    return score_pullback_scalp_symbol_from_indicator_df(
        symbol=symbol,
        df=df,
        benchmark_returns=benchmark_returns,
        regime=regime,
    )


# ============================================================
# Adaptive update override: harsher symbol cutoff
# ============================================================

def adaptive_symbol_allowed(symbol, symbol_stats, day_symbol_stats, day_key):
    stats = symbol_stats.setdefault(symbol, make_adaptive_symbol_stats())
    day_stats = day_symbol_stats.setdefault((day_key, symbol), make_adaptive_day_stats())

    if int(day_stats["losses"]) >= ADAPTIVE_DISABLE_SYMBOL_AFTER_DAILY_LOSSES:
        return False, "symbol_disabled_after_one_daily_loss"

    if int(stats["losses"]) >= ADAPTIVE_DISABLE_SYMBOL_AFTER_TOTAL_LOSSES and float(stats["pnl"]) < 0:
        return False, "symbol_disabled_total_losses"

    trades = int(stats["trades"])

    if trades < ADAPTIVE_MIN_TRADES_BEFORE_GATE:
        return True, "adaptive_warmup"

    wr = adaptive_win_rate(stats)
    pf = adaptive_profit_factor(stats)

    if float(stats["pnl"]) < 0 and wr < ADAPTIVE_MIN_WIN_RATE:
        return False, "symbol_failed_winrate_gate"

    if float(stats["pnl"]) < 0 and pf < ADAPTIVE_MIN_PROFIT_FACTOR:
        return False, "symbol_failed_profit_factor_gate"

    return True, "adaptive_passed"

# ============================================================
# AURA TRADER V2.8 SELECTIVE HIGH-TURNOVER SCALP OVERRIDE
# Paste below v2.7 and above "# CLI".
#
# Goal:
#   - Keep 90% exposure available
#   - Improve win rate
#   - Stop broad symbol spraying
#   - Trade only the symbols that have shown some ability to work
#   - Require tight trend + VWAP + controlled pullback/reclaim
# ============================================================

LOG_SCHEMA_VERSION = "v2_8_selective_scalp"
LOG_FILE = f"trades_{LOG_SCHEMA_VERSION}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"


# ============================================================
# Narrower universe from symbols that showed positive or near-flat behavior
# across recent tests. This is not proof of future edge.
# ============================================================

TRADE_SYMBOLS = [
    "AAPL", "MSFT", "AMD", "AVGO", "MU", "MRVL", "KLAC",
    "MSTR", "BABA", "PDD",
    "ROKU", "NET", "AFRM", "SNOW", "DIS",
    "JPM", "WFC", "MS",
    "XOM", "HD", "ABBV",
    "QQQ", "DIA",
]

INVERSE_SYMBOLS = []
LEVERAGED_ETFS = set()
ALL_SYMBOLS = sorted(list(set(TRADE_SYMBOLS + REGIME_SYMBOLS)))


# ============================================================
# Risk / turnover
# ============================================================

MAX_OPEN_POSITIONS = 5

MAX_TOTAL_EXPOSURE = 0.90
MAX_POSITION_ALLOCATION = 0.20
MIN_POSITION_ALLOCATION = 0.05
MAX_LEVERAGED_ETF_ALLOCATION = 0.00

MIN_SCORE_TO_TRADE = 9.0
SCORE_POWER = 1.00

ORDER_COOLDOWN_SECONDS = 420

TRADE_START_ET = (9, 45)
TRADE_END_ET = (14, 45)
FORCE_FLATTEN_AFTER_ET = (15, 55)

DAILY_KILL_SWITCH = 0.025

MIN_VOLUME_RATIO_TO_TRADE = 1.15
MIN_RELATIVE_STRENGTH_TO_TRADE = 0.0000
MIN_R3_TO_TRADE = -0.0008
MIN_R5_TO_TRADE = -0.0015

MAX_EXTENSION_FROM_VWAP_ATR = 0.65
MAX_RSI_MOMENTUM = 66

# Faster, more realistic scalp target.
MIN_TAKE_PROFIT_PCT = 0.0015
MAX_TAKE_PROFIT_PCT = 0.0028

# Loss must be close to target. Previous versions let losses dominate.
MIN_STOP_LOSS_PCT = 0.0015
MAX_STOP_LOSS_PCT = 0.0028

ATR_TAKE_PROFIT_MULT = 0.34
ATR_STOP_LOSS_MULT = 0.36

BREAKEVEN_TRIGGER_PCT = 0.0012
BREAKEVEN_LOCK_PCT = 0.0002

TRAIL_TRIGGER_PCT = 0.0020
TRAIL_GIVEBACK_PCT = 0.0009

MAX_HOLD_SECONDS = 18 * 60
MAX_ENTRY_CHASE_PCT = 0.0006

BACKTEST_DEFAULT_DAYS = 30
BACKTEST_DEFAULT_SLIPPAGE_BPS = 5.0


# ============================================================
# Adaptive gates: very strict
# ============================================================

ADAPTIVE_MIN_TRADES_BEFORE_GATE = 2
ADAPTIVE_MIN_WIN_RATE = 0.50
ADAPTIVE_MIN_PROFIT_FACTOR = 1.05

ADAPTIVE_DISABLE_SYMBOL_AFTER_DAILY_LOSSES = 1
ADAPTIVE_DISABLE_SYMBOL_AFTER_TOTAL_LOSSES = 2
ADAPTIVE_GLOBAL_LOSS_STREAK_LIMIT = 3
ADAPTIVE_MAX_DAILY_REALIZED_LOSS_PCT = 0.010

ADAPTIVE_PREFERRED_SYMBOLS = {
    "KLAC", "SNOW", "HD", "AAPL", "MSTR", "MSFT",
    "ROKU", "MRVL", "MU", "BABA", "WFC", "DIS", "JPM",
    "NET", "AFRM", "AVGO", "ABBV", "PDD", "XOM",
}


# ============================================================
# Regime: only trade long when SPY/QQQ are not actively breaking down
# ============================================================

def get_regime_from_precomputed(spy_df, qqq_df):
    if spy_df is None or qqq_df is None:
        return "unknown", "missing_SPY_or_QQQ"

    if len(spy_df) < MIN_BARS_REQUIRED or len(qqq_df) < MIN_BARS_REQUIRED:
        return "unknown", "insufficient_regime_data"

    s = spy_df.iloc[-1]
    q = qqq_df.iloc[-1]

    spy_price = float(s["close"])
    qqq_price = float(q["close"])

    spy_breakdown = (
        spy_price < float(s["ema8"])
        and spy_price < float(s["session_vwap"])
        and float(s["r3"]) < -0.0015
        and float(s["r5"]) < -0.0025
    )

    qqq_breakdown = (
        qqq_price < float(q["ema8"])
        and qqq_price < float(q["session_vwap"])
        and float(q["r3"]) < -0.0015
        and float(q["r5"]) < -0.0025
    )

    if spy_breakdown and qqq_breakdown:
        return "bearish", "both_indexes_breaking_down"

    spy_tradeable = spy_price > float(s["ema21"]) or spy_price > float(s["session_vwap"])
    qqq_tradeable = qqq_price > float(q["ema21"]) or qqq_price > float(q["session_vwap"])

    if spy_tradeable or qqq_tradeable:
        return "bullish", "long_scalps_allowed"

    return "choppy", "not_bearish_but_mixed"


def compute_bracket_prices(entry_price, atr_value):
    price = float(entry_price)
    atr = float(atr_value) if atr_value and atr_value > 0 else price * 0.0025

    atr_pct = atr / price

    take_profit_pct = ATR_TAKE_PROFIT_MULT * atr_pct
    stop_loss_pct = ATR_STOP_LOSS_MULT * atr_pct

    take_profit_pct = max(MIN_TAKE_PROFIT_PCT, min(MAX_TAKE_PROFIT_PCT, take_profit_pct))
    stop_loss_pct = max(MIN_STOP_LOSS_PCT, min(MAX_STOP_LOSS_PCT, stop_loss_pct))

    take_profit_price = price * (1 + take_profit_pct)
    stop_loss_price = price * (1 - stop_loss_pct)

    return take_profit_price, stop_loss_price


def conflicts_with_existing(symbol, selected_symbols, held_symbols, pending_symbols):
    existing = set(selected_symbols) | set(held_symbols) | set(pending_symbols)
    return symbol in existing


# ============================================================
# New signal:
#   - Controlled pullback
#   - Current reclaim
#   - Not a chase candle
#   - Close near EMA/VWAP value
#   - Micro trend positive
# ============================================================

def score_selective_scalp_symbol_from_indicator_df(symbol, df, benchmark_returns, regime):
    if regime == "bearish":
        return None

    if df is None or len(df) < MIN_BARS_REQUIRED:
        return None

    latest = df.iloc[-1]
    prev = df.iloc[-2]
    prev2 = df.iloc[-3]

    price = float(latest["close"])

    if price < MIN_PRICE:
        return None

    ema3 = float(latest["ema3"])
    ema8 = float(latest["ema8"])
    ema21 = float(latest["ema21"])

    prev_ema3 = float(prev["ema3"])
    prev_ema8 = float(prev["ema8"])

    session_vwap = float(latest["session_vwap"])
    rolling_vwap = float(latest["rolling_vwap20"])

    atr = float(latest["atr14"]) if pd.notna(latest["atr14"]) else price * 0.0025
    rsi = float(latest["rsi14"]) if pd.notna(latest["rsi14"]) else 50.0

    r1 = float(latest["r1"])
    r3 = float(latest["r3"])
    r5 = float(latest["r5"])
    r15 = float(latest["r15"]) if pd.notna(latest["r15"]) else 0.0

    volume_ratio = float(latest["volume_ratio"]) if pd.notna(latest["volume_ratio"]) else 0.0

    spy_r3 = benchmark_returns.get("spy_r3", 0.0)
    qqq_r3 = benchmark_returns.get("qqq_r3", 0.0)
    benchmark_r3 = max(spy_r3, qqq_r3)

    relative_strength = r3 - benchmark_r3

    green_bar = float(latest["close"]) > float(prev["close"])
    prev_not_green = float(prev["close"]) <= float(prev2["close"])

    above_ema3 = price > ema3
    above_ema8 = price > ema8
    above_vwap = price > session_vwap or price > rolling_vwap

    micro_trend_ok = ema3 >= ema8 * 0.9995 and ema8 >= ema21 * 0.9975

    touched_value = (
        float(prev["low"]) <= prev_ema3
        or float(prev["low"]) <= prev_ema8
        or float(prev["low"]) <= float(prev["session_vwap"])
    )

    reclaimed_value = (
        price > ema3
        and price > ema8 * 0.999
        and price > min(session_vwap, rolling_vwap) * 0.999
    )

    distance_to_vwap_atr = abs(price - session_vwap) / atr if atr > 0 else 999
    distance_to_ema8_atr = abs(price - ema8) / atr if atr > 0 else 999

    near_value = (
        distance_to_vwap_atr <= MAX_EXTENSION_FROM_VWAP_ATR
        or distance_to_ema8_atr <= 0.60
    )

    # Avoid chase entries.
    if r1 > 0.0018:
        return invalid_candidate(symbol, price, "one_bar_chase_too_high", r1, r3, r5, relative_strength, atr)

    if r3 > 0.0045:
        return invalid_candidate(symbol, price, "three_bar_chase_too_high", r1, r3, r5, relative_strength, atr)

    if volume_ratio < MIN_VOLUME_RATIO_TO_TRADE:
        return invalid_candidate(symbol, price, "volume_too_weak", r1, r3, r5, relative_strength, atr)

    if relative_strength < MIN_RELATIVE_STRENGTH_TO_TRADE:
        return invalid_candidate(symbol, price, "relative_strength_too_weak", r1, r3, r5, relative_strength, atr)

    if r3 < MIN_R3_TO_TRADE:
        return invalid_candidate(symbol, price, "r3_too_weak", r1, r3, r5, relative_strength, atr)

    if r5 < MIN_R5_TO_TRADE:
        return invalid_candidate(symbol, price, "r5_too_weak", r1, r3, r5, relative_strength, atr)

    if rsi > MAX_RSI_MOMENTUM:
        return invalid_candidate(symbol, price, "rsi_too_hot", r1, r3, r5, relative_strength, atr)

    if not near_value:
        return invalid_candidate(symbol, price, "not_near_value", r1, r3, r5, relative_strength, atr)

    valid = (
        green_bar
        and prev_not_green
        and touched_value
        and reclaimed_value
        and micro_trend_ok
        and above_ema3
        and above_ema8
        and above_vwap
        and r1 > 0
    )

    if not valid:
        return invalid_candidate(symbol, price, "no_selective_pullback_reclaim", r1, r3, r5, relative_strength, atr)

    score = 0.0

    score += 2.0 if green_bar else 0
    score += 1.25 if prev_not_green else 0
    score += 2.5 if touched_value else 0
    score += 2.5 if reclaimed_value else 0
    score += 1.5 if micro_trend_ok else 0
    score += 1.0 if above_vwap else 0

    score += min(volume_ratio, 3.0)
    score += max(0, relative_strength) * 1000
    score += max(0, r1) * 550
    score += max(0, r3) * 400
    score += max(0, r5) * 250
    score += max(0, r15) * 125

    if symbol in ADAPTIVE_PREFERRED_SYMBOLS:
        score += 1.5

    if 42 <= rsi <= 60:
        score += 1.0

    if distance_to_vwap_atr <= 0.45:
        score += 1.0

    if distance_to_ema8_atr <= 0.45:
        score += 1.0

    return {
        "symbol": symbol,
        "valid": True,
        "price": price,
        "score": round(max(score, 0), 4),
        "reason": "v2_8_selective_pullback_reclaim_scalp",
        "r1": r1,
        "r3": r3,
        "r5": r5,
        "relative_strength": relative_strength,
        "atr": atr,
    }


def score_symbol(symbol, df, benchmark_returns, regime, session):
    if df is None or len(df) < MIN_BARS_REQUIRED:
        return None

    if not bar_is_fresh(df, session):
        return None

    df = add_indicators(df)

    if df.empty or len(df) < MIN_BARS_REQUIRED:
        return None

    return score_selective_scalp_symbol_from_indicator_df(
        symbol=symbol,
        df=df,
        benchmark_returns=benchmark_returns,
        regime=regime,
    )


def score_symbol_backtest(symbol, df, benchmark_returns, regime):
    return score_selective_scalp_symbol_from_indicator_df(
        symbol=symbol,
        df=df,
        benchmark_returns=benchmark_returns,
        regime=regime,
    )


def adaptive_symbol_allowed(symbol, symbol_stats, day_symbol_stats, day_key):
    stats = symbol_stats.setdefault(symbol, make_adaptive_symbol_stats())
    day_stats = day_symbol_stats.setdefault((day_key, symbol), make_adaptive_day_stats())

    if int(day_stats["losses"]) >= ADAPTIVE_DISABLE_SYMBOL_AFTER_DAILY_LOSSES:
        return False, "symbol_disabled_after_one_daily_loss"

    if int(stats["losses"]) >= ADAPTIVE_DISABLE_SYMBOL_AFTER_TOTAL_LOSSES and float(stats["pnl"]) < 0:
        return False, "symbol_disabled_total_losses"

    trades = int(stats["trades"])

    if trades < ADAPTIVE_MIN_TRADES_BEFORE_GATE:
        return True, "adaptive_warmup"

    wr = adaptive_win_rate(stats)
    pf = adaptive_profit_factor(stats)

    if float(stats["pnl"]) < 0 and wr < ADAPTIVE_MIN_WIN_RATE:
        return False, "symbol_failed_winrate_gate"

    if float(stats["pnl"]) < 0 and pf < ADAPTIVE_MIN_PROFIT_FACTOR:
        return False, "symbol_failed_profit_factor_gate"

    return True, "adaptive_passed"

# ============================================================
# CLI
# ============================================================

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
        help="Allow 4:00 PM-8:00 PM ET after-hours management using extended-hours limit orders.",
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

    parser.add_argument(
        "--backtest",
        action="store_true",
        help="Run a historical minute-bar backtest instead of live scanning.",
    )

    parser.add_argument(
        "--backtest-start",
        type=str,
        default=None,
        help="Backtest start date/time. Example: 2026-05-01 or 2026-05-01T09:30:00",
    )

    parser.add_argument(
        "--backtest-end",
        type=str,
        default=None,
        help="Backtest end date/time. Example: 2026-05-04 or 2026-05-04T16:00:00",
    )

    parser.add_argument(
        "--backtest-days",
        type=int,
        default=BACKTEST_DEFAULT_DAYS,
        help="Number of calendar days to backtest if --backtest-start is not supplied.",
    )

    parser.add_argument(
        "--backtest-equity",
        type=float,
        default=BACKTEST_DEFAULT_STARTING_EQUITY,
        help="Starting equity for the backtest.",
    )

    parser.add_argument(
        "--backtest-slippage-bps",
        type=float,
        default=BACKTEST_DEFAULT_SLIPPAGE_BPS,
        help="Slippage in basis points per side. Example: 5 = 0.05 percent.",
    )

    parser.add_argument(
        "--backtest-feed",
        type=str,
        default="iex",
        help="Historical data feed: iex or sip if your Alpaca plan supports it.",
    )

    args = parser.parse_args()

    if args.backtest:
        run_backtest(
            start=args.backtest_start,
            end=args.backtest_end,
            days=args.backtest_days,
            starting_equity=args.backtest_equity,
            slippage_bps=args.backtest_slippage_bps,
            feed_name=args.backtest_feed,
        )

    elif args.paper:
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


# END BLOCK 3 OF 3