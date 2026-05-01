import os
import time
import csv
import argparse
from datetime import datetime, timedelta, timezone

import pandas as pd
from dotenv import load_dotenv

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed


# ============================================================
# AURA TRADER - PAPER-ONLY MOMENTUM SCALPER
# ============================================================
# Fixes included:
# 1. Uses IEX feed to avoid SIP subscription error.
# 2. Does not buy when market is closed unless explicitly allowed.
# 3. Checks pending/open orders before buying.
# 4. Uses a cooldown after order submission.
# 5. Avoids repeated buys of the same symbol.
# ============================================================


load_dotenv()

API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

if not API_KEY or not SECRET_KEY:
    raise ValueError("Missing ALPACA_API_KEY or ALPACA_SECRET_KEY in .env file")


SYMBOLS = [
    # Major ETFs
    "SPY", "QQQ", "IWM", "DIA",
    "TQQQ", "SQQQ", "SOXL", "SOXS", "SPXL", "SPXS", "TNA", "TZA",

    # Mega-cap tech / AI / semis
    "NVDA", "TSLA", "AMD", "AAPL", "MSFT", "META", "AMZN", "GOOGL", "GOOG",
    "AVGO", "SMCI", "ARM", "MU", "INTC", "QCOM", "MRVL", "AMAT", "LRCX", "KLAC",

    # High-beta tech / software / internet
    "PLTR", "COIN", "MSTR", "NFLX", "UBER", "SHOP", "SNOW", "CRWD", "PANW",
    "NET", "DDOG", "RBLX", "HOOD", "SQ", "PYPL", "AFRM", "ROKU", "DKNG",

    # EV / China high-beta
    "RIVN", "LCID", "NIO", "XPEV", "BABA", "PDD", "JD",

    # Banks / financials
    "JPM", "BAC", "C", "GS", "MS", "WFC",

    # Energy / industrials
    "XOM", "CVX", "OXY", "SLB", "BA", "GE", "CAT", "DE",

    # Healthcare / large caps
    "LLY", "NVO", "UNH", "MRK", "PFE", "ABBV",

    # Retail / consumer
    "WMT", "COST", "TGT", "HD", "LOW", "DIS", "NKE", "SBUX",
]


# =========================
# Strategy / risk settings
# =========================

MAX_OPEN_POSITIONS = 1
MAX_ALLOCATION_PER_TRADE = 0.95

TAKE_PROFIT_PCT = 0.0045       # +0.45%
STOP_LOSS_PCT = -0.0025        # -0.25%
FAST_FAIL_PCT = -0.0018        # -0.18%

DAILY_KILL_SWITCH = 0.03       # Stop after -3% account drawdown
LOOP_SECONDS = 15
LOOKBACK_DAYS = 2
MIN_BARS_REQUIRED = 35

ORDER_COOLDOWN_SECONDS = 180
MIN_BUYING_POWER_FLOOR = 1000

LOG_FILE = "trades.csv"


trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)
data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)

last_order_time_by_symbol = {}


# =========================
# Logging
# =========================

def init_log():
    try:
        with open(LOG_FILE, "x", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp",
                "symbol",
                "price",
                "signal",
                "qty",
                "equity",
                "buying_power",
                "action",
                "reason",
                "score",
                "r1",
                "r3",
                "r5",
                "unrealized_plpc"
            ])
    except FileExistsError:
        pass


def log_event(
    symbol,
    price,
    signal,
    qty,
    equity,
    buying_power,
    action,
    reason,
    score="",
    r1="",
    r3="",
    r5="",
    unrealized_plpc=""
):
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().isoformat(),
            symbol,
            price,
            signal,
            qty,
            equity,
            buying_power,
            action,
            reason,
            score,
            r1,
            r3,
            r5,
            unrealized_plpc
        ])


# =========================
# Alpaca helpers
# =========================

def market_is_open():
    try:
        clock = trading_client.get_clock()
        return bool(clock.is_open)
    except Exception as e:
        print(f"Clock check failed: {e}")
        return False


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


def has_any_open_order():
    orders = get_open_orders()
    return len(orders) > 0


def has_open_order_for_symbol(symbol):
    orders = get_open_orders()

    for order in orders:
        if getattr(order, "symbol", None) == symbol:
            return True

    return False


def symbol_on_cooldown(symbol):
    last_time = last_order_time_by_symbol.get(symbol)

    if last_time is None:
        return False

    elapsed = time.time() - last_time
    return elapsed < ORDER_COOLDOWN_SECONDS


def submit_market_order(symbol, qty, side):
    order = MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=side,
        time_in_force=TimeInForce.DAY
    )

    result = trading_client.submit_order(order)

    last_order_time_by_symbol[symbol] = time.time()

    return result


# =========================
# Data
# =========================

def fetch_all_bars(symbols):
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=LOOKBACK_DAYS)

    request = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame.Minute,
        start=start,
        end=end,
        feed=DataFeed.IEX
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
                result[symbol] = df.tail(180)
            except Exception:
                pass
    else:
        result[symbols[0]] = bars.sort_index().tail(180)

    return result


# =========================
# Strategy
# =========================

def add_indicators(df):
    df = df.copy()

    df["ema3"] = df["close"].ewm(span=3, adjust=False).mean()
    df["ema8"] = df["close"].ewm(span=8, adjust=False).mean()
    df["ema21"] = df["close"].ewm(span=21, adjust=False).mean()

    df["r1"] = df["close"].pct_change(1)
    df["r3"] = df["close"].pct_change(3)
    df["r5"] = df["close"].pct_change(5)

    df["volume_avg20"] = df["volume"].rolling(20).mean()
    df["volume_ratio"] = df["volume"] / df["volume_avg20"]

    df["high20"] = df["high"].rolling(20).max().shift(1)

    return df.dropna()


def score_symbol(symbol, df):
    if df is None or len(df) < MIN_BARS_REQUIRED:
        return None

    df = add_indicators(df)

    if df.empty or len(df) < MIN_BARS_REQUIRED:
        return None

    latest = df.iloc[-1]
    prev = df.iloc[-2]

    price = float(latest["close"])

    if price <= 3:
        return None

    r1 = float(latest["r1"])
    r3 = float(latest["r3"])
    r5 = float(latest["r5"])

    ema3 = float(latest["ema3"])
    ema8 = float(latest["ema8"])
    ema21 = float(latest["ema21"])
    volume_ratio = float(latest["volume_ratio"]) if pd.notna(latest["volume_ratio"]) else 0.0

    above_ema8 = price > ema8
    fast_stack = ema3 > ema8
    trend_stack = ema8 > ema21
    green_last_bar = latest["close"] > prev["close"]

    momentum_valid = (
        above_ema8
        and fast_stack
        and green_last_bar
        and r1 > 0.00005
        and r3 > 0.00025
    )

    surge_valid = (
        price > ema21
        and r5 > 0.0012
        and r1 > -0.0004
    )

    valid = momentum_valid or surge_valid

    if not valid:
        return {
            "symbol": symbol,
            "valid": False,
            "price": price,
            "score": 0,
            "reason": "no_momentum",
            "r1": r1,
            "r3": r3,
            "r5": r5
        }

    score = 0.0

    if above_ema8:
        score += 1.0

    if fast_stack:
        score += 1.0

    if trend_stack:
        score += 0.75

    if green_last_bar:
        score += 0.5

    score += max(0, r1) * 1200
    score += max(0, r3) * 900
    score += max(0, r5) * 600
    score += min(max(volume_ratio, 0), 3) * 0.25

    try:
        if price > float(latest["high20"]):
            score += 1.25
    except Exception:
        pass

    return {
        "symbol": symbol,
        "valid": True,
        "price": price,
        "score": round(score, 4),
        "reason": "aggressive_momentum_candidate",
        "r1": r1,
        "r3": r3,
        "r5": r5
    }


def calculate_qty(equity, buying_power, price):
    max_capital_from_equity = equity * MAX_ALLOCATION_PER_TRADE
    max_capital_from_buying_power = buying_power * 0.95

    capital = min(max_capital_from_equity, max_capital_from_buying_power)

    if capital < MIN_BUYING_POWER_FLOOR:
        return 0

    qty = int(capital // price)
    return max(qty, 0)


def should_exit_position(position, latest_price=None):
    try:
        unrealized_plpc = float(position.unrealized_plpc)
    except Exception:
        unrealized_plpc = 0.0

    if unrealized_plpc >= TAKE_PROFIT_PCT:
        return True, "take_profit_hit", unrealized_plpc

    if unrealized_plpc <= STOP_LOSS_PCT:
        return True, "stop_loss_hit", unrealized_plpc

    try:
        avg_entry = float(position.avg_entry_price)

        if latest_price is not None and avg_entry > 0:
            move_from_entry = (latest_price - avg_entry) / avg_entry

            if move_from_entry <= FAST_FAIL_PCT:
                return True, "fast_momentum_failure", unrealized_plpc
    except Exception:
        pass

    return False, "hold_position", unrealized_plpc


def print_top_candidates(candidates, limit=8):
    ranked = sorted(candidates, key=lambda x: x["score"], reverse=True)
    top = ranked[:limit]

    print("\nTop momentum candidates:")

    for c in top:
        print(
            f"  {c['symbol']:5s} | "
            f"score={c['score']:6.3f} | "
            f"price={c['price']:8.2f} | "
            f"r1={c['r1'] * 100:7.3f}% | "
            f"r3={c['r3'] * 100:7.3f}% | "
            f"r5={c['r5'] * 100:7.3f}%"
        )


# =========================
# Main bot
# =========================

def run_bot(dry_run=True, ignore_market_hours=False):
    init_log()

    account = trading_client.get_account()
    starting_equity = float(account.equity)

    print("============================================================")
    print("AuraTrader paper momentum scalper starting")
    print("============================================================")
    print(f"Mode: {'DRY RUN' if dry_run else 'PAPER ORDERS'}")
    print(f"Starting equity: ${starting_equity:,.2f}")
    print(f"Symbols scanned: {len(SYMBOLS)}")
    print(f"Max open positions: {MAX_OPEN_POSITIONS}")
    print(f"Allocation per trade: {MAX_ALLOCATION_PER_TRADE * 100:.0f}%")
    print(f"Take profit: {TAKE_PROFIT_PCT * 100:.2f}%")
    print(f"Stop loss: {STOP_LOSS_PCT * 100:.2f}%")
    print(f"Loop seconds: {LOOP_SECONDS}")
    print(f"Ignore market hours: {ignore_market_hours}")
    print("Press CTRL+C to stop.")
    print("============================================================")

    while True:
        try:
            account = trading_client.get_account()
            equity = float(account.equity)
            buying_power = float(account.buying_power)

            print("\n" + "=" * 80)
            print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            print(f"Equity=${equity:,.2f} | Buying Power=${buying_power:,.2f}")

            if equity <= starting_equity * (1 - DAILY_KILL_SWITCH):
                print("KILL SWITCH ACTIVE. Account drawdown limit hit.")
                break

            is_open = market_is_open()

            if not is_open and not ignore_market_hours:
                print("MARKET CLOSED | Not scanning or sending new orders.")
                print("Use --ignore-market-hours only for dry-run/testing stale data.")
                time.sleep(60)
                continue

            positions = get_positions_dict()
            open_orders = get_open_orders()

            print(f"Open Positions={len(positions)} | Open Orders={len(open_orders)}")

            bars_by_symbol = fetch_all_bars(SYMBOLS)

            if not bars_by_symbol:
                print("No market data returned.")
                time.sleep(LOOP_SECONDS)
                continue

            # ============================================================
            # 1. Manage existing position first
            # ============================================================

            if positions:
                for symbol, position in positions.items():
                    qty = abs(int(float(position.qty)))

                    latest_price = None

                    if symbol in bars_by_symbol:
                        try:
                            latest_price = float(bars_by_symbol[symbol].iloc[-1]["close"])
                        except Exception:
                            latest_price = None

                    should_exit, reason, unrealized_plpc = should_exit_position(position, latest_price)

                    print(
                        f"POSITION | {symbol} | qty={qty} | "
                        f"unrealized={unrealized_plpc * 100:.3f}% | reason={reason}"
                    )

                    if should_exit and qty > 0:
                        if has_open_order_for_symbol(symbol):
                            action = "SKIP_SELL_PENDING_ORDER"
                            print(f"SKIP SELL | {symbol} | pending order exists")

                        elif dry_run:
                            action = "DRY_RUN_SELL"
                            print(f"{action} | {symbol} | qty={qty} | reason={reason}")

                        else:
                            submit_market_order(symbol, qty, OrderSide.SELL)
                            action = "SELL_ORDER_SENT"
                            print(f"{action} | {symbol} | qty={qty} | reason={reason}")

                        log_event(
                            symbol=symbol,
                            price=latest_price if latest_price else "",
                            signal="SELL",
                            qty=qty,
                            equity=equity,
                            buying_power=buying_power,
                            action=action,
                            reason=reason,
                            unrealized_plpc=unrealized_plpc
                        )

                time.sleep(LOOP_SECONDS)
                continue

            # ============================================================
            # 2. Do not buy if any open order exists
            # ============================================================

            if open_orders:
                print("WAIT | Open order exists. Not submitting another buy.")

                for order in open_orders:
                    print(
                        f"  OPEN ORDER | {getattr(order, 'symbol', '')} | "
                        f"{getattr(order, 'side', '')} | "
                        f"qty={getattr(order, 'qty', '')} | "
                        f"status={getattr(order, 'status', '')}"
                    )

                time.sleep(LOOP_SECONDS)
                continue

            # ============================================================
            # 3. Scan all symbols and rank candidates
            # ============================================================

            candidates = []

            for symbol, df in bars_by_symbol.items():
                try:
                    result = score_symbol(symbol, df)

                    if result is None:
                        continue

                    if result["valid"]:
                        candidates.append(result)

                    log_event(
                        symbol=symbol,
                        price=result.get("price", ""),
                        signal="BUY_CANDIDATE" if result["valid"] else "HOLD",
                        qty=0,
                        equity=equity,
                        buying_power=buying_power,
                        action="SCANNED",
                        reason=result.get("reason", ""),
                        score=result.get("score", ""),
                        r1=result.get("r1", ""),
                        r3=result.get("r3", ""),
                        r5=result.get("r5", ""),
                        unrealized_plpc=""
                    )

                except Exception as e:
                    print(f"{symbol} | scan error | {e}")

            if not candidates:
                print("No valid momentum candidates this loop.")
                time.sleep(LOOP_SECONDS)
                continue

            print_top_candidates(candidates)

            best = sorted(candidates, key=lambda x: x["score"], reverse=True)[0]

            symbol = best["symbol"]
            price = float(best["price"])
            score = float(best["score"])

            if symbol_on_cooldown(symbol):
                print(f"SKIP | {symbol} | cooldown active")
                time.sleep(LOOP_SECONDS)
                continue

            if has_open_order_for_symbol(symbol):
                print(f"SKIP | {symbol} | pending open order already exists")
                time.sleep(LOOP_SECONDS)
                continue

            if buying_power < MIN_BUYING_POWER_FLOOR:
                print(f"SKIP | buying power too low: ${buying_power:,.2f}")
                time.sleep(LOOP_SECONDS)
                continue

            qty = calculate_qty(equity, buying_power, price)

            if qty <= 0:
                print(f"SKIP | {symbol} | qty=0 | price={price}")
                time.sleep(LOOP_SECONDS)
                continue

            if dry_run:
                action = "DRY_RUN_BUY"
                last_order_time_by_symbol[symbol] = time.time()
            else:
                submit_market_order(symbol, qty, OrderSide.BUY)
                action = "BUY_ORDER_SENT"

            print(
                f"{action} | {symbol} | qty={qty} | price={price:.2f} | "
                f"score={score:.3f} | reason={best['reason']}"
            )

            log_event(
                symbol=symbol,
                price=price,
                signal="BUY",
                qty=qty,
                equity=equity,
                buying_power=buying_power,
                action=action,
                reason=best["reason"],
                score=score,
                r1=best["r1"],
                r3=best["r3"],
                r5=best["r5"],
                unrealized_plpc=""
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
    parser.add_argument("--dry-run", action="store_true", help="Scan and log only. No orders.")
    parser.add_argument("--paper", action="store_true", help="Send paper orders to Alpaca.")
    parser.add_argument(
        "--ignore-market-hours",
        action="store_true",
        help="Allow scanning/trading logic outside normal market hours. Use mainly with --dry-run."
    )

    args = parser.parse_args()

    if args.paper:
        run_bot(dry_run=False, ignore_market_hours=args.ignore_market_hours)
    else:
        run_bot(dry_run=True, ignore_market_hours=args.ignore_market_hours)