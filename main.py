#!/usr/bin/env python3
"""
Main worker for Crypto Signal (Railway ready)

- Splits symbols into 4 blocks and scans twice per hour:
  block1 at :00 & :30, block2 at :05 & :35, block3 at :10 & :40, block4 at :15 & :45 (Asia/Ho_Chi_Minh)
- Workflow per symbol:
  1) fetch OHLCV for 1H/4H/1D (1H drop partial bar; 4H/1D keep realtime)
  2) enrich indicators (EMA/RSI/BB/ATR/volume, candle anatomy)
  3) compute features_by_tf (trend/momentum/volatility/SR + volume profile bands)
  4) build evidence bundle (STRUCT JSON)
  5) decide ENTER/WAIT/AVOID; optionally push Telegram
"""
import os, sys, time, json, logging
from typing import Dict, List
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from universe import get_universe_from_env  # uses DEFAULT_UNIVERSE if SYMBOLS not set
from kucoin_api import fetch_batch, _exchange  # spot-only; 1H drop-partial
from indicators import enrich_indicators, enrich_more
from feature_primitives import compute_features_by_tf
from evidence_evaluators import build_evidence_bundle, Config
from decision_engine import decide

TZ = ZoneInfo("Asia/Ho_Chi_Minh")
TIMEFRAMES = ("1H", "4H", "1D")

log = logging.getLogger("worker")
logging.basicConfig(level=os.getenv("LOG_LEVEL","INFO"),
                    format="%(asctime)s %(levelname)s %(message)s")

def split_into_4_blocks(symbols: List[str]) -> List[List[str]]:
    """Stable split: [s[0], s[4], ...], [s[1], s[5], ...], ..."""
    return [symbols[i::4] for i in range(4)]

def which_block_for_minute(minute: int):
    # Twice per hour schedule
    mapping = {0:0, 5:1, 10:2, 15:3, 30:0, 35:1, 40:2, 45:3}
    return mapping.get(minute % 60)

def send_telegram(text: str):
    tok = os.getenv("TELEGRAM_BOT_TOKEN")
    chat = os.getenv("TELEGRAM_CHAT_ID")
    if not tok or not chat or not text:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{tok}/sendMessage",
            json={"chat_id": chat, "text": text}
        )
    except Exception as e:
        log.warning(f"Telegram send failed: {e}")

def _enrich_all(dfs: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    out = {}
    for tf, df in dfs.items():
        if df is None or df.empty:
            out[tf] = df
            continue
        x = enrich_indicators(df)
        x = enrich_more(x)
        out[tf] = x
    return out

def process_symbol(symbol: str, cfg: Config, limit: int, ex=None):
    t0 = time.time()
    log.info(f"[{symbol}] fetching OHLCV…")
    # fetch with partial-bar drop for 1H; realtime for 4H/1D (handled in fetch_batch)
    sleep_between_tf = float(os.getenv("SLEEP_BETWEEN_TF", "0.3"))
    dfs = fetch_batch(
        symbol,
        timeframes=TIMEFRAMES,
        limit=limit,
        drop_partial=True,        # only applied to 1H internally
        sleep_between_tf=sleep_between_tf,  # reduce burst per symbol
        ex=ex                     # reuse shared exchange to avoid 429
    )
    t_fetch = time.time() - t0
    df1 = dfs.get("1H")
    df4 = dfs.get("4H")
    dfD = dfs.get("1D")
    l1 = 0 if df1 is None else len(df1.index)
    l4 = 0 if df4 is None else len(df4.index)
    lD = 0 if dfD is None else len(dfD.index)
    log.info(f"[{symbol}] fetched: 1H={l1}, 4H={l4}, 1D={lD} in {t_fetch:.2f}s")

    # enrich indicators → features_by_tf
    t1 = time.time()
    dfs = _enrich_all(dfs)
    log.info(f"[{symbol}] enrich done in {time.time()-t1:.2f}s")
    t2 = time.time()
    feats_by_tf = compute_features_by_tf(dfs)   # builds trend/momentum/volatility/levels/vp-bands,…
    log.info(f"[{symbol}] features done in {time.time()-t2:.2f}s")
    # attach df to 1H for decision (decision engine expects it)
    if '1H' in feats_by_tf:
        feats_by_tf['1H']['df'] = dfs.get('1H')

    # evidence bundle (STRUCT JSON)
    t3 = time.time()
    bundle = build_evidence_bundle(symbol, feats_by_tf, cfg)
    log.info(f"[{symbol}] bundle done in {time.time()-t3:.2f}s")

    # decide on 1H as primary TF
    t4 = time.time()
    try:
        out = decide(symbol, "1H", feats_by_tf, bundle)
    except Exception as e:
        log.exception(f"[{symbol}] decide failed: {e}")
        # Fallback để tiếp tục vòng lặp, không làm gãy block
        out = {
            "symbol": symbol,
            "decision": "AVOID",
            "state": None,
            "plan": {},
            "logs": {"AVOID": {"reasons": ["internal_error"]}},
        }
        print(json.dumps(out, ensure_ascii=False), flush=True)
        return
    elapsed_dec = time.time() - t4
    total_time = time.time() - t0
    dec = out.get("decision")
    state = out.get("state")
    plan = out.get("plan") or {}
    log.info(f"[{symbol}] decide done in {elapsed_dec:.2f}s; total {total_time:.2f}s")
   # Prefer concise headline from decision_engine if available (already includes DIR/TP ladder)
    headline = out.get("headline")
    if headline:
        log.info(headline)
    else:
        # Build TP ladder + RR ladder + direction
        dir_val = (plan.get("direction") or plan.get("dir") or "-")
        # Backward compatible: if only single tp/rr exists, map to TP1/RR1
        tp1 = plan.get("tp1", plan.get("tp"))
        tp2 = plan.get("tp2")
        tp3 = plan.get("tp3")
        rr1 = plan.get("rr1", plan.get("rr"))
        rr2 = plan.get("rr2")
        rr3 = plan.get("rr3")

        tp_parts = []
        if tp1 is not None: tp_parts.append(f"TP1={tp1}")
        if tp2 is not None: tp_parts.append(f"TP2={tp2}")
        if tp3 is not None: tp_parts.append(f"TP3={tp3}")
        rr_parts = []
        if rr1 is not None: rr_parts.append(f"RR1={rr1}")
        if rr2 is not None: rr_parts.append(f"RR2={rr2}")
        if rr3 is not None: rr_parts.append(f"RR3={rr3}")

        tp_str = " ".join(tp_parts)
        rr_str = " ".join(rr_parts)

        log.info(
            f"[{symbol}] DECISION={dec} | STATE={state} | "
            f"DIR={str(dir_val).upper()} | "
            f"entry={plan.get('entry')} entry2={plan.get('entry2')} "
            f"sl={plan.get('sl')} "
            f"{(tp_str + ' ' + rr_str).strip()}".strip()
        )
    if dec == "WAIT":
        miss = (out.get("logs", {}).get("WAIT", {}).get("missing"))
        log.info(f"[{symbol}] WAIT missing={miss}")
    if dec == "AVOID":
        reasons = (out.get("logs", {}).get("AVOID", {}).get("reasons"))
        log.info(f"[{symbol}] AVOID reasons={reasons}")

    # log JSON line
    print(json.dumps(out, ensure_ascii=False), flush=True)
    if out.get("telegram_signal"):
        send_telegram(out["telegram_signal"])

def run_block(block_idx: int, symbols: List[str], cfg: Config, limit: int, ex=None):
    log.info(f"=== Running block {block_idx+1}/4 ({len(symbols)} symbols) ===")
    sleep_between_symbols = float(os.getenv("SLEEP_BETWEEN_SYMBOLS", "0.15"))
    for sym in symbols:
        try:
            process_symbol(sym, cfg, limit, ex=ex)
            time.sleep(sleep_between_symbols)  # tiny pause between symbols to smooth rate limit
        except Exception as e:
            log.exception(f"[{sym}] error: {e}")

def loop_scheduler():
    symbols = get_universe_from_env()
    blocks = split_into_4_blocks(symbols)
    cfg = Config()  # default thresholds per TF
    limit = int(os.getenv("BATCH_LIMIT", "300"))
    # Create ONE shared exchange to let ccxt throttler pace requests correctly
    shared_ex = _exchange(
        kucoin_key=os.getenv("KUCOIN_API_KEY"),
        kucoin_secret=os.getenv("KUCOIN_API_SECRET"),
        kucoin_passphrase=os.getenv("KUCOIN_API_PASSPHRASE"),
    )

    if os.getenv("RUN_ONCE") == "1":
        # Run all blocks immediately (useful for CI/test)
        for i in range(4):
            run_block(i, blocks[i], cfg, limit, ex=shared_ex)
        return

    log.info(f"Universe size={len(symbols)}; block sizes={[len(b) for b in blocks]}")
    log.info("Schedule each hour: block0 at :00 & :30, block1 at :05 & :35, "
             "block2 at :10 & :40, block3 at :15 & :45 (Asia/Ho_Chi_Minh)")
 

    last_tick = None
    while True:
        now = datetime.now(TZ)
        blk = which_block_for_minute(now.minute)
        # Include half-hour slot so each block can run twice per hour
        half = 0 if now.minute < 30 else 1
        tick_key = (now.year, now.month, now.day, now.hour, half, blk)
        if blk is not None and tick_key != last_tick and now.second < 10:
            last_tick = tick_key
            run_block(blk, blocks[blk], cfg, limit, ex=shared_ex)
        # sleep until next 5-minute boundary
        secs = now.second + now.minute*60
        to_next = 300 - (secs % 300)
        time.sleep(max(5, min(60, to_next)))

if __name__ == "__main__":
    try:
        loop_scheduler()
    except KeyboardInterrupt:
        sys.exit(0)
