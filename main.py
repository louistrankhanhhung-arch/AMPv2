#!/usr/bin/env python3
"""
Main worker for Crypto Signal (Railway ready)

- Splits symbols into 4 blocks and scans every hour:
  block1 at :00, block2 at :05, block3 at :10, block4 at :15 (Asia/Ho_Chi_Minh)
- Workflow per symbol:
  1) fetch OHLCV for 1H/4H/1D (1H drop partial bar; 4H/1D keep realtime)
  2) enrich indicators (EMA/RSI/BB/ATR/volume, candle anatomy)
  3) compute features_by_tf (trend/momentum/volatility/SR + volume profile bands)
  4) build evidence bundle (STRUCT JSON)
  5) decide ENTER/WAIT/AVOID; optionally push Telegram

Env:
  SYMBOLS=BTC/USDT,ETH/USDT,...   # optional, else use DEFAULT_UNIVERSE
  KUCOIN_API_KEY=... (optional)
  KUCOIN_API_SECRET=... (optional)
  KUCOIN_API_PASSPHRASE=... (optional)
  BATCH_LIMIT=300
  TELEGRAM_BOT_TOKEN=xxxx (optional)
  TELEGRAM_CHAT_ID=12345 (optional)
  RUN_ONCE=1  # run all 4 blocks immediately and exit (for testing)
"""
import os, sys, time, json, logging
from typing import Dict, List
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

from universe import get_universe_from_env  # uses DEFAULT_UNIVERSE if SYMBOLS not set  :contentReference[oaicite:6]{index=6}
from kucoin_api import fetch_batch           # spot-only client; drop partial only for 1H  :contentReference[oaicite:7]{index=7}
from indicators import enrich_indicators, enrich_more
from feature_primitives import compute_features_by_tf
from evidence_evaluators import build_evidence_bundle, Config
from decision_engine import decide

TZ = ZoneInfo("Asia/Ho_Chi_Minh")
TIMEFRAMES = ("1H", "4H", "1D")

log = logging.getLogger("worker")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

def split_into_4_blocks(symbols: List[str]) -> List[List[str]]:
    """Stable split: [s[0], s[4], ...], [s[1], s[5], ...], ..."""
    return [symbols[i::4] for i in range(4)]

def which_block_for_minute(minute: int):
    mapping = {0:0, 5:1, 10:2, 15:3}
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

def process_symbol(symbol: str, cfg: Config, limit: int):
    log.info(f"[{symbol}] fetching OHLCV…")
    # fetch with partial-bar drop for 1H; realtime for 4H/1D (handled in fetch_batch)
    dfs = fetch_batch(
        symbol,
        timeframes=TIMEFRAMES,
        limit=limit,
        drop_partial=True,  # only applied to 1H internally
    )

    # enrich indicators → features_by_tf
    dfs = _enrich_all(dfs)
    feats_by_tf = compute_features_by_tf(dfs)   # builds trend/momentum/volatility/levels/vp-bands,…  :contentReference[oaicite:8]{index=8}
    # attach df to 1H for decision (decision engine expects it)
    if '1H' in feats_by_tf:
        feats_by_tf['1H']['df'] = dfs.get('1H')

    # evidence bundle (STRUCT JSON)
    bundle = build_evidence_bundle(symbol, feats_by_tf, cfg)  # returns state + evidence blocks  :contentReference[oaicite:9]{index=9}

    # decide on 1H as primary TF
    out = decide(symbol, "1H", feats_by_tf, bundle)          # validated DecisionOut + telegram_signal  :contentReference[oaicite:10]{index=10}

    # log JSON line
    print(json.dumps(out, ensure_ascii=False))
    if out.get("telegram_signal"):
        send_telegram(out["telegram_signal"])

def run_block(block_idx: int, symbols: List[str], cfg: Config, limit: int):
    log.info(f"=== Running block {block_idx+1}/4 ({len(symbols)} symbols) ===")
    for sym in symbols:
        try:
            process_symbol(sym, cfg, limit)
        except Exception as e:
            log.exception(f"[{sym}] error: {e}")

def loop_scheduler():
    symbols = get_universe_from_env()
    blocks = split_into_4_blocks(symbols)
    cfg = Config()  # default thresholds per TF
    limit = int(os.getenv("BATCH_LIMIT", "300"))

    if os.getenv("RUN_ONCE") == "1":
        # Run all blocks immediately (useful for CI/test)
        for i in range(4):
            run_block(i, blocks[i], cfg, limit)
        return

    log.info(f"Universe size={len(symbols)}; block sizes={[len(b) for b in blocks]}")
    log.info("Schedule each hour: block0 at :00, block1 at :05, block2 at :10, block3 at :15 (Asia/Ho_Chi_Minh)")

    last_tick = None
    while True:
        now = datetime.now(TZ)
        blk = which_block_for_minute(now.minute)
        tick_key = (now.year, now.month, now.day, now.hour, blk)
        if blk is not None and tick_key != last_tick and now.second < 10:
            last_tick = tick_key
            run_block(blk, blocks[blk], cfg, limit)
        # sleep until next 5-minute boundary
        secs = now.second + now.minute*60
        to_next = 300 - (secs % 300)
        time.sleep(max(5, min(60, to_next)))

if __name__ == "__main__":
    try:
        loop_scheduler()
    except KeyboardInterrupt:
        sys.exit(0)
