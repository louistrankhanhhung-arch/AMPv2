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
from typing import Any, Dict, List, TYPE_CHECKING
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from universe import get_universe_from_env  # uses DEFAULT_UNIVERSE if SYMBOLS not set
from kucoin_api import fetch_batch, _exchange  # spot-only; 1H drop-partial
from indicators import enrich_indicators, enrich_more
from feature_primitives import compute_features_by_tf
from engine_adapter import decide
from evidence_evaluators import build_evidence_bundle, Config


from notifier_telegram import TelegramNotifier
from storage import SignalPerfDB, JsonStore, UserDB
from templates import render_update

TZ = ZoneInfo("Asia/Ho_Chi_Minh")
TIMEFRAMES = ("1H", "4H", "1D")

log = logging.getLogger("worker")
logging.basicConfig(level=os.getenv("LOG_LEVEL","INFO"),
                    format="%(asctime)s %(levelname)s %(message)s")

# -------- helper: list evidences that are OK ----------
def _extract_evidence_ok(bundle: dict):
    """
    Trả về list evidence đang 'ok' (kèm side nếu có), ví dụ:
    ['retest:long', 'mean_reversion:long', 'volume_impulse_up']
    Bundle có dạng {'evidence': {...}} hoặc object tương đương.
    """
    try:
        ev = bundle.get('evidence', {}) if isinstance(bundle, dict) else {}
    except Exception:
        ev = {}
    out = []
    for name, obj in (ev or {}).items():
        ok = False
        side = None
        if isinstance(obj, dict):
            ok = bool(obj.get('ok'))
            side = obj.get('side')
        else:
            ok = bool(getattr(obj, 'ok', False))
            side = getattr(obj, 'side', None)
        if ok:
            out.append(f"{name}:{side}" if side in ("long","short") else name)
    return sorted(out)

# --- Telegram Teaser Notifier (init-once, lazy) ---
TN = None
def _get_notifier():
    """Create TelegramNotifier once; return False if init failed."""
    global TN
    if TN is None:
        try:
            TN = TelegramNotifier()
        except Exception as e:
            log.warning(f"TelegramNotifier init failed; disabled. reason={e}")
            TN = False
    return TN
# --- end telegram notifier helper ---

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
    log.debug(f"[{symbol}] fetching OHLCV…")
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
    log.debug(f"[{symbol}] fetched: 1H={l1}, 4H={l4}, 1D={lD} in {t_fetch:.2f}s")

    # enrich indicators → features_by_tf
    t1 = time.time()
    dfs = _enrich_all(dfs)
    log.debug(f"[{symbol}] enrich done in {time.time()-t1:.2f}s")
    t2 = time.time()
    feats_by_tf = compute_features_by_tf(dfs)   # builds trend/momentum/volatility/levels/vp-bands,…
    log.debug(f"[{symbol}] features done in {time.time()-t2:.2f}s")
    # attach df to 1H for decision (decision engine expects it)
    if '1H' in feats_by_tf:
        feats_by_tf['1H']['df'] = dfs.get('1H')

    # evidence bundle (STRUCT JSON)
    t3 = time.time()
    bundle = build_evidence_bundle(symbol, feats_by_tf, cfg)
    log.debug(f"[{symbol}] bundle done in {time.time()-t3:.2f}s")

    # decide on 4H as execution TF (1H trigger, 4H execution, 1D context)
    t4 = time.time()
    try:
        out = decide(symbol, "4H", feats_by_tf, bundle)
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
    log.debug(f"[{symbol}] decide done in {elapsed_dec:.2f}s; total {total_time:.2f}s")
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
        logs = out.get("logs", {})
        wait_log = {}
        if isinstance(logs, dict):
            wait_log = logs.get("WAIT", {}) or {}
        miss = None
        if isinstance(wait_log, dict):
            miss = wait_log.get("missing") or wait_log.get("reasons")
        if miss is None:
            miss = out.get("reasons")
        log.info(f"[{symbol}] WAIT missing={miss} have={_extract_evidence_ok(bundle)}")

    # log JSON line
    # --- post teaser to Telegram Channel when ENTER ---
    if dec == "ENTER":
        tn = _get_notifier()
        if tn:
            try:
                plan_for_teaser = dict(plan or {})
                plan_for_teaser.update({
                    "symbol": symbol,
                    "DIRECTION": (plan.get("direction") or plan.get("dir") or "-").upper() if isinstance(plan, dict) else "-",
                    "STATE": state,
                    "notes": out.get("notes", []),
                })
                perf = SignalPerfDB(JsonStore(os.getenv("DATA_DIR","./data")))
                # 12h cooldown
                if perf.cooldown_active(symbol, seconds=12*3600):
                    log.info(f"[{symbol}] skip ENTER due to cooldown (12h)")
                else:
                    sid, msg_id = tn.post_teaser(plan_for_teaser)
                    perf.open(sid, plan_for_teaser, message_id=msg_id)
            except Exception as e:
                log.warning(f"[{symbol}] teaser post failed: {e}")
    # --- end teaser post ---
# --- progress check: update TP/SL hits for existing OPEN trades ---
    try:
        df_1h = dfs.get("1H")
        if df_1h is None or df_1h.empty:
            raise ValueError("missing 1H frame")
        price_now = float(df_1h["close"].iloc[-1])

        perf = SignalPerfDB(JsonStore(os.getenv("DATA_DIR","./data")))
        open_trades = perf.by_symbol(symbol)
        if open_trades:
            tn2 = _get_notifier()
            for t in open_trades:
                def crossed(side, price, level):
                    return (side=="LONG" and price>=level) or (side=="SHORT" and price<=level)
                side = (t.get("dir") or "").upper()
                msg_id = t.get("message_id")
                entry = float(t.get("entry") or 0.0)
                def margin_pct(hit_price: float) -> float:
                    if not entry: return 0.0
                    if side == "LONG":
                        return (hit_price - entry) / entry * 100.0
                    else:
                        return (entry - hit_price) / entry * 100.0
                # Đã hit rồi thì bỏ qua (idempotent)
                hits = t.get("hits", {})
                msg_id = t.get("message_id")
                entry = float(t.get("entry") or 0.0)
                def margin_pct(hit_price: float) -> float:
                    if not entry: return 0.0
                    return ((hit_price - entry) / entry * 100.0) if side=="LONG" else ((entry - hit_price) / entry * 100.0)

                # TP1
                if t.get("status")=="OPEN" and not hits.get("TP1") and t.get("tp1") and crossed(side, price_now, t["tp1"]):
                    perf.set_hit(t["sid"], "TP1", (t.get("r_ladder",{}) or {}).get("tp1") or 0.0)
                    hits["TP1"] = int(__import__("time").time())
                    t["status"] = "TP1"
                    note = "🎯 TP1 hit — Nâng SL lên Entry để bảo toàn lợi nhuận."
                    extra = {"margin_pct": margin_pct(float(t["tp1"]))}
                    if tn2:
                        if msg_id:
                            tn2.send_channel_update(int(msg_id), render_update(t, note, extra))
                        else:
                            tn2.send_channel(render_update(t, note, extra))
                        
                # TP2
                if t.get("status") in ("OPEN","TP1") and not hits.get("TP2") and t.get("tp2") and crossed(side, price_now, t["tp2"]):
                    perf.set_hit(t["sid"], "TP2", (t.get("r_ladder",{}) or {}).get("tp2") or 0.0)
                    hits["TP2"] = int(__import__("time").time())
                    t["status"] = "TP2"
                    note = "🎯 TP2 hit — Nâng SL lên Entry để bảo toàn lợi nhuận."
                    extra = {"margin_pct": margin_pct(float(t["tp2"]))}
                    if tn2:
                        if msg_id:
                            tn2.send_channel_update(int(msg_id), render_update(t, note, extra))
                        else:
                            tn2.send_channel(render_update(t, note, extra))

                # TP3 => đóng lệnh
                if t.get("status") in ("OPEN","TP1","TP2") and t.get("tp3") and crossed(side, price_now, t["tp3"]):
                    perf.close(t["sid"], "TP3")
                    t["status"] = "TP3"
                    note = "🎯 TP3 hit — Đóng lệnh."
                    extra = {"margin_pct": margin_pct(float(t["tp3"]))}
                    if tn2:
                        if msg_id:
                            tn2.send_channel_update(int(msg_id), render_update(t, note, extra))
                        else:
                            tn2.send_channel(render_update(t, note, extra))

                # NEW: Nếu đã đạt TP1/TP2 mà giá quay ngược về Entry -> ĐÓNG LỆNH,
                # và hiển thị lợi nhuận theo TP cao nhất đã đạt.
                if t.get("status") in ("TP1","TP2") and entry:
                    retraced = (side == "LONG" and price_now <= entry) or (side == "SHORT" and price_now >= entry)
                    if retraced:
                        # chọn TP cao nhất đã đạt để tính % lợi nhuận hiển thị
                        highest = None
                        hit_price = None
                        if (t.get("status") == "TP2") or (hits.get("TP2")):
                            highest = "TP2"
                            hit_price = float(t.get("tp2") or entry)
                        else:
                            highest = "TP1"
                            hit_price = float(t.get("tp1") or entry)
                        perf.close(t["sid"], "ENTRY")
                        t["status"] = "CLOSE"
                        note = f"📌 Đóng lệnh — Giá quay về Entry sau khi đã đạt {highest}."
                        extra = {"margin_pct": margin_pct(hit_price)}
                        if tn2:
                            if msg_id:
                                tn2.send_channel_update(int(msg_id), render_update(t, note, extra))
                            else:
                                tn2.send_channel(render_update(t, note, extra))

                # --- SL => đóng lệnh (CHỈ khi chưa từng chạm TP nào)
                slv = t.get("sl")
                has_tp_hit = bool(hits.get("TP1") or hits.get("TP2") or hits.get("TP3"))
                if t.get("status") == "OPEN" and not has_tp_hit and slv and (
                    (side == "LONG"  and price_now <= slv) or
                    (side == "SHORT" and price_now >= slv)
                ):
                    perf.close(t["sid"], "SL")
                    t["status"] = "SL"
                    note = "⚠️ SL hit — Đóng lệnh."
                    extra = {"margin_pct": margin_pct(float(slv))}
                    if tn2:
                        if msg_id:
                            tn2.send_channel_update(int(msg_id), render_update(t, note, extra))
                        else:
                            tn2.send_channel(render_update(t, note, extra))

                  
        # nếu không có open_trades -> không làm gì, không log warning
    except Exception as e:
        log.warning("progress-check failed: %s", e)

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
    last_kpi_day = None  # NEW: để gửi KPI 1 lần/ngày
    while True:
        now = datetime.now(TZ)
        blk = which_block_for_minute(now.minute)
        # Include half-hour slot so each block can run twice per hour
        half = 0 if now.minute < 30 else 1
        tick_key = (now.year, now.month, now.day, now.hour, half, blk)
        if blk is not None and tick_key != last_tick and now.second < 10:
            last_tick = tick_key
            run_block(blk, blocks[blk], cfg, limit, ex=shared_ex)
        # NEW: KPI lúc 08:18 local (VN) ~ 01:18 UTC
        try:
            if now.hour == 8 and now.minute == 18 and (last_kpi_day != (now.year, now.month, now.day)):
                last_kpi_day = (now.year, now.month, now.day)
                # dựng báo cáo 24H
                perf = SignalPerfDB(JsonStore(os.getenv("DATA_DIR","./data")))
                detail = perf.kpis_24h_detail()
                # ngày/tháng cho header
                report_date_str = now.strftime("%d/%m/%Y")
                from templates import render_kpi_24h
                tn = _get_notifier()
                if tn:
                    html = render_kpi_24h(detail, report_date_str, upgrade_url=f"https://t.me/{tn.username}?start=upgrade")
                    tn.send_kpi24(html)
        except Exception as e:
            log.warning(f"KPI-24H send failed: {e}")
        # sleep until next 5-minute boundary
        secs = now.second + now.minute*60
        to_next = 300 - (secs % 300)
        time.sleep(max(5, min(60, to_next)))

if __name__ == "__main__":
    try:
        loop_scheduler()
    except KeyboardInterrupt:
        sys.exit(0)
