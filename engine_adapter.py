
"""
engine_adapter.py
- Thin wrapper so main.py can call decide(...) without decision_engine.py.
- Uses tiny_core_side_state to compute decision and formats a legacy-compatible dict.
"""
from typing import Dict, Any, List, Optional
from tiny_core_side_state import SideCfg, run_side_state_core

def _rr(entry: Optional[float], sl: Optional[float], tp: Optional[float], side: Optional[str]) -> Optional[float]:
    if entry is None or sl is None or tp is None or side is None:
        return None
    if side == "long":
        risk = entry - sl
        reward = tp - entry
    else:
        risk = sl - entry
        reward = entry - tp
    if risk <= 0:
        return None
    return reward / risk

def decide(symbol: str, timeframe: str, features_by_tf: Dict[str, Dict[str, Any]], evidence_bundle: Dict[str, Any]) -> Dict[str, Any]:
    # evidence_bundle expected to include 'evidence' object; pass through as eb-like
    eb = evidence_bundle.get("evidence") or evidence_bundle  # tolerate both shapes
    cfg = SideCfg()
    dec = run_side_state_core(features_by_tf, eb, cfg)

    # Build plan (legacy fields)
    tps = dec.setup.tps or []
    tp1 = tps[0] if len(tps) > 0 else None
    tp2 = tps[1] if len(tps) > 1 else None
    tp3 = tps[2] if len(tps) > 2 else None

    # RR calculations
    rr1 = _rr(dec.setup.entry, dec.setup.sl, tp1, dec.side)
    rr2 = _rr(dec.setup.entry, dec.setup.sl, tp2, dec.side)
    rr3 = _rr(dec.setup.entry, dec.setup.sl, tp3, dec.side)

    # ---------- price formatting helpers ----------
    def _infer_dp(symbol: str, price: Optional[float], features_by_tf: Dict[str, Any], evidence_bundle: Dict[str, Any]) -> int:
        """
        Ưu tiên:
        1) meta.price_dp / meta.tick_size -> dp
        2) Heuristic theo giá (crypto)
        3) VN stock (không có '/') -> 0 lẻ
        """
        # 1) từ features/meta nếu có
        try:
            meta = (features_by_tf or {}).get("1H", {}).get("meta", {}) or {}
            dp = meta.get("price_dp")
            if isinstance(dp, int) and 0 <= dp <= 8:
                return dp
            tick = meta.get("tick_size") or evidence_bundle.get("meta", {}).get("tick_size")
            if tick:
                s = f"{tick}"
                if "." in s:
                    return min(8, max(0, len(s.split(".")[1].rstrip("0"))))
                # tick là số nguyên -> 0 lẻ
                return 0
        except Exception:
            pass
        # 2) Heuristic theo giá (crypto)
        if "/" in symbol:
            p = float(price or evidence_bundle.get("last_price") or 0.0)
            if p >= 1000: return 1
            if p >= 100:  return 2
            if p >= 1:    return 3
            if p >= 0.1:  return 4
            if p >= 0.01: return 5
            return 6
        # 3) VN stock (mã không có '/'): 0 lẻ (VND)
        return 0

    def _fmt(x: Optional[float], dp: int) -> Optional[str]:
        if x is None:
            return None
        try:
            return f"{float(x):.{dp}f}"
        except Exception:
            return f"{x}"
    # ---------- end helpers ----------

    plan = {
        "direction": dec.side.upper() if dec.side else None,
        "entry": dec.setup.entry,
        "entry2": None,               # kept for compatibility; tiny core emits single entry
        "sl": dec.setup.sl,
        "tp": tp1,                    # fallback single TP
        "tp1": tp1,
        "tp2": tp2,
        "tp3": tp3,
        "rr": rr1,                    # primary RR
        "rr2": rr2,
        "rr3": rr3,
    }

    # ---- ensure locals before logging ----
    decision = dec.decision or "WAIT"
    state = dec.state
    confidence = 0.0
    try:
        if isinstance(dec.meta, dict):
            confidence = float(dec.meta.get("confidence", 0.0) or 0.0)
    except Exception:
        confidence = 0.0

    # ---- logging with exchange-like decimals ----
    dp = _infer_dp(symbol, dec.setup.entry, features_by_tf, evidence_bundle)
    f_entry = _fmt(dec.setup.entry, dp)
    f_sl    = _fmt(dec.setup.sl, dp)
    f_tp1   = _fmt(tp1, dp) if tp1 is not None else None
    f_tp2   = _fmt(tp2, dp) if tp2 is not None else None
    f_tp3   = _fmt(tp3, dp) if tp3 is not None else None

    # legacy log line(s) — keep for backward-compat printing
    legacy_lines = []
    legacy_lines.append(
        " ".join(
            [
                f"[{symbol}]",
                f"DECISION={decision}",
                f"| STATE={state or '-'}",
                f"| DIR={plan['direction'] or '-'}",
                f"| entry={f_entry}",
                f"sl={f_sl}",
                f"TP1={f_tp1}" if f_tp1 is not None else "TP1=None",
                f"TP2={f_tp2}" if f_tp2 is not None else "TP2=None",
                f"TP3={f_tp3}" if f_tp3 is not None else "TP3=None",
                f"RR1={f'{rr1:.1f}' if rr1 is not None else 'None'}",
                f"RR2={f'{rr2:.1f}' if rr2 is not None else 'None'}",
                f"RR3={f'{rr3:.1f}' if rr3 is not None else 'None'}",
            ]
        )
    )

    # headline (one-liner) — show all three TPs
    _tp_parts_hl = [
        f"TP1={f_tp1}" if f_tp1 is not None else "TP1=None",
        f"TP2={f_tp2}" if f_tp2 is not None else "TP2=None",
        f"TP3={f_tp3}" if f_tp3 is not None else "TP3=None",
    ]
    _tp_text_hl = " ".join(_tp_parts_hl)
    headline = f"[{symbol}] {decision} | {state or '-'} {plan['direction'] or '-'} | E={f_entry} SL={f_sl} {_tp_text_hl}"

    # Telegram signal (nếu ENTER): format theo dp
    telegram_signal = None
    if decision == "ENTER" and plan["direction"] and dec.setup.sl is not None and (dec.setup.entry is not None or tp1 is not None):
        strategy = (state or "").replace("_", " ").title()
        entry_lines = []
        if dec.setup.entry is not None:
            entry_lines.append(f"Entry: {f_entry}")
        if f_tp1 is not None:
            entry_lines.append(f"TP1: {f_tp1}")
        if f_tp2 is not None:
            entry_lines.append(f"TP2: {f_tp2}")
        if f_tp3 is not None:
            entry_lines.append(f"TP3: {f_tp3}")
        telegram_signal = "\n".join(
            [
                f"#{symbol.replace('/', '')} {plan['direction']}",
                f"State: {state or '-'} | Strategy: {strategy}",
                *entry_lines,
                f"SL: {f_sl}",
                f"RR1: {rr1:.1f}" if rr1 is not None else "",
            ]
        ).strip()
    # Chuẩn hoá logs cho main.py:
    # - Giữ legacy text trong logs["TEXT"] (list)
    # - Cung cấp cấu trúc cho WAIT/ENTER để main.py lấy missing/reasons
    logs: Dict[str, Any] = {
        "TEXT": legacy_lines,
        "ENTER": {"state_meta": dec.meta} if decision == "ENTER" else {},
        "WAIT": (
            {
                "missing": list(dec.reasons or []),
                "reasons": list(dec.reasons or []),
                "state_meta": dec.meta,
            }
            if decision != "ENTER"
            else {}
        ),
        "AVOID": {},
    }
    notes: List[str] = []
    if dec.state == "none_state":
        notes.append("No clear retest/break context — WAIT")
    if "far_from_entry" in dec.reasons:
        notes.append("Proximity guard: too far from entry")
    if "rr_too_low" in dec.reasons:
        notes.append("RR min not satisfied")

    out = {
        "symbol": symbol,
        "timeframe": timeframe,
        "asof": evidence_bundle.get("asof"),
        "state": state,
        "confidence": round(confidence, 3),
        "decision": decision,
        "plan": plan,
        "logs": logs,
        "reasons": list(dec.reasons or []),  # tiện lợi, phòng khi caller cần
        "notes": notes,
        "headline": headline,
        "telegram_signal": telegram_signal,
    }
    return out
