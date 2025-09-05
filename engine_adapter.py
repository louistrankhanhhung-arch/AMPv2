
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

     # NOTE: keep both "missing" (legacy) and "reasons" for compatibility with main.py logging
    logs = {
        "ENTER": {"state_meta": dec.meta} if dec.decision == "ENTER" else {},
        "WAIT":  (
            {"missing": dec.reasons, "reasons": dec.reasons, "state_meta": dec.meta}
            if dec.decision != "ENTER" else {}
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

    # Telegram signal when ENTER (include both entries if applicable)
    telegram_signal: Optional[str] = None
    decision = dec.decision
    direction = plan["direction"]
    state = dec.state
    confidence = 0.60 if decision == "ENTER" else 0.50

    if decision == 'ENTER' and direction and plan["sl"] is not None and (plan["entry"] is not None or plan["entry2"] is not None):
        strategy = state.replace('_', ' ').title()
        entry_lines = []
        if plan["entry"] is not None:
            entry_lines.append(f"Entry: {plan['entry']}")
        if plan["entry2"] is not None:
            entry_lines.append(f"Entry2: {plan['entry2']}")
        tp_lines = []
        if plan["tp1"] is not None: tp_lines.append(f"TP1: {plan['tp1']}")
        if plan["tp2"] is not None: tp_lines.append(f"TP2: {plan['tp2']}")
        if plan["tp3"] is not None: tp_lines.append(f"TP3: {plan['tp3']}")
        if not tp_lines and plan["tp"] is not None:
            tp_lines.append(f"TP: {plan['tp']}")
        entries_text = "\n".join(entry_lines) if entry_lines else ""
        tps_text = "\n".join(tp_lines) if tp_lines else ""
        telegram_signal = (
            f"{direction} | {symbol}\n"
            f"Strategy: {strategy} ({timeframe})\n"
            f"{entries_text}\n"
            f"SL: {plan['sl']}\n"
            f"{tps_text}"
        )

    # Compose a short headline for logging (includes direction and TP ladder)
    _tp_parts = []
    if plan["tp1"] is not None: _tp_parts.append(f"TP1={plan['tp1']}")
    if plan["tp2"] is not None: _tp_parts.append(f"TP2={plan['tp2']}")
    if plan["tp3"] is not None: _tp_parts.append(f"TP3={plan['tp3']}")
    _tp_text = " ".join(_tp_parts) if _tp_parts else (f"TP={plan['tp']}" if plan["tp"] is not None else "")
    headline = (
        f"DECISION={decision} | STATE={state} | DIR={(plan['direction'] or '-')} | "
        f"entry={plan['entry']} entry2={plan['entry2']} sl={plan['sl']} "
        f"{_tp_text} rr={(plan['rr'] if plan['rr'] is not None else plan['rr2'])}"
    )

    out = {
        "symbol": symbol,
        "timeframe": timeframe,
        "asof": evidence_bundle.get("asof"),
        "state": state,
        "confidence": round(confidence, 3),
        "decision": decision,
        "plan": plan,
        "logs": logs,
        "notes": notes,
        "headline": headline,
        "telegram_signal": telegram_signal,
    }
    return out
