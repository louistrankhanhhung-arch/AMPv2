"""
Decision Engine v2
------------------
- Consumes evidence bundle (v2) and features_by_tf.
- Builds multiple entry candidates with entry zones (direct/throwback/pullback/trend-break).
- Chooses best candidate by RR & proximity while respecting required/avoid filters.
- Outputs decision (ENTER/WAIT/AVOID), plan, logs, and Telegram signal.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

try:
    from pydantic import BaseModel, Field, ConfigDict, confloat
except Exception:
    BaseModel = object  # type: ignore
    def Field(*a, **k): return None  # type: ignore
    def ConfigDict(**k): return {}    # type: ignore
    def confloat(*a, **k): return float  # type: ignore

# --------------------------------------------------------------------------------------
# Rules
# --------------------------------------------------------------------------------------

@dataclass
class DecisionRules:
    rr_min: float = 1.5
    rr_avoid: float = 1.2
    proximity_atr: float = 0.3       # proximity for generic modes
    retest_zone_atr: float = 0.15    # proximity for throwback/pullback
    trend_break_buf_atr: float = 0.20
    retest_pad_atr: float = 0.05
    vol_z_hot: float = 3.0
    rsi_overbought: float = 80.0
    rsi_oversold: float = 20.0
    hvn_avoid_atr: float = 0.3

# --------------------------------------------------------------------------------------
# Output models
# --------------------------------------------------------------------------------------

class PlanOut(BaseModel):
    direction: Optional[str] = Field(None, description="long/short")
    mode: Optional[str] = None
    entry: Optional[float] = None
    entry_zone: Optional[List[float]] = None
    sl: Optional[float] = None
    tp: Optional[float] = None
    rr: Optional[confloat(ge=0)] = None  # type: ignore
    note: Optional[str] = None
    model_config = ConfigDict(extra="forbid")

class LogsOut(BaseModel):
    ENTER: Dict[str, Any] = Field(default_factory=dict)
    WAIT: Dict[str, Any] = Field(default_factory=dict)
    AVOID: Dict[str, Any] = Field(default_factory=dict)
    model_config = ConfigDict(extra="forbid")

class DecisionOut(BaseModel):
    symbol: str
    timeframe: str
    asof: Optional[str] = None
    state: str
    confidence: float
    decision: str
    plan: PlanOut
    logs: LogsOut
    telegram_signal: Optional[str] = None
    model_config = ConfigDict(extra="forbid")

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------

def _smart_round(x: float) -> float:
    ax = abs(x)
    if ax >= 100000: return round(x, 0)
    if ax >= 10000:  return round(x, 1)
    if ax >= 1000:   return round(x, 2)
    if ax >= 100:    return round(x, 3)
    if ax >= 10:     return round(x, 4)
    if ax >= 1:      return round(x, 5)
    return round(x, 6)


def _nearest_band_tp(levels: Dict[str, Any], price: float, side: str) -> Optional[float]:
    buckets = levels.get('bands_up' if side == 'long' else 'bands_down') or []
    if not buckets:
        return None
    forward = [b for b in buckets if (b['tp'] > price) if side == 'long' else (b['tp'] < price)]
    if forward:
        forward.sort(key=lambda b: abs(b['tp'] - price))
        return float(forward[0]['tp'])
    buckets = sorted(buckets, key=lambda b: b.get('score', 0), reverse=True)
    return float(buckets[0]['tp']) if buckets else None


def _protective_sl(levels: Dict[str, Any], ref_level: float, atr: float, side: str) -> Optional[float]:
    pad = 0.3 * atr
    if side == 'long':
        for b in levels.get('bands_up', []) + levels.get('bands_down', []):
            lo, hi = float(b['band'][0]), float(b['band'][1])
            if lo <= ref_level <= hi:
                return float(min(lo, ref_level) - pad)
        return float(ref_level - pad)
    else:
        for b in levels.get('bands_up', []) + levels.get('bands_down', []):
            lo, hi = float(b['band'][0]), float(b['band'][1])
            if lo <= ref_level <= hi:
                return float(max(hi, ref_level) + pad)
        return float(ref_level + pad)


def _rr(direction: str, entry: float, sl: float, tp: float) -> float:
    if direction == 'long':
        risk = max(1e-9, entry - sl); reward = max(0.0, tp - entry)
    else:
        risk = max(1e-9, sl - entry); reward = max(0.0, entry - tp)
    return float(reward / risk) if risk > 0 else 0.0


def _price(df: pd.DataFrame) -> float:
    return float(df['close'].iloc[-1])


def _in_zone(price: float, zone: List[float]) -> bool:
    lo, hi = min(zone), max(zone)
    return (lo <= price <= hi)


def _dist_to_zone(price: float, zone: List[float]) -> float:
    lo, hi = min(zone), max(zone)
    if price < lo: return lo - price
    if price > hi: return price - hi
    return 0.0

# --------------------------------------------------------------------------------------
# Candidate builder & decision
# --------------------------------------------------------------------------------------

def decide(symbol: str,
           timeframe: str,
           features_by_tf: Dict[str, Dict[str, Any]],
           evidence_bundle: Dict[str, Any],
           rules: DecisionRules = DecisionRules()) -> Dict[str, Any]:

    f1 = features_by_tf.get('1H', {})
    df1: pd.DataFrame = f1.get('df')
    if df1 is None:
        raise ValueError("features_by_tf['1H']['df'] is required for decision")

    atr = float(f1.get('volatility', {}).get('atr', 0.0) or 0.0)
    price_now = _price(df1)
    levels = f1.get('levels', {})

    eb = evidence_bundle.get('evidence', {})

    state = evidence_bundle.get('state', 'undefined')
    confidence = float(evidence_bundle.get('confidence', 0.0))

    # Determine side hint
    direction: Optional[str] = None
    if state == 'breakout': direction = 'long'
    elif state == 'breakdown': direction = 'short'
    elif state == 'reclaim': direction = (eb.get('price_reclaim',{}).get('ref') or {}).get('side')

    # Required confirmations
    price_ok = (
        (direction=='long' and eb.get('price_breakout',{}).get('ok')) or
        (direction=='short' and eb.get('price_breakdown',{}).get('ok')) or
        (state=='reclaim' and eb.get('price_reclaim',{}).get('ok'))
    )
    vol_ok = bool((eb.get('volume') or {}).get('ok', False))
    trend_ok = bool((eb.get('trend') or {}).get('ok', False))

    # Avoid filters
    rsi = float((f1.get('momentum') or {}).get('rsi', 50.0))
    div = (f1.get('momentum') or {}).get('divergence', 'none')
    vz = float((f1.get('volume') or {}).get('vol_z20', 0.0))
    rsi_extreme = (rsi >= rules.rsi_overbought) or (rsi <= rules.rsi_oversold)
    div_against = (direction == 'long' and div == 'bearish') or (direction == 'short' and div == 'bullish')
    vol_blowoff = (vz >= rules.vol_z_hot)
    liq_ok = bool((eb.get('liquidity') or {}).get('ok', True))

    avoid_reasons = []
    if rsi_extreme: avoid_reasons.append('rsi_extreme')
    if div_against: avoid_reasons.append('rsi_divergence_against')
    if vol_blowoff: avoid_reasons.append('volume_blowoff')
    if not liq_ok:  avoid_reasons.append('heavy_liquidity_ahead')

    # Build candidates list
    candidates: List[Dict[str, Any]] = []

    def add_candidate(mode: str, side: str, zone: List[float], ref_level: Optional[float]=None, prox_atr: float=None):
        if not (np.isfinite(atr) and atr>0 and zone and all(np.isfinite(z) for z in zone)):
            return
        entry = float(sum(zone)/2.0)
        sl = _protective_sl(levels, ref_level=ref_level if ref_level is not None else entry, atr=atr, side=side)
        tp = _nearest_band_tp(levels, price_now, side=side)
        if sl is None or tp is None:
            return
        rr = _rr(side, entry, sl, tp)
        dist = _dist_to_zone(price_now, zone)
        prox_limit = (prox_atr if prox_atr is not None else rules.proximity_atr) * atr
        proximity_ok = dist <= max(1e-9, prox_limit)
        candidates.append({
            'mode': mode,
            'direction': side,
            'entry_zone': [float(zone[0]), float(zone[1])],
            'entry': float(entry), 'sl': float(sl), 'tp': float(tp), 'rr': float(rr),
            'proximity_ok': bool(proximity_ok)
        })

    # Throwback candidate (breakout/breakdown true)
    tb = eb.get('throwback') or {}
    if direction in ('long','short') and tb.get('ok') and tb.get('zone'):
        add_candidate('throwback_retest', direction, tb['zone'], ref_level=float(tb.get('ref', price_now)), prox_atr=rules.retest_zone_atr)

    # Direct break candidate (only when explosive + BB expanding)
    bbx = (eb.get('bb') or {}).get('ok', False)
    exp = (eb.get('volume_explosive') or {}).get('ok', False)
    if direction in ('long','short') and exp and bbx:
        delta = 0.02 * atr
        zone = [price_now - delta, price_now + delta]
        add_candidate('direct_break', direction, zone, ref_level=float((tb.get('ref') or (eb.get('price_breakout',{}).get('ref') or eb.get('price_breakdown',{}).get('ref') or {})).get('hh' if direction=='long' else 'll', price_now)))

    # Trend break candidate (nearest swing + buffer)
    pb_ref = (eb.get('price_breakout',{}).get('ref') or {})
    pd_ref = (eb.get('price_breakdown',{}).get('ref') or {})
    if direction == 'long' and ('hh' in pb_ref):
        hh = float(pb_ref['hh']); buf = rules.trend_break_buf_atr * atr
        add_candidate('trend_break', 'long', [hh + buf - 0.01*atr, hh + buf + 0.01*atr], ref_level=hh)
    if direction == 'short' and ('ll' in pd_ref):
        ll = float(pd_ref['ll']); buf = rules.trend_break_buf_atr * atr
        add_candidate('trend_break', 'short', [ll - buf - 0.01*atr, ll - buf + 0.01*atr], ref_level=ll)

    # Pullback candidates (EMA20 / BB mid zones)
    pbk = eb.get('pullback') or {}
    if direction in ('long','short') and pbk.get('ok'):
        if pbk.get('zone'): add_candidate('trend_pullback_ema', direction, pbk['zone'], ref_level=float((pb_ref.get('hh') if direction=='long' else pd_ref.get('ll')) or price_now), prox_atr=rules.retest_zone_atr)
        if pbk.get('fallback_zone'): add_candidate('trend_pullback_bbmid', direction, pbk['fallback_zone'], ref_level=float((pb_ref.get('hh') if direction=='long' else pd_ref.get('ll')) or price_now), prox_atr=rules.retest_zone_atr)

    # Reclaim retest
    if state == 'reclaim':
        prc = eb.get('price_reclaim') or {}
        lvl = (prc.get('ref') or {}).get('level')
        if lvl is not None:
            if direction == 'long':
                zone = [float(lvl + 0.02*atr), float(lvl + 0.08*atr)]
            else:
                zone = [float(lvl - 0.08*atr), float(lvl - 0.02*atr)]
            add_candidate('reclaim_retest', direction, zone, ref_level=float(lvl), prox_atr=rules.retest_zone_atr)

    # Filter candidates by required confirmations
    valid = []
    for c in candidates:
        if not price_ok or not vol_ok or not trend_ok:
            c['reason'] = 'missing_required'
            continue
        if c['rr'] < rules.rr_min:
            c['reason'] = 'rr_min'
            continue
        if not c['proximity_ok']:
            c['reason'] = 'price_far_from_zone'
            continue
        c['reason'] = 'ok'
        valid.append(c)

    # Decide
    decision = 'WAIT'
    plan_dict = {
        'direction': direction,
        'mode': None,
        'entry': None,
        'entry_zone': None,
        'sl': None,
        'tp': None,
        'rr': None,
        'note': None,
    }

    if avoid_reasons:
        decision = 'AVOID'
    elif valid:
        # pick best RR
        best = sorted(valid, key=lambda x: x['rr'], reverse=True)[0]
        decision = 'ENTER'
        plan_dict.update(best)
    else:
        # keep WAIT; include top candidate diagnostics
        pass

    plan = PlanOut(**{k: ( _smart_round(v) if isinstance(v,(int,float)) else v ) for k,v in plan_dict.items()})

    logs = LogsOut(
        ENTER={'candidates': candidates, 'valid': valid},
        WAIT={'missing_required': not (price_ok and vol_ok and trend_ok), 'reason_of_candidates': [c.get('reason') for c in candidates]},
        AVOID={'reasons': avoid_reasons, 'rsi': rsi, 'divergence': div, 'vol_z20': vz}
    )

    # Telegram signal when ENTER
    telegram_signal = None
    if decision == 'ENTER' and plan.direction and plan.entry_zone and plan.sl is not None and plan.tp is not None:
        lo, hi = plan.entry_zone
        telegram_signal = (
            f"{plan.direction.upper()} | {symbol}\n"
            f"Strategy: {state.capitalize()} ({timeframe})\n"
            f"Mode: {plan.mode}\n"
            f"Entry Zone: {_smart_round(lo)} - {_smart_round(hi)}\n"
            f"SL: {plan.sl}\nTP: {plan.tp}"
        )

    out = DecisionOut(
        symbol=symbol,
        timeframe=timeframe,
        asof=evidence_bundle.get('asof'),
        state=state,
        confidence=round(confidence,3),
        decision=decision,
        plan=plan,
        logs=logs,
        telegram_signal=telegram_signal,
    )
    return out.model_dump() if hasattr(out,'model_dump') else out.__dict__
