"""
Decision Engine
---------------
- Validate STRUCT JSON (evidence bundle) using Pydantic.
- Compute entry plan (Entry/SL/TP) and classify: ENTER / WAIT / AVOID.
- Designed for T+ trading with multi-timeframe inputs: 1H (primary), 4H (confirm), 1D (context).

Inputs
------
- features_by_tf: output from feature_primitives.compute_features_by_tf({'1H': df1h, '4H': df4h, '1D': df1d})
  (Recommended to attach the actual DataFrame as features_by_tf[tf]['df'] before calling)
- evidence_bundle: output from evidence_evaluators.build_evidence_bundle(symbol, features_by_tf, cfg)

Outputs
-------
- Dict with keys: decision, state, confidence, plan {direction, entry, sl, tp, rr},
  logs for ENTER/WAIT/AVOID, and optional telegram_signal text when ENTER.

Notes
-----
- This module *decides* based on validated inputs. Evidence scoring stays in evidence_evaluators.
- RR (risk:reward) is computed against nearest bands.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from pydantic import BaseModel, Field, ConfigDict, conlist, confloat, field_validator, model_validator
except Exception:  # graceful fallback if pydantic is missing
    BaseModel = object  # type: ignore
    def Field(*args, **kwargs):  # type: ignore
        return None
    def ConfigDict(**kwargs):  # type: ignore
        return {}
    def conlist(*args, **kwargs):  # type: ignore
        return list
    def confloat(*args, **kwargs):  # type: ignore
        return float
    def field_validator(*args, **kwargs):  # type: ignore
        def deco(fn):
            return fn
        return deco
    def model_validator(*args, **kwargs):  # type: ignore
        def deco(fn):
            return fn
        return deco

# =====================================================
# 1) Models (lenient on input; strict on output)
# =====================================================

class EvidenceItemIn(BaseModel):
    ok: bool = False
    score: float = 0.0
    why: str = ""
    missing: List[str] = Field(default_factory=list)
    ref: Optional[Dict[str, Any]] = None
    model_config = ConfigDict(extra="ignore")

class VolumeBundleIn(BaseModel):
    primary: EvidenceItemIn
    confirm: Optional[EvidenceItemIn] = None
    ok: bool = False
    model_config = ConfigDict(extra="ignore")

class MomentumBundleIn(BaseModel):
    primary: EvidenceItemIn
    confirm: Optional[EvidenceItemIn] = None
    model_config = ConfigDict(extra="ignore")


class EvidenceIn(BaseModel):
    price_breakout: EvidenceItemIn
    price_breakdown: EvidenceItemIn
    price_reclaim: EvidenceItemIn
    sideways: EvidenceItemIn
    volume: VolumeBundleIn
    momentum: MomentumBundleIn
    trend: EvidenceItemIn
    candles: EvidenceItemIn
    liquidity: EvidenceItemIn
    # --- new evidences ---
    bb: Optional[EvidenceItemIn] = None
    volume_explosive: Optional[EvidenceItemIn] = None
    throwback: Optional[EvidenceItemIn] = None
    pullback: Optional[EvidenceItemIn] = None
    mean_reversion: Optional[EvidenceItemIn] = None
    false_breakout: Optional[EvidenceItemIn] = None
    false_breakdown: Optional[EvidenceItemIn] = None
    trend_follow_up: Optional[EvidenceItemIn] = None
    trend_follow_down: Optional[EvidenceItemIn] = None
    rejection: Optional[EvidenceItemIn] = None
    divergence: Optional[EvidenceItemIn] = None
    compression_ready: Optional[EvidenceItemIn] = None
    volatility_breakout: Optional[EvidenceItemIn] = None
    model_config = ConfigDict(extra="ignore")


class EvidenceBundleIn(BaseModel):
    symbol: str
    asof: Optional[str] = None
    timeframes: List[str]
    state: str
    confidence: float = 0.0
    why: str = ""
    evidence: EvidenceIn
    model_config = ConfigDict(extra="ignore")

# Strict output
class PlanOut(BaseModel):
    direction: Optional[str] = Field(None, description="long/short or None")
    # Primary entry (retest for breakout/breakdown; break for trend-follow)
    entry: Optional[float] = None
    sl: Optional[float] = None
    tp: Optional[float] = None
    rr: Optional[confloat(ge=0)] = None  # type: ignore
    # Optional secondary entry for trend-follow (EMA20/BB mid pullback)
    entry2: Optional[float] = None
    rr2: Optional[confloat(ge=0)] = None  # type: ignore
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
    decision: str  # ENTER/WAIT/AVOID
    plan: PlanOut
    logs: LogsOut
    telegram_signal: Optional[str] = None
    model_config = ConfigDict(extra="forbid")

# =====================================================
# 2) Decision rules
# =====================================================

@dataclass
class DecisionRules:
    rr_min: float = 1.5
    rr_avoid: float = 1.2
    proximity_atr: float = 0.3  # price must be within 0.3*ATR of entry to ENTER
    vol_z_hot: float = 3.0
    rsi_overbought: float = 80.0
    rsi_oversold: float = 20.0
    hvn_avoid_atr: float = 0.3  # if heavy zone within 0.3*ATR → avoid
    # --- New: entry setup tuning ---
    retest_pad_atr: float = 0.05      # small pad above/below level for retest entry
    retest_zone_atr: float = 0.15     # acceptable distance from price to retest entry to consider ENTER
    trend_break_buf_atr: float = 0.20 # trend-follow Entry1 uses break of nearest swing with this buffer


# =====================================================
# 2.5) State mapping helpers (new trade types)
# =====================================================

STATE_TO_DIR = {
    'breakout': 'long',
    'breakdown': 'short',
    'reclaim': None,          # derived from evidence.price_reclaim.ref.side
    'trend_follow_up': 'long',
    'trend_follow_down': 'short',
    'trend_follow_pullback': None,  # derive from momentum side if available
    'false_breakout': 'short',
    'false_breakdown': 'long',
    'mean_reversion': None,   # derive from evidence.mean_reversion.side
    'rejection': None,        # derive from evidence.rejection.side
    'divergence_up': 'long',
    'divergence_down': 'short',

    'sideways': None,
    'compression_ready': None,
    'volatility_breakout': None,  # infer from price breakout/breakdown flags
    'throwback_long': 'long',
    'throwback_short': 'short',
}

# Required evidence keys per state (must be True to ENTER)
REQUIRED_BY_STATE = {
    'breakout': ['price_breakout', 'volume', 'trend', 'bb'],
    'breakdown': ['price_breakdown', 'volume', 'trend', 'bb'],
    'reclaim': ['price_reclaim', 'volume'],
    'trend_follow_up': ['trend', 'momentum'],
    'trend_follow_down': ['trend', 'momentum'],
    'trend_follow_pullback': ['pullback', 'trend'],
    'false_breakout': ['false_breakout'],
    'false_breakdown': ['false_breakdown'],
    'mean_reversion': ['mean_reversion'],
    'rejection': ['rejection'],
    'divergence_up': ['divergence'],
    'divergence_down': ['divergence'],

    'sideways': ['sideways'],
    'range': ['sideways'],
    'compression_ready': ['sideways'],  # wait for break
    'volatility_breakout': ['volatility_breakout', 'bb'],
    'throwback_long': ['throwback'],
    'throwback_short': ['throwback'],
}

# =====================================================
# 3) Helpers
# =====================================================

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
    # choose first target beyond price in direction; fallback to best score
    forward = [b for b in buckets if ((side == 'long' and b['tp'] > price) or (side == 'short' and b['tp'] < price))]
    if forward:
        forward.sort(key=lambda b: abs(b['tp'] - price))
        return float(forward[0]['tp'])
    # fallback: highest score band
    buckets = sorted(buckets, key=lambda b: b.get('score', 0), reverse=True)
    return float(buckets[0]['tp']) if buckets else None


def _protective_sl(levels: Dict[str, Any], ref_level: float, atr: float, side: str) -> Optional[float]:
    pad = 0.3 * atr
    if side == 'long':
        # SL a bit below lower of the band containing ref_level if available
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
    if state == 'throwback_long' and direction == 'long':
        hh = (eb.evidence.price_breakout.ref or {}).get('hh') if hasattr(eb.evidence, 'price_breakout') else None
        entry, sl, tp, note = _retest_entry('long', float(hh) if hh is not None else price_now)
    elif state == 'trend_follow_pullback' and direction == 'long':
        # Prefer EMA20/BB mid zone from pullback evidence if available
        z = ((eb.evidence.__dict__.get('pullback') or {}).get('zone') if hasattr(eb.evidence, '__dict__') else None)
        if z and isinstance(z, (list, tuple)):
            entry = float((z[0] + z[1]) / 2.0); sl = _protective_sl(levels, ref_level=z[0], atr=atr, side='long'); tp = _nearest_band_tp(levels, price_now, side='long'); note = 'pullback_zone_entry'
        else:
            e1, e2 = _trend_follow_entries('long'); entry, entry2 = e1, e2; sl = _protective_sl(levels, ref_level=(entry - rules.trend_break_buf_atr*atr) if entry else price_now, atr=atr, side='long'); tp = _nearest_band_tp(levels, price_now, side='long'); note = 'trend_follow_pullback_fallback'
    elif state == 'false_breakout' and direction == 'short':
        ll = (eb.evidence.price_breakdown.ref or {}).get('ll') if hasattr(eb.evidence, 'price_breakdown') else None
        ref = ll if ll is not None else float(df1['low'].iloc[-2])
        entry, sl, tp, note = _retest_entry('short', float(ref))
    elif state == 'mean_reversion' and direction == 'long':
        ref = float(df1['low'].iloc[-2]); entry = price_now; sl = float(ref - 0.2*atr); tp = _nearest_band_tp(levels, price_now, side='long'); note = 'mean_reversion_rebound'
    elif state == 'rejection' and direction == 'long':
        ref = float(df1['low'].iloc[-2]); entry = price_now; sl = float(ref - 0.2*atr); tp = _nearest_band_tp(levels, price_now, side='long'); note = 'rejection_long'
    elif state == 'divergence_up' and direction == 'long':
        e1, e2 = _trend_follow_entries('long'); entry, entry2 = e1, e2; sl = _protective_sl(levels, ref_level=(entry - rules.trend_break_buf_atr*atr) if entry else price_now, atr=atr, side='long'); tp = _nearest_band_tp(levels, price_now, side='long'); note = 'divergence_break_entry'
    elif state == 'volatility_breakout' and direction == 'long':
        hh = (eb.evidence.price_breakout.ref or {}).get('hh'); entry, sl, tp, note = _retest_entry('long', float(hh) if hh is not None else price_now)
    elif direction == 'long':
        risk = max(1e-9, entry - sl)
        reward = max(0.0, tp - entry)
    else:
        risk = max(1e-9, sl - entry)
        reward = max(0.0, entry - tp)
    return float(reward / risk) if risk > 0 else 0.0


def _price(df: pd.DataFrame) -> float:
    return float(df['close'].iloc[-1])

# =====================================================
# 4) Core decision
# =====================================================

def decide(symbol: str,
           timeframe: str,
           features_by_tf: Dict[str, Dict[str, Any]],
           evidence_bundle: Dict[str, Any],
           rules: DecisionRules = DecisionRules()) -> Dict[str, Any]:
    """Return DecisionOut as dict (validated)."""
    # Validate input evidence (lenient)
    eb = EvidenceBundleIn(**evidence_bundle)

    f1 = features_by_tf.get('1H', {})
    df1: pd.DataFrame = f1.get('df')  # Orchestrator should attach df
    if df1 is None:
        raise ValueError("features_by_tf['1H']['df'] is required for decision")

    atr = float(f1.get('volatility', {}).get('atr', 0.0) or 0.0)
    price_now = _price(df1)
    levels = f1.get('levels', {})

    # Determine side by state
    state = eb.state
    confidence = float(eb.confidence)
    direction: Optional[str] = None
    if state == 'reclaim':
        side_ref = (eb.evidence.price_reclaim.ref or {}).get('side')
        direction = side_ref if side_ref in ('long','short') else None
    elif state == 'mean_reversion':
        side_ref = (eb.evidence.__dict__.get('mean_reversion') or {}).get('side') if hasattr(eb.evidence, '__dict__') else None
        direction = side_ref if side_ref in ('long','short') else None
    elif state == 'rejection':
        side_ref = (eb.evidence.__dict__.get('rejection') or {}).get('side') if hasattr(eb.evidence, '__dict__') else None
        direction = side_ref if side_ref in ('long','short') else None
    elif state == 'false_breakout':
        direction = 'short'
    elif state == 'false_breakdown':
        direction = 'long'
    elif state == 'trend_follow_up':
        direction = 'long'
    elif state == 'trend_follow_down':
        direction = 'short'
    elif state == 'breakout':
        direction = 'long'
    elif state == 'breakdown':
        direction = 'short'
    elif state == 'volatility_breakout':
        # infer from price evidences
        if eb.evidence.price_breakout.ok: direction = 'long'
        elif eb.evidence.price_breakdown.ok: direction = 'short'
        else: direction = None
    elif state == 'divergence_up':
        direction = 'long'
    elif state == 'divergence_down':
        direction = 'short'
    elif state == 'throwback_long':
        direction = 'long'
    elif state == 'throwback_short':
        direction = 'short'
    elif state == 'trend_follow_pullback':
        # derive from momentum bias
        rsi = float(features_by_tf.get('1H', {}).get('momentum', {}).get('rsi', 50.0))
        direction = 'long' if rsi >= 50 else 'short'

    
    # Required confirmations per state
    # Required confirmations
    req_ok: List[bool] = []
    miss_reasons: List[str] = []

    # Helper lấy evidence theo tên; chịu lỗi an toàn
    def _ev(name: str) -> Any:
        try:
            return getattr(eb.evidence, name)
        except Exception:
            return None

    # Lấy danh sách required theo state
    req_keys = REQUIRED_BY_STATE.get(state, [])
    # Nếu state chưa có map (trường hợp state lạ), fallback theo direction
    if not req_keys:
        if direction == 'long':
            req_keys = ['price_breakout', 'volume', 'trend']
        elif direction == 'short':
            req_keys = ['price_breakdown', 'volume', 'trend']
    # Chỉ giữ các key thực sự tồn tại trong EvidenceIn hiện tại
    req_keys = [k for k in req_keys if hasattr(eb.evidence, k)]

    for k in req_keys:
        item = _ev(k)
        ok_flag = bool(getattr(item, 'ok', False)) if item is not None else False
        req_ok.append(ok_flag)
        if not ok_flag:
            # Chuẩn hoá tên thiếu cho volume
            miss_reasons.append('volume' if k == 'volume' else k)

    # Optional/guards (không phải required)
    vol_ok = bool(getattr(eb.evidence.volume, 'ok', False))
    tr_ok = bool(getattr(eb.evidence.trend, 'ok', False))
    mom_ok = bool(getattr(eb.evidence.momentum.primary, 'ok', False))
    cdl_ok = bool(getattr(eb.evidence.candles, 'ok', False))
    liq_ok = bool(getattr(eb.evidence.liquidity, 'ok', False))

    # Generic guards used across strategies
    vol_ok = bool(getattr(eb.evidence.volume, 'ok', False))
    tr_ok = bool(getattr(eb.evidence.trend, 'ok', False))
    mom_ok = bool(getattr(eb.evidence.momentum.primary, 'ok', False))
    cdl_ok = bool(getattr(eb.evidence.candles, 'ok', False))
    liq_ok = bool(getattr(eb.evidence.liquidity, 'ok', False))
    
    mom_ok = bool(eb.evidence.momentum.primary.ok)
    cdl_ok = bool(eb.evidence.candles.ok)
    liq_ok = bool(eb.evidence.liquidity.ok)

    # Candidate entry plan (revised setups)
    note = ""
    # --- init plan vars early to avoid UnboundLocalError
    entry: Optional[float] = None
    entry2: Optional[float] = None
    sl: Optional[float] = None
    tp: Optional[float] = None
    rr: Optional[float] = None
    rr2: Optional[float] = None
    proximity_ok = False
    proximity_ok2 = False

# Helper to build a retest entry around a reference level
    def _retest_entry(side: str, ref: float) -> Tuple[Optional[float], Optional[float], Optional[float], str]:
        if not np.isfinite(ref) or atr <= 0:
            return None, None, None, ""
        pad = rules.retest_pad_atr * atr
        if side == 'long':
            e = float(ref + pad)         # enter slightly above reclaimed level
            s = _protective_sl(levels, ref_level=ref, atr=atr, side='long')
            t = _nearest_band_tp(levels, price_now, side='long')
            return e, s, t, "retest_of_level"
        else:
            e = float(ref - pad)
            s = _protective_sl(levels, ref_level=ref, atr=atr, side='short')
            t = _nearest_band_tp(levels, price_now, side='short')
            return e, s, t, "retest_of_level"

    # Helper to build trend-follow entries (Entry1: break nearest swing; Entry2: EMA20/BB mid)
    def _trend_follow_entries(side: str) -> Tuple[Optional[float], Optional[float]]:
        try:
            if side == 'long':
                hh_list = f1.get('swings', {}).get('last_HH') or []
                ref = float(hh_list[0]) if hh_list else np.nan
                e1 = float(ref + rules.trend_break_buf_atr * atr) if np.isfinite(ref) and atr > 0 else None
                ema20 = float(f1.get('trend', {}).get('ema20', np.nan))
                bb_mid = float(f1.get('soft_levels', {}).get('soft_up', [{}])[0].get('level', np.nan)) if False else float(df1['bb_mid'].iloc[-1]) if 'bb_mid' in df1.columns else np.nan
                # Prefer EMA20; fallback to BB mid
                e2 = float(ema20) if np.isfinite(ema20) else (float(bb_mid) if np.isfinite(bb_mid) else None)
                return e1, e2
            else:
                ll_list = f1.get('swings', {}).get('last_LL') or []
                ref = float(ll_list[0]) if ll_list else np.nan
                e1 = float(ref - rules.trend_break_buf_atr * atr) if np.isfinite(ref) and atr > 0 else None
                ema20 = float(f1.get('trend', {}).get('ema20', np.nan))
                bb_mid = float(df1['bb_mid'].iloc[-1]) if 'bb_mid' in df1.columns else np.nan
                e2 = float(ema20) if np.isfinite(ema20) else (float(bb_mid) if np.isfinite(bb_mid) else None)
                return e1, e2
        except Exception:
            return None, None

    # Build plan according to state with new setups
    if state == 'throwback_long' and direction == 'long':
        hh = (eb.evidence.price_breakout.ref or {}).get('hh') if hasattr(eb.evidence, 'price_breakout') else None
        entry, sl, tp, note = _retest_entry('long', float(hh) if hh is not None else price_now)
    elif state == 'trend_follow_pullback' and direction == 'long':
        # Prefer EMA20/BB mid zone from pullback evidence if available
        z = ((eb.evidence.__dict__.get('pullback') or {}).get('zone') if hasattr(eb.evidence, '__dict__') else None)
        if z and isinstance(z, (list, tuple)):
            entry = float((z[0] + z[1]) / 2.0); sl = _protective_sl(levels, ref_level=z[0], atr=atr, side='long'); tp = _nearest_band_tp(levels, price_now, side='long'); note = 'pullback_zone_entry'
        else:
            e1, e2 = _trend_follow_entries('long'); entry, entry2 = e1, e2; sl = _protective_sl(levels, ref_level=(entry - rules.trend_break_buf_atr*atr) if entry else price_now, atr=atr, side='long'); tp = _nearest_band_tp(levels, price_now, side='long'); note = 'trend_follow_pullback_fallback'
    elif state == 'false_breakout' and direction == 'short':
        ll = (eb.evidence.price_breakdown.ref or {}).get('ll') if hasattr(eb.evidence, 'price_breakdown') else None
        ref = ll if ll is not None else float(df1['low'].iloc[-2])
        entry, sl, tp, note = _retest_entry('short', float(ref))
    elif state == 'mean_reversion' and direction == 'long':
        ref = float(df1['low'].iloc[-2]); entry = price_now; sl = float(ref - 0.2*atr); tp = _nearest_band_tp(levels, price_now, side='long'); note = 'mean_reversion_rebound'
    elif state == 'rejection' and direction == 'long':
        ref = float(df1['low'].iloc[-2]); entry = price_now; sl = float(ref - 0.2*atr); tp = _nearest_band_tp(levels, price_now, side='long'); note = 'rejection_long'
    elif state == 'divergence_up' and direction == 'long':
        e1, e2 = _trend_follow_entries('long'); entry, entry2 = e1, e2; sl = _protective_sl(levels, ref_level=(entry - rules.trend_break_buf_atr*atr) if entry else price_now, atr=atr, side='long'); tp = _nearest_band_tp(levels, price_now, side='long'); note = 'divergence_break_entry'
    elif state == 'volatility_breakout' and direction == 'long':
        hh = (eb.evidence.price_breakout.ref or {}).get('hh'); entry, sl, tp, note = _retest_entry('long', float(hh) if hh is not None else price_now)
    elif direction == 'long':
        hh = (eb.evidence.price_breakout.ref or {}).get('hh')
        if state == 'breakout' and hh is not None:
            entry, sl, tp, note = _retest_entry('long', float(hh))
        elif state == 'reclaim':
            lvl = (eb.evidence.price_reclaim.ref or {}).get('level')
            entry, sl, tp, note = _retest_entry('long', float(lvl) if lvl is not None else float('nan'))
        else:
            # trend-follow entries
            e1, e2 = _trend_follow_entries('long')
            entry, entry2 = e1, e2
            if entry is not None:
                sl = _protective_sl(levels, ref_level=(entry - rules.trend_break_buf_atr*atr), atr=atr, side='long')
                tp = _nearest_band_tp(levels, price_now, side='long')
                note = "trend_follow: break + ema20/bb_mid"
    elif state == 'throwback_short' and direction == 'short':
        ll = (eb.evidence.price_breakdown.ref or {}).get('ll') if hasattr(eb.evidence, 'price_breakdown') else None
        entry, sl, tp, note = _retest_entry('short', float(ll) if ll is not None else price_now)
    elif state == 'trend_follow_pullback' and direction == 'short':
        z = ((eb.evidence.__dict__.get('pullback') or {}).get('zone') if hasattr(eb.evidence, '__dict__') else None)
        if z and isinstance(z, (list, tuple)):
            entry = float((z[0] + z[1]) / 2.0); sl = _protective_sl(levels, ref_level=z[1], atr=atr, side='short'); tp = _nearest_band_tp(levels, price_now, side='short'); note = 'pullback_zone_entry'
        else:
            e1, e2 = _trend_follow_entries('short'); entry, entry2 = e1, e2; sl = _protective_sl(levels, ref_level=(entry + rules.trend_break_buf_atr*atr) if entry else price_now, atr=atr, side='short'); tp = _nearest_band_tp(levels, price_now, side='short'); note = 'trend_follow_pullback_fallback'
    elif state == 'false_breakdown' and direction == 'long':
        hh = (eb.evidence.price_breakout.ref or {}).get('hh') if hasattr(eb.evidence, 'price_breakout') else None
        ref = hh if hh is not None else float(df1['high'].iloc[-2])
        entry, sl, tp, note = _retest_entry('long', float(ref))
    elif state == 'mean_reversion' and direction == 'short':
        ref = float(df1['high'].iloc[-2]); entry = price_now; sl = float(ref + 0.2*atr); tp = _nearest_band_tp(levels, price_now, side='short'); note = 'mean_reversion_snapback'
    elif state == 'rejection' and direction == 'short':
        ref = float(df1['high'].iloc[-2]); entry = price_now; sl = float(ref + 0.2*atr); tp = _nearest_band_tp(levels, price_now, side='short'); note = 'rejection_short'
    elif state == 'divergence_down' and direction == 'short':
        e1, e2 = _trend_follow_entries('short'); entry, entry2 = e1, e2; sl = _protective_sl(levels, ref_level=(entry + rules.trend_break_buf_atr*atr) if entry else price_now, atr=atr, side='short'); tp = _nearest_band_tp(levels, price_now, side='short'); note = 'divergence_break_entry'
    elif state == 'volatility_breakout' and direction == 'short':
        ll = (eb.evidence.price_breakdown.ref or {}).get('ll'); entry, sl, tp, note = _retest_entry('short', float(ll) if ll is not None else price_now)
    elif direction == 'short':
        ll = (eb.evidence.price_breakdown.ref or {}).get('ll')
        if state == 'breakdown' and ll is not None:
            entry, sl, tp, note = _retest_entry('short', float(ll))
        elif state == 'reclaim':
            lvl = (eb.evidence.price_reclaim.ref or {}).get('level')
            entry, sl, tp, note = _retest_entry('short', float(lvl) if lvl is not None else float('nan'))
        else:
            e1, e2 = _trend_follow_entries('short')
            entry, entry2 = e1, e2
            if entry is not None:
                sl = _protective_sl(levels, ref_level=(entry + rules.trend_break_buf_atr*atr), atr=atr, side='short')
                tp = _nearest_band_tp(levels, price_now, side='short')
                note = "trend_follow: break + ema20/bb_mid"
    else:
        # state undefined/sideways -> no explicit plan
        pass

    # Compute RR & proximity for primary/secondary entries (if present)
    if direction and entry is not None and sl is not None and tp is not None and atr > 0:
        rr = _rr(direction, entry, sl, tp)
        proximity_ok = (abs(price_now - entry) <= rules.retest_zone_atr * atr)
    # trend-follow secondary entry (EMA20/BB mid)
    if direction and 'entry2' in locals() and entry2 is not None and sl is not None and tp is not None and atr > 0:
        rr2 = _rr(direction, float(entry2), sl, tp)
        proximity_ok2 = (abs(price_now - float(entry2)) <= rules.proximity_atr * atr)



    # AVOID conditions
    rsi = float(f1.get('momentum', {}).get('rsi', 50.0))
    div = (f1.get('momentum', {}).get('divergence') or 'none')
    vz = float(f1.get('volume', {}).get('vol_z20', 0.0))

    rsi_extreme = (rsi >= rules.rsi_overbought) or (rsi <= rules.rsi_oversold)
    div_against = (direction == 'long' and div == 'bearish') or (direction == 'short' and div == 'bullish')
    vol_blowoff = (vz >= rules.vol_z_hot)
    rr_bad = (rr is not None and rr < rules.rr_avoid)
    liq_heavy_close = (not liq_ok) and (eb.evidence.liquidity.why or '').startswith('heavy_zone')

    avoid_reasons = []
    if rsi_extreme: avoid_reasons.append('rsi_extreme')
    if div_against: avoid_reasons.append('rsi_divergence_against')
    if vol_blowoff: avoid_reasons.append('volume_blowoff')
    if rr_bad: avoid_reasons.append('rr_bad')
    if liq_heavy_close: avoid_reasons.append('heavy_liquidity_ahead')

    # Decision tree with revised proximity rules (allow either entry or entry2)
    decision = 'WAIT'
    if avoid_reasons:
        decision = 'AVOID'
    else:
        any_prox = proximity_ok or proximity_ok2
        rr_ok_any = (rr is not None and rr >= rules.rr_min) or (rr2 is not None and rr2 >= rules.rr_min)
        if all(req_ok) and any_prox and rr_ok_any:
            decision = 'ENTER'
        else:
            if not all(req_ok):
                pass
            if not any_prox:
                miss_reasons.append('price_far_from_entry')
            if not rr_ok_any:
                miss_reasons.append('rr_min')

    # Build plan out
    plan = PlanOut(
        direction=direction,
        entry=_smart_round(entry) if isinstance(entry, (int,float)) else None,
        sl=_smart_round(sl) if isinstance(sl, (int,float)) else None,
        tp=_smart_round(tp) if isinstance(tp, (int,float)) else None,
        rr=round(rr, 3) if isinstance(rr, (int,float)) else None,
        entry2=_smart_round(entry2) if 'entry2' in locals() and isinstance(entry2, (int,float)) else None,
        rr2=round(rr2, 3) if isinstance(rr2, (int,float)) else None,
        note=note or None,
    )

    # Logs for three buckets
    logs = LogsOut(
        ENTER={
            'required_ok': all(req_ok),
            'proximity_ok_primary': proximity_ok,
            'proximity_ok_secondary': proximity_ok2,
            'rr_ok_primary': (rr is not None and rr >= rules.rr_min),
            'rr_ok_secondary': (rr2 is not None and rr2 >= rules.rr_min),
            'plan': plan.model_dump() if hasattr(plan, 'model_dump') else plan.__dict__,
        },
        WAIT={
            'missing': sorted(set(miss_reasons)),
            'vol_ok': vol_ok,
            'momentum_ok': mom_ok,
            'candles_ok': cdl_ok,
            'liquidity_ok': liq_ok,
        },
        AVOID={
            'reasons': avoid_reasons,
            'rsi': rsi,
            'divergence': div,
            'vol_z20': vz,
            'rr': rr,
            'rr2': rr2,
        },
    )

    # Telegram signal when ENTER (include both entries if applicable)
    telegram_signal = None
    if decision == 'ENTER' and direction and plan.sl is not None and plan.tp is not None and (plan.entry is not None or plan.entry2 is not None):
        strategy = state.capitalize()
        entry_lines = []
        if plan.entry is not None:
            entry_lines.append(f"Entry: {plan.entry}")
        if plan.entry2 is not None:
            entry_lines.append(f"Entry2: {plan.entry2}")
        entries_text = "\n".join(entry_lines) if entry_lines else ""
        telegram_signal = (
            f"{direction.upper()} | {symbol}\n"
            f"Strategy: {strategy} ({timeframe})\n"
            f"{entries_text}\n"
            f"SL: {plan.sl}\n"
            f"TP: {plan.tp}"
        )
    
    out = DecisionOut(
        symbol=symbol,
        timeframe=timeframe,
        asof=evidence_bundle.get('asof'),
        state=state,
        confidence=round(confidence, 3),
        decision=decision,
        plan=plan,
        logs=logs,
        telegram_signal=telegram_signal,
    )
    
    return out.model_dump() if hasattr(out, 'model_dump') else out.__dict__

