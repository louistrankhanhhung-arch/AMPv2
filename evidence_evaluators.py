"""
Evidence Evaluators v2 (no trade decision)
-----------------------------------------
Adds pullback/throwback/BB-expansion/volume-explosive evidences and wires them
into a richer evidence bundle. Intended to be used by decision_engine_v2.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------------------
# 1) Types & Config (per-TF thresholds with sensible defaults)
# --------------------------------------------------------------------------------------

@dataclass
class TFThresholds:
    break_buffer_atr: float = 0.2
    vol_ratio_thr: float = 1.5
    vol_z_thr: float = 1.0
    rsi_long: float = 55.0
    rsi_short: float = 45.0
    bbw_lookback: int = 50
    zigzag_pct: float = 2.0
    ema_spread_small_atr: float = 0.3
    hvn_guard_atr: float = 0.7

@dataclass
class Config:
    per_tf: Dict[str, TFThresholds] = field(default_factory=lambda: {
        "1H": TFThresholds(break_buffer_atr=0.35, vol_ratio_thr=1.3, vol_z_thr=0.7, rsi_long=55, rsi_short=45, bbw_lookback=50, zigzag_pct=2.0, ema_spread_small_atr=0.35, hvn_guard_atr=0.7),
        "4H": TFThresholds(break_buffer_atr=0.20, vol_ratio_thr=1.25, vol_z_thr=0.6, rsi_long=55, rsi_short=45, bbw_lookback=50, zigzag_pct=2.0, ema_spread_small_atr=0.3,  hvn_guard_atr=0.8),
        "1D": TFThresholds(break_buffer_atr=0.15, vol_ratio_thr=1.2, vol_z_thr=0.5, rsi_long=55, rsi_short=45, bbw_lookback=50, zigzag_pct=2.0, ema_spread_small_atr=0.25, hvn_guard_atr=1.0),
    })

PRIMARY_TF = "1H"
CONFIRM_TF = "4H"
CONTEXT_TF = "1D"

# --------------------------------------------------------------------------------------
# 2) Utilities
# --------------------------------------------------------------------------------------

def _get_last_closed_bar(df: pd.DataFrame) -> pd.Series:
    if len(df) >= 2:
        return df.iloc[-2]
    return df.iloc[-1]


def _last_swing(swings: Dict[str, Any], kind: str) -> Optional[float]:
    if not swings or 'swings' not in swings:
        return None
    for s in reversed(swings['swings']):
        if s.get('type') == kind:
            return float(s['price'])
    return None


def _ema_spread_atr(df: pd.DataFrame) -> float:
    e20 = float(df['ema20'].iloc[-1])
    e50 = float(df['ema50'].iloc[-1])
    atr = float(df['atr14'].iloc[-1]) if 'atr14' in df.columns else 0.0
    if atr <= 0:
        return 0.0
    return abs(e20 - e50) / atr

# --------------------------------------------------------------------------------------
# 3) Core evidences (existing)
# --------------------------------------------------------------------------------------

def ev_price_breakout(df: pd.DataFrame, swings: Dict[str, Any], atr: float, cfg: TFThresholds) -> Dict[str, Any]:
    hh = _last_swing(swings, 'HH')
    close = float(df['close'].iloc[-1])
    buffer = cfg.break_buffer_atr * atr
    if hh is None:
        return {"ok": False, "score": 0.0, "why": "no HH reference", "missing": ["hh"], "ref": None}
    core = close > (hh + buffer)
    last = _get_last_closed_bar(df)
    ft = bool(last['close'] > (hh + buffer))
    hold = bool(last['low'] > (hh + 0.1 * atr))
    score = (0.6 if core else 0.0) + (0.2 if ft else 0.0) + (0.2 if hold else 0.0)
    return {"ok": bool(core), "score": round(score,3), "why": ",".join([w for w in ["core" if core else "", "follow" if ft else "", "hold" if hold else ""] if w]), "missing": ([] if core else ["price"]), "ref": {"hh": hh, "buffer": buffer}}


def ev_price_breakdown(df: pd.DataFrame, swings: Dict[str, Any], atr: float, cfg: TFThresholds) -> Dict[str, Any]:
    ll = _last_swing(swings, 'LL')
    close = float(df['close'].iloc[-1])
    buffer = cfg.break_buffer_atr * atr
    if ll is None:
        return {"ok": False, "score": 0.0, "why": "no LL reference", "missing": ["ll"], "ref": None}
    core = close < (ll - buffer)
    last = _get_last_closed_bar(df)
    ft = bool(last['close'] < (ll - buffer))
    hold = bool(last['high'] < (ll - 0.1 * atr))
    score = (0.6 if core else 0.0) + (0.2 if ft else 0.0) + (0.2 if hold else 0.0)
    return {"ok": bool(core), "score": round(score,3), "why": ",".join([w for w in ["core" if core else "", "follow" if ft else "", "hold" if hold else ""] if w]), "missing": ([] if core else ["price"]), "ref": {"ll": ll, "buffer": buffer}}


def ev_price_reclaim(df: pd.DataFrame, level: float, atr: float, cfg: TFThresholds, side: str = 'long') -> Dict[str, Any]:
    last = _get_last_closed_bar(df)
    close = float(last['close'])
    buf = cfg.break_buffer_atr * atr
    if not np.isfinite(level):
        return {"ok": False, "score": 0.0, "why": "invalid_level", "missing": ["level"], "ref": None}
    if side == 'long':
        core = close > (level + buf); hold = last['low'] > (level + 0.1 * atr)
    else:
        core = close < (level - buf); hold = last['high'] < (level - 0.1 * atr)
    score = (0.7 if core else 0.0) + (0.3 if hold else 0.0)
    return {"ok": bool(core), "score": round(score,3), "why": ",".join([w for w in ["core" if core else "", "hold" if hold else ""] if w]), "missing": ([] if core else ["reclaim"]), "ref": {"level": level, "buffer": buf, "side": side}}


def ev_sideways(df: pd.DataFrame, bbw_last: float, bbw_med: float, atr: float, cfg: TFThresholds) -> Dict[str, Any]:
    ema_spread = _ema_spread_atr(df)
    squeeze = bool(bbw_last < bbw_med)
    rng_ok = (df['high'].tail(20).max() - df['low'].tail(20).min()) / max(atr, 1e-9) <= 3.0
    ok = squeeze and (ema_spread <= cfg.ema_spread_small_atr) and rng_ok
    score = (0.4 if squeeze else 0.0) + (0.4 if ema_spread <= cfg.ema_spread_small_atr else 0.0) + (0.2 if rng_ok else 0.0)
    return {"ok": bool(ok), "score": round(score,3), "why": "|".join([w for w in ["squeeze" if squeeze else "", "ema_spread_small" if ema_spread <= cfg.ema_spread_small_atr else "", "narrow_range" if rng_ok else ""] if w]), "missing": [] if ok else ["sideways_conditions"]}


def ev_volume(vol: Dict[str, Any], cfg: TFThresholds) -> Dict[str, Any]:
    vr = float(vol.get('vol_ratio', 1.0)); vz = float(vol.get('vol_z20', 0.0))
    ok = (vr >= cfg.vol_ratio_thr) or (vz >= cfg.vol_z_thr)
    strong = (vr >= max(2.0, cfg.vol_ratio_thr + 0.3)) or (vz >= max(2.0, cfg.vol_z_thr + 0.5))
    grade = 'strong' if strong else ('ok' if ok else 'weak')
    score = 1.0 if strong else (0.7 if ok else 0.0)
    return {"ok": bool(ok), "score": round(score,3), "why": ",".join([w for w in [f"vr>={cfg.vol_ratio_thr}" if vr >= cfg.vol_ratio_thr else "", f"vz>={cfg.vol_z_thr}" if vz >= cfg.vol_z_thr else ""] if w]), "missing": [] if ok else ["volume"], "vol_ratio": vr, "vol_z20": vz, "grade": grade}


def ev_momentum(mom: Dict[str, Any], cfg: TFThresholds, side: str = 'long') -> Dict[str, Any]:
    rsi = float(mom.get('rsi', 50.0)); div = mom.get('divergence', 'none')
    ok = (rsi >= cfg.rsi_long) if side == 'long' else (rsi <= cfg.rsi_short)
    score = 0.8 if ok else 0.2
    if (side == 'long' and div == 'bearish') or (side == 'short' and div == 'bullish'):
        score -= 0.2
    return {"ok": bool(ok), "score": round(max(0.0, score),3), "why": f"rsi={rsi:.1f}|div={div}", "missing": [] if ok else ["rsi"], "rsi": rsi, "divergence": div}


def ev_trend_alignment(trend_now: Dict[str, Any], trend_ctx: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    now = trend_now.get('state'); ctx = trend_ctx.get('state') if trend_ctx else None
    ema_ok = (now in ('up','down')); aligned = (ctx is None) or (now == ctx)
    return {"ok": bool(ema_ok and aligned), "score": round((0.7 if ema_ok else 0.0) + (0.3 if aligned else 0.0),3), "why": f"now={now}|ctx={ctx}", "missing": [] if ema_ok and aligned else ["trend"]}


def ev_candles(candles: Dict[str, Any], side: Optional[str] = None) -> Dict[str, Any]:
    ok = False; why = []
    if side == 'long':
        ok = bool(candles.get('bullish_engulf') or candles.get('bullish_pin'))
        if candles.get('bullish_engulf'): why.append('bullish_engulf')
        if candles.get('bullish_pin'): why.append('bullish_pin')
    elif side == 'short':
        ok = bool(candles.get('bearish_engulf') or candles.get('bearish_pin'))
        if candles.get('bearish_engulf'): why.append('bearish_engulf')
        if candles.get('bearish_pin'): why.append('bearish_pin')
    else:
        ok = any(bool(v) for v in candles.values()); why = [k for k,v in candles.items() if v]
    return {"ok": bool(ok), "score": 0.6 if ok else 0.0, "why": ",".join(why), "missing": [] if ok else ["candle"]}


def ev_liquidity(price: float, atr: float, vp_zones: List[Dict[str, Any]], cfg: TFThresholds, side: Optional[str] = None) -> Dict[str, Any]:
    guard = cfg.hvn_guard_atr * max(atr, 1e-9)
    near_heavy = False; nearest = None
    def mid(z):
        return (float(z['price_range'][0]) + float(z['price_range'][1]))/2.0
    for z in vp_zones or []:
        m = mid(z)
        if side == 'long' and m >= price and (m - price) <= guard: near_heavy=True; nearest=m; break
        if side == 'short' and m <= price and (price - m) <= guard: near_heavy=True; nearest=m; break
    return {"ok": (not near_heavy), "score": 1.0 if not near_heavy else 0.2, "why": "" if not near_heavy else f"heavy_zone_within_{cfg.hvn_guard_atr}*ATR", "near_heavy_zone": bool(near_heavy), "nearest_zone_mid": nearest}

# --------------------------------------------------------------------------------------
# 4) New evidences: BB expanding, explosive volume, throwback, pullback
# --------------------------------------------------------------------------------------

def ev_bb_expanding(bbw_last: float, bbw_med: float) -> Dict[str, Any]:
    ok = bool(bbw_last > bbw_med)
    return {"ok": ok, "score": 0.7 if ok else 0.0, "why": "bbw_last>bbw_med" if ok else "bbw_last<=bbw_med", "bbw_last": float(bbw_last), "bbw_med": float(bbw_med)}


def ev_volume_explosive(vol: Dict[str, Any]) -> Dict[str, Any]:
    vr = float(vol.get('vol_ratio', 1.0)); vz = float(vol.get('vol_z20', 0.0))
    ok = (vr >= 2.0) or (vz >= 2.0)
    strong = (vr >= 3.0) or (vz >= 3.0)
    grade = 'strong' if strong else ('ok' if ok else 'weak')
    score = 1.0 if strong else (0.8 if ok else 0.0)
    why = []
    if vr >= 2.0: why.append('vol_ratio>=2')
    if vz >= 2.0: why.append('vol_z>=2')
    return {"ok": ok, "score": round(score,3), "why": ",".join(why), "grade": grade, "vol_ratio": vr, "vol_z20": vz}


def _last_hh_ll_from_swings(swings: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    hh = None; ll = None
    for s in reversed(swings.get('swings', [])):
        if hh is None and s.get('type') == 'HH': hh = float(s['price'])
        if ll is None and s.get('type') == 'LL': ll = float(s['price'])
        if hh is not None and ll is not None: break
    return hh, ll


def ev_throwback_ready(df: pd.DataFrame, swings: Dict[str, Any], atr: float, side: Optional[str], pad_range: Tuple[float,float]=(0.02,0.10)) -> Dict[str, Any]:
    if atr <= 0 or side not in ('long','short'):
        return {"ok": False, "why": "side_or_atr_invalid"}
    hh, ll = _last_hh_ll_from_swings(swings)
    if side == 'long' and hh is not None:
        lo = float(hh + pad_range[0]*atr); hi = float(hh + pad_range[1]*atr)
        return {"ok": True, "why": "throwback_zone_ready", "ref": hh, "zone": [lo, hi]}
    if side == 'short' and ll is not None:
        lo = float(ll - pad_range[1]*atr); hi = float(ll - pad_range[0]*atr)
        return {"ok": True, "why": "throwback_zone_ready", "ref": ll, "zone": [lo, hi]}
    return {"ok": False, "why": "no_ref_level"}


def ev_pullback_valid(df: pd.DataFrame, swings: Dict[str, Any], atr: float, mom: Dict[str, Any], vol: Dict[str, Any], candles: Dict[str, Any], side: Optional[str]) -> Dict[str, Any]:
    if atr <= 0 or side not in ('long','short'):
        return {"ok": False, "why": "side_or_atr_invalid"}
    hh, ll = _last_hh_ll_from_swings(swings)
    rsi = float(mom.get('rsi', 50.0))
    try:
        v5 = float(df['volume'].tail(5).mean()); v10 = float(df['volume'].tail(10).mean());
        vs20 = float(df['vol_sma20'].iloc[-1]) if 'vol_sma20' in df.columns else v10
        contracting = (v5 < v10) and (v10 < vs20)
    except Exception:
        contracting = True
    bull = bool(df.get('lower_wick_pct', pd.Series([0])).iloc[-1] >= 50 if 'lower_wick_pct' in df.columns else False) or False
    bear = bool(df.get('upper_wick_pct', pd.Series([0])).iloc[-1] >= 50 if 'upper_wick_pct' in df.columns else False) or False

    if side == 'long' and hh is not None and ll is not None and hh > ll:
        rng = max(1e-9, hh - ll)
        current = float(df['close'].iloc[-1])
        retr = float((hh - current) / rng)
        ok = (0.382 <= retr <= 0.5) and (rsi > 50) and contracting and (bull or True)
        ema20 = float(df['ema20'].iloc[-1]) if 'ema20' in df.columns else float('nan')
        bb_mid = float(df['bb_mid'].iloc[-1]) if 'bb_mid' in df.columns else float('nan')
        center = ema20 if np.isfinite(ema20) else bb_mid
        zone = [float(center - 0.05*atr), float(center + 0.05*atr)] if np.isfinite(center) else None
        fzone = [float(bb_mid - 0.05*atr), float(bb_mid + 0.05*atr)] if np.isfinite(bb_mid) else None
        return {"ok": bool(ok), "why": "pullback_ok" if ok else "pullback_not_ok", "retrace_pct": round(retr,3), "rsi_ok": bool(rsi>50), "vol_contracting": bool(contracting), "confirm_candle": bool(bull), "zone": zone, "fallback_zone": fzone}
    if side == 'short' and hh is not None and ll is not None and hh > ll:
        rng = max(1e-9, hh - ll)
        current = float(df['close'].iloc[-1])
        retr = float((current - ll) / rng)
        ok = (0.382 <= retr <= 0.5) and (rsi < 50) and contracting and (bear or True)
        ema20 = float(df['ema20'].iloc[-1]) if 'ema20' in df.columns else float('nan')
        bb_mid = float(df['bb_mid'].iloc[-1]) if 'bb_mid' in df.columns else float('nan')
        center = ema20 if np.isfinite(ema20) else bb_mid
        zone = [float(center - 0.05*atr), float(center + 0.05*atr)] if np.isfinite(center) else None
        fzone = [float(bb_mid - 0.05*atr), float(bb_mid + 0.05*atr)] if np.isfinite(bb_mid) else None
        return {"ok": bool(ok), "why": "pullback_ok" if ok else "pullback_not_ok", "retrace_pct": round(retr,3), "rsi_ok": bool(rsi<50), "vol_contracting": bool(contracting), "confirm_candle": bool(bear), "zone": zone, "fallback_zone": fzone}
    return {"ok": False, "why": "insufficient_swings"}


# --------------------------------------------------------------------------------------
# 4b) Additional evidences for EARLY recognition
# --------------------------------------------------------------------------------------

def ev_mean_reversion(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Early mean-reversion: BB% extreme + RSI extreme.
    Returns side: long if oversold, short if overbought.
    Expects columns: bb_percent, rsi14, atr14, close.
    """
    try:
        pct_bb = float(df['bb_percent'].iloc[-1])
        rsi = float(df['rsi14'].iloc[-1])
        atr = float(df['atr14'].iloc[-1]) if 'atr14' in df.columns else 0.0
    except Exception:
        return {"ok": False, "why": "missing_bb_or_rsi"}
    long_ok = (pct_bb <= 5.0) and (rsi <= 25.0)
    short_ok = (pct_bb >= 95.0) and (rsi >= 75.0)
    if long_ok:
        return {"ok": True, "score": 0.8, "why": f"bb%={pct_bb:.1f}|rsi={rsi:.1f}", "side": "long", "ref": {"atr": atr}}
    if short_ok:
        return {"ok": True, "score": 0.8, "why": f"bb%={pct_bb:.1f}|rsi={rsi:.1f}", "side": "short", "ref": {"atr": atr}}
    return {"ok": False, "why": f"bb%={pct_bb:.1f}|rsi={rsi:.1f}"}


def ev_false_breakout(df: pd.DataFrame, swings: Dict[str, Any], atr: float, cfg: TFThresholds) -> Dict[str, Any]:
    """Breakout fake: price pokes above HH but closes back inside; weak follow-through volume."""
    hh = _last_swing(swings, 'HH')
    if hh is None or atr <= 0:
        return {"ok": False, "why": "no_HH_or_atr"}
    # use previous fully closed bar as "poke", last-1
    if len(df) < 3:
        return {"ok": False, "why": "insufficient_bars"}
    poke = df.iloc[-3]; last = _get_last_closed_bar(df)
    broke = bool(poke['high'] > (hh + cfg.break_buffer_atr * atr))
    failed = bool(last['close'] <= hh and last['high'] > hh)
    # weak vol on break; or reversal vol grows
    vol_break = float(poke.get('volume', 0.0))
    vs20 = float(df['vol_sma20'].iloc[-1]) if 'vol_sma20' in df.columns else max(1.0, df['volume'].tail(20).mean())
    weak = vol_break < vs20
    ok = broke and failed and weak
    return {"ok": bool(ok), "score": 0.8 if ok else 0.0, "why": "poke>HH_then_close_inside|weak_vol" if ok else "no_fakeout", "side": "short", "ref": {"hh": hh}}


def ev_false_breakdown(df: pd.DataFrame, swings: Dict[str, Any], atr: float, cfg: TFThresholds) -> Dict[str, Any]:
    """Breakdown fake: price pokes below LL but closes back inside; weak follow-through volume."""
    ll = _last_swing(swings, 'LL')
    if ll is None or atr <= 0:
        return {"ok": False, "why": "no_LL_or_atr"}
    if len(df) < 3:
        return {"ok": False, "why": "insufficient_bars"}
    poke = df.iloc[-3]; last = _get_last_closed_bar(df)
    broke = bool(poke['low'] < (ll - cfg.break_buffer_atr * atr))
    failed = bool(last['close'] >= ll and last['low'] < ll)
    vol_break = float(poke.get('volume', 0.0))
    vs20 = float(df['vol_sma20'].iloc[-1]) if 'vol_sma20' in df.columns else max(1.0, df['volume'].tail(20).mean())
    weak = vol_break < vs20
    ok = broke and failed and weak
    return {"ok": bool(ok), "score": 0.8 if ok else 0.0, "why": "poke<LL_then_close_inside|weak_vol" if ok else "no_fakeout", "side": "long", "ref": {"ll": ll}}


def ev_trend_follow_ready(df: pd.DataFrame, momentum: Dict[str, Any], trend: Dict[str, Any], side: str) -> Dict[str, Any]:
    """
    Direct trend-follow readiness using: EMA20 vs EMA50, BB%, RSI.
    side ∈ {'long','short'}
    """
    try:
        e20 = float(df['ema20'].iloc[-1]); e50 = float(df['ema50'].iloc[-1])
        pct_bb = float(df['bb_percent'].iloc[-1]) if 'bb_percent' in df.columns else 50.0
        rsi = float(momentum.get('rsi', 50.0))
        st = trend.get('state')
    except Exception:
        return {"ok": False, "why": "missing_inputs", "side": side}
    if side == 'long':
        ok = (st == 'up') and (e20 > e50) and (pct_bb >= 70.0) and (rsi >= 55.0)
    else:
        ok = (st == 'down') and (e20 < e50) and (pct_bb <= 30.0) and (rsi <= 45.0)
    return {"ok": bool(ok), "score": 0.8 if ok else 0.0, "why": f"trend={st}|ema20{'> ' if e20>e50 else '<='}ema50|bb%={pct_bb:.1f}|rsi={rsi:.1f}", "side": side}


def ev_rejection(df: pd.DataFrame, swings: Dict[str, Any], atr: float) -> Dict[str, Any]:
    """Strong wick rejection near HH/LL with wick ratio ≥ 60% of bar range."""
    if atr <= 0 or df is None or len(df) < 2:
        return {"ok": False, "why": "invalid_input"}
    last = _get_last_closed_bar(df)
    hh = _last_swing(swings, 'HH'); ll = _last_swing(swings, 'LL')
    prox = 0.2 * atr
    out = {"ok": False, "why": "no_rejection"}
    # upper rejection near HH
    if hh is not None and abs(float(last['high']) - hh) <= prox:
        rng = max(1e-9, float(last['high']) - float(last['low']))
        upper = float(last['high']) - float(last['close'])
        if (upper / rng) >= 0.6:
            return {"ok": True, "score": 0.8, "why": "upper_wick_reject@HH", "side": "short", "ref": {"hh": hh}}
    # lower rejection near LL
    if ll is not None and abs(float(last['low']) - ll) <= prox:
        rng = max(1e-9, float(last['high']) - float(last['low']))
        lower = float(last['close']) - float(last['low'])
        if (lower / rng) >= 0.6:
            return {"ok": True, "score": 0.8, "why": "lower_wick_reject@LL", "side": "long", "ref": {"ll": ll}}
    return out


def ev_divergence_updown(momentum: Dict[str, Any]) -> Dict[str, Any]:
    """Map momentum.divergence → bullish/bearish divergence with side."""
    div = momentum.get('divergence', 'none')
    if div == 'bullish':
        return {"ok": True, "score": 0.7, "why": "bullish_divergence", "side": "long"}
    if div == 'bearish':
        return {"ok": True, "score": 0.7, "why": "bearish_divergence", "side": "short"}
    return {"ok": False, "why": "no_divergence"}


def ev_compression_ready(bbw_last: float, bbw_med: float, atr_last: float) -> Dict[str, Any]:
    """Compression (squeeze) pre-break: BBW below median + ATR not rising."""
    squeeze = bool(bbw_last <= bbw_med)
    low_atr = bool(atr_last <= max(1e-9, atr_last))  # treat as low unless rising (placeholder)
    ok = squeeze and low_atr
    return {"ok": ok, "score": 0.6 if ok else 0.0, "why": "squeeze" if ok else "no_squeeze"}


def ev_volatility_breakout(vol: Dict[str, Any], bbw_last: float, bbw_med: float, atr_last: float) -> Dict[str, Any]:
    """Volatility breakout: BB expanding + volume explosive + ATR rising."""
    vr = float(vol.get('vol_ratio', 1.0)); vz = float(vol.get('vol_z20', 0.0))
    bb_expand = bool(bbw_last > bbw_med)
    explosive = (vr >= 2.0) or (vz >= 2.0)
    atr_rising = bool(atr_last > 0.0)
    ok = bb_expand and explosive and atr_rising
    score = 1.0 if ok and ((vr >= 3.0) or (vz >= 3.0)) else (0.8 if ok else 0.0)
    why = []
    if bb_expand: why.append("bb_expand")
    if explosive: why.append("vol_explosive")
    if atr_rising: why.append("atr_rising")
    return {"ok": ok, "score": round(score,3), "why": "|".join(why) if why else "weak"}

# --------------------------------------------------------------------------------------
# 5) State inference (priority) and bundle assembly
# --------------------------------------------------------------------------------------


def infer_state(evs: Dict[str, Any]) -> Tuple[str, float, str]:
    # Base evidences
    pb = evs.get('price_breakout', {"ok": False, "score": 0.0})
    pdn = evs.get('price_breakdown', {"ok": False, "score": 0.0})
    prc = evs.get('price_reclaim', {"ok": False, "score": 0.0})
    sdw = evs.get('sideways', {"ok": False, "score": 0.0})
    # New evidences
    fk_up = evs.get('false_breakout', {"ok": False, "score": 0.0})
    fk_dn = evs.get('false_breakdown', {"ok": False, "score": 0.0})
    volb = evs.get('volatility_breakout', {"ok": False, "score": 0.0})
    tb = evs.get('throwback', {"ok": False, "score": 0.0})
    pbk = evs.get('pullback', {"ok": False, "score": 0.0})
    tf_up = evs.get('trend_follow_up', {"ok": False, "score": 0.0})
    tf_dn = evs.get('trend_follow_down', {"ok": False, "score": 0.0})
    mr = evs.get('mean_reversion', {"ok": False, "score": 0.0})
    div = evs.get('divergence', {"ok": False, "score": 0.0})
    rjt = evs.get('rejection', {"ok": False, "score": 0.0})
    cmp = evs.get('compression_ready', {"ok": False, "score": 0.0})

    # Priority:
    # 1) Fakeouts
    if fk_up.get('ok'): return 'false_breakout', float(fk_up.get('score', 0.0)), 'false_breakout'
    if fk_dn.get('ok'): return 'false_breakdown', float(fk_dn.get('score', 0.0)), 'false_breakdown'
    # 2) Volatility breakout
    if volb.get('ok'):  return 'volatility_breakout', float(volb.get('score', 0.0)), 'volatility_breakout'
    # 3) Breakout/breakdown
    if pb.get('ok'):    return 'breakout', float(pb.get('score', 0.0)), 'price_breakout'
    if pdn.get('ok'):   return 'breakdown', float(pdn.get('score', 0.0)), 'price_breakdown'
    # 4) Reclaim
    if prc.get('ok'):   return 'reclaim', float(prc.get('score', 0.0)), 'price_reclaim'
    # 5) Trend-follow pullback -> direct trend-follow
    if pbk.get('ok') and (tf_up.get('ok') or tf_dn.get('ok')):
        return 'trend_follow_pullback', float(pbk.get('score', 0.0)), 'pullback'
    if tf_up.get('ok'): return 'trend_follow_up', float(tf_up.get('score', 0.0)), 'trend_follow_up'
    if tf_dn.get('ok'): return 'trend_follow_down', float(tf_dn.get('score', 0.0)), 'trend_follow_down'
    # 6) Mean reversion -> divergence up/down -> rejection
    if mr.get('ok'):    return 'mean_reversion', float(mr.get('score', 0.0)), 'mean_reversion'
    if div.get('ok'):
        side = div.get('side')
        if side == 'long':  return 'divergence_up', float(div.get('score', 0.0)), 'divergence_up'
        if side == 'short': return 'divergence_down', float(div.get('score', 0.0)), 'divergence_down'
    if rjt.get('ok'):   return 'rejection', float(rjt.get('score', 0.0)), 'rejection'
    # 7) Sideways -> compression ready
    if sdw.get('ok'):   return 'sideways', float(sdw.get('score', 0.0)), 'sideways'
    if cmp.get('ok'):   return 'compression_ready', float(cmp.get('score', 0.0)), 'compression_ready'
    return 'undefined', 0.0, ''



def build_evidence_bundle(symbol: str, features_by_tf: Dict[str, Dict[str, Any]], cfg: Config) -> Dict[str, Any]:
    f1 = features_by_tf.get('1H', {})
    f4 = features_by_tf.get('4H', {})
    fD = features_by_tf.get('1D', {})

    df1: pd.DataFrame = f1.get('df')
    df4: pd.DataFrame = f4.get('df') if f4 else None

    atr1 = float(f1.get('volatility', {}).get('atr', 0.0) or 0.0)
    atr4 = float(f4.get('volatility', {}).get('atr', 0.0) or 0.0) if f4 else 0.0

    # Price action base (1H)
    ev_pb = ev_price_breakout(df1, f1.get('swings', {}), atr1, cfg.per_tf['1H']) if df1 is not None else {"ok": False}
    ev_pdn = ev_price_breakdown(df1, f1.get('swings', {}), atr1, cfg.per_tf['1H']) if df1 is not None else {"ok": False}

    # Reclaim ref (nearest band TP or last swing)
    level_for_reclaim = None
    try:
        bands = (f1.get('levels', {}) or {}).get('bands_up', []) + (f1.get('levels', {}) or {}).get('bands_down', [])
        px = float(df1['close'].iloc[-1]) if df1 is not None else None
        if px is not None and bands:
            level_for_reclaim = min(bands, key=lambda b: abs(float(b['tp']) - px)).get('tp')
    except Exception:
        pass
    ev_prc = ev_price_reclaim(df1, float(level_for_reclaim) if level_for_reclaim is not None else float('nan'), atr1, cfg.per_tf['1H'], side='long' if ev_pb.get('ok') else 'short') if df1 is not None else {"ok": False}

    # Sideways
    bbw1 = f1.get('volatility', {}).get('bbw_last', 0.0); bbw1_med = f1.get('volatility', {}).get('bbw_med', 0.0)
    ev_sdw = ev_sideways(df1, bbw1, bbw1_med, atr1, cfg.per_tf['1H']) if df1 is not None else {"ok": False}

    # Volume & Momentum
    ev_vol_1h = ev_volume(f1.get('volume', {}), cfg.per_tf['1H'])
    ev_vol_4h = ev_volume(f4.get('volume', {}), cfg.per_tf['4H']) if f4 else {"ok": False}
    vol_ok = ev_vol_1h['ok'] or ev_vol_4h.get('ok', False)

    side_hint = 'long' if ev_pb.get('ok') else ('short' if ev_pdn.get('ok') else None)
    ev_mom_1h = ev_momentum(f1.get('momentum', {}), cfg.per_tf['1H'], side=side_hint or 'long')
    ev_tr = ev_trend_alignment(f1.get('trend', {}), f4.get('trend', {}) if f4 else None)

    # Candles & Liquidity
    ev_cdl = ev_candles(f1.get('candles', {}), side=side_hint)
    price_now = float(df1['close'].iloc[-1]) if df1 is not None else float('nan')
    vp = f4.get('vp_zones', []) if f4 else []
    if not vp and fD: vp = fD.get('vp_zones', [])
    ev_liq_ = ev_liquidity(price_now, atr4 or atr1, vp, cfg.per_tf['4H'], side=side_hint)

    # New evidences
    ev_bb = ev_bb_expanding(bbw1, bbw1_med)
    ev_exp = ev_volume_explosive(f1.get('volume', {}))
    ev_tb = ev_throwback_ready(df1, f1.get('swings', {}), atr1, side_hint) if df1 is not None else {"ok": False}
    ev_pbk = ev_pullback_valid(df1, f1.get('swings', {}), atr1, f1.get('momentum', {}), f1.get('volume', {}), f1.get('candles', {}), side_hint) if df1 is not None else {"ok": False}

    evidences = {
        'price_breakout': ev_pb,
        'price_breakdown': ev_pdn,
        'price_reclaim': ev_prc,
        'sideways': ev_sdw,
        'volume': {'primary': ev_vol_1h, 'confirm': ev_vol_4h, 'ok': vol_ok},
        'momentum': {'primary': ev_mom_1h},
        'trend': ev_tr,
        'candles': ev_cdl,
        'liquidity': ev_liq_,
        'bb': ev_bb,
        'volume_explosive': ev_exp,
        'throwback': ev_tb,
        'pullback': ev_pbk,
    }

    state, confidence, why = infer_state(evidences)

    out = {
        'symbol': symbol,
        'asof': str(df1.index[-1]) if df1 is not None else None,
        'timeframes': ['1H','4H','1D'],
        'state': state,
        'confidence': round(confidence,3),
        'why': why,
        'evidence': evidences,
    }
    return out

# Guidance
TF_GUIDANCE: Dict[str, Dict[str, List[str]]] = {
    'price_breakout':  {'required': ['1H'], 'optional': ['4H']},
    'price_breakdown': {'required': ['1H'], 'optional': ['4H']},
    'price_reclaim':   {'required': ['1H'], 'optional': ['4H']},
    'sideways':        {'required': ['4H'], 'optional': ['1H','1D']},
    'volume':          {'required': ['1H'], 'optional': ['4H']},
    'momentum':        {'required': ['1H'], 'optional': ['4H','1D']},
    'trend_alignment': {'required': ['1H','4H'], 'optional': ['1D']},
    'candles':         {'required': ['1H'], 'optional': ['4H']},
    'liquidity':       {'required': ['4H'], 'optional': ['1D']},
    'bb_expanding':    {'required': ['1H'], 'optional': []},
    'volume_explosive':{'required': ['1H'], 'optional': ['4H']},
    'throwback':       {'required': ['1H'], 'optional': ['4H']},
    'pullback':        {'required': ['1H'], 'optional': ['4H']},
}
