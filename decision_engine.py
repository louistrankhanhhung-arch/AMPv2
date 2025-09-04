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
from datetime import datetime

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
    false_breakout: EvidenceItemIn
    false_breakdown: EvidenceItemIn
    trend_follow_up: EvidenceItemIn
    trend_follow_down: EvidenceItemIn
    mean_reversion: EvidenceItemIn
    divergence: EvidenceItemIn
    rejection: EvidenceItemIn
    compression_ready: EvidenceItemIn
    volatility_breakout: EvidenceItemIn
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
    entry: Optional[float] = None
    sl: Optional[float] = None
    tp: Optional[float] = None           # legacy = TP2
    rr: Optional[confloat(ge=0)] = None  # legacy RR at TP2

    # --- NEW: layered targets ---
    tp1: Optional[float] = None
    tp2: Optional[float] = None
    tp3: Optional[float] = None
    rr1: Optional[confloat(ge=0)] = None
    rr2: Optional[confloat(ge=0)] = None
    rr3: Optional[confloat(ge=0)] = None

    # Secondary entry (giữ nguyên)
    entry2: Optional[float] = None
    # Lưu ý: rr2 ở trên dùng cho layered; RR của entry2 (EMA/BB mid) nếu bạn cần thì đặt tên khác (vd: rr_entry2)
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
    
    headline: Optional[str] = None
    model_config = ConfigDict(extra="forbid")

# =====================================================
# 2) Decision rules
# =====================================================

@dataclass
class DecisionRules:
    rr_min: float = 1.5
    rr_max: float = 7.0            # chặn RR ảo
    rr_avoid: float = 1.2
    proximity_atr: float = 0.5  # price must be within 0.5*ATR of entry to ENTER (sửa 0.3 -> 0.5)
    vol_z_hot: float = 3.0
    rsi_overbought: float = 80.0
    rsi_oversold: float = 20.0
    hvn_avoid_atr: float = 0.3  # if heavy zone within 0.3*ATR → avoid
    # --- New: entry setup tuning ---
    retest_pad_atr: float = 0.05      # small pad above/below level for retest entry
    retest_zone_atr: float = 0.50     # acceptable distance from price to retest entry to consider ENTER (sửa 0.15 -> 0.3)
    retest_zone_atr_reclaim: float = 0.60  
    trend_break_buf_atr: float = 0.10 # trend-follow Entry1 uses break of nearest swing with this buffer (sửa 0.2 -> 0.1)
    sl_min_atr: float = 0.5       # SL tối thiểu = 0.5*ATR
    tp_ladder_n: int = 3           # số bậc TP
    # SL pads by setup type (A/B testable)
    sl_pad_breakout_atr: float = 0.5
    sl_pad_reclaim_atr: float = 1.0
    sl_pad_trend_follow_atr: float = 0.6
    sl_pad_mean_reversion_atr: float = 1.2
    # --- NEW: multi-TF confluence gates ---
    rsi1h_long: float = 55.0
    rsi4h_long_soft: float = 50.0
    rsi1h_long_ctr: float = 60.0    # yêu cầu khi countertrend 4H
    rsi1h_short: float = 45.0
    rsi4h_short_soft: float = 50.0
    rsi1h_short_ctr: float = 40.0   # yêu cầu khi countertrend 4H
    confluence_enter_thr: float = 0.50
    confluence_enter_thr_reclaim: float = 0.55
    confluence_bonus_ctx: float = 0.10   # thưởng khi 1D đồng hướng
    confluence_penalty_ctx: float = 0.15 # phạt khi 1D ngược hướng

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
    'breakout': ['price_breakout', 'volume', 'trend', 'bb|volatility_breakout'],
    'breakdown': ['price_breakdown', 'volume', 'trend', 'bb|volatility_breakout'],
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
    'compression_ready': ['compression_ready'],  # wait for break
    'volatility_breakout': ['volatility_breakout', 'bb'],
    'throwback_long': ['throwback'],
    'throwback_short': ['throwback'],
}

# --- Coarse state mapping (rút gọn 4 nhóm) ---
REQUIRED_BY_GROUP = {
    # TREND_BREAK: cần phá HH/LL + volume (hoặc BB bung/volatility breakout)
    'TREND_BREAK': ['price_breakout|price_breakdown', 'volume|bb|volatility_breakout'],
    # TREND_RETEST: cần tín hiệu hồi về vùng chuẩn + trend cùng hướng
    'TREND_RETEST': ['pullback|throwback', 'trend'],
    # REVERSAL: đảo chiều cần pattern mean-rev hoặc rejection
    'REVERSAL': ['rejection|mean_reversion'],
    # RANGE: sideway
    'RANGE': ['sideways'],
}

def classify_state_coarse(eb: EvidenceBundleIn) -> Tuple[str, List[str]]:
    """Gom state thành 4 nhóm và trả thêm tags (không ảnh hưởng gating cứng).
    Ưu tiên theo sức mạnh: TREND_BREAK > TREND_RETEST > REVERSAL > RANGE.
    """
    e = eb.evidence
    tags: List[str] = []
    try:
        if getattr(e, 'bb', None) and getattr(e.bb, 'ok', False):
            tags.append('bb_expanding')
    except Exception:
        pass
    try:
        if getattr(e, 'volatility_breakout', None) and getattr(e.volatility_breakout, 'ok', False):
            tags.append('volatility_breakout')
    except Exception:
        pass
    try:
        if getattr(e, 'false_breakout', None) and getattr(e.false_breakout, 'ok', False):
            tags.append('false_breakout')
        if getattr(e, 'false_breakdown', None) and getattr(e.false_breakdown, 'ok', False):
            tags.append('false_breakdown')
    except Exception:
        pass
    try:
        if getattr(e, 'trend', None) and getattr(e.trend, 'ok', False):
            # keep as tag only
            pass
    except Exception:
        pass
    try:
        if getattr(e, 'divergence', None) and getattr(e.divergence, 'ok', False):
            tags.append('divergence')
    except Exception:
        pass

    # Decide coarse group
    if getattr(e, 'price_breakout', None) and getattr(e.price_breakout, 'ok', False):
        return 'TREND_BREAK', tags
    if getattr(e, 'price_breakdown', None) and getattr(e.price_breakdown, 'ok', False):
        return 'TREND_BREAK', tags

    if (getattr(e, 'pullback', None) and getattr(e.pullback, 'ok', False)) or \
       (getattr(e, 'throwback', None) and getattr(e.throwback, 'ok', False)) or \
       (getattr(e, 'price_reclaim', None) and getattr(e.price_reclaim, 'ok', False)):
        return 'TREND_RETEST', tags

    if (getattr(e, 'rejection', None) and getattr(e.rejection, 'ok', False)) or \
       (getattr(e, 'mean_reversion', None) and getattr(e.mean_reversion, 'ok', False)):
        return 'REVERSAL', tags

    if getattr(e, 'sideways', None) and getattr(e.sideways, 'ok', False):
        return 'RANGE', tags

    # fallback by original state name
    st = (eb.state or '').lower()
    if st in ('breakout','breakdown'): return 'TREND_BREAK', tags
    if st in ('reclaim','trend_follow_up','trend_follow_down','trend_follow_pullback','throwback_long','throwback_short'):
        if st.startswith('trend_follow'): tags.append(st)
        return 'TREND_RETEST', tags
    if st in ('mean_reversion','rejection','false_breakout','false_breakdown','divergence_up','divergence_down'):
        if st.startswith('false_'): tags.append(st)
        return 'REVERSAL', tags
    if st in ('sideways','range','compression_ready'): return 'RANGE', tags
    return 'RANGE', tags


def infer_side_vote(features_by_tf: Dict[str, Dict[str, Any]], eb: EvidenceBundleIn) -> Tuple[Optional[str], float, Dict[str, float]]:
    """Chọn side bằng cơ chế vote có trọng số. Trả (side, vote_sum, breakdown).
    side: 'long' / 'short' / None (khi mơ hồ).
    """
    e = eb.evidence
    f1 = features_by_tf.get('1H', {}) or {}
    f4 = features_by_tf.get('4H', {}) or {}
    fD = features_by_tf.get('1D', {}) or {}

    votes = 0.0
    brk = {}

    # (A) Bằng chứng trực tiếp
    if getattr(e, 'price_breakout', None) and getattr(e.price_breakout, 'ok', False):
        votes += 3.0; brk['price_breakout'] = +3.0
    if getattr(e, 'price_breakdown', None) and getattr(e.price_breakdown, 'ok', False):
        votes -= 3.0; brk['price_breakdown'] = -3.0
    # reclaim side
    try:
        ref_side = (getattr(e.price_reclaim, 'ref', None) or {}).get('side')
        if ref_side == 'long': votes += 2.0; brk['reclaim'] = +2.0
        elif ref_side == 'short': votes -= 2.0; brk['reclaim'] = -2.0
    except Exception:
        pass
    # pullback/throwback + trend alignment
    try:
        tr1 = (f1.get('trend', {}) or {}).get('state')
        tr4 = (f4.get('trend', {}) or {}).get('state')
        if ((getattr(e, 'pullback', None) and getattr(e.pullback, 'ok', False)) or \
            (getattr(e, 'throwback', None) and getattr(e.throwback, 'ok', False))):
            if tr1 == tr4 == 'up': votes += 2.0; brk['pbk/thb_trend'] = +2.0
            if tr1 == tr4 == 'down': votes -= 2.0; brk['pbk/thb_trend'] = -2.0
    except Exception:
        pass

    # (B) Đồng pha đa khung
    try:
        if tr1 == tr4 == 'up': votes += 2.0; brk['align_1H4H'] = +2.0
        elif tr1 == tr4 == 'down': votes -= 2.0; brk['align_1H4H'] = -2.0
    except Exception:
        pass
    try:
        trD = (fD.get('trend', {}) or {}).get('state')
        if trD == 'up': votes += 1.0; brk['ctx_1D'] = +1.0
        elif trD == 'down': votes -= 1.0; brk['ctx_1D'] = -1.0
    except Exception:
        pass

    # (C) Bộ điều kiện phụ
    try:
        rsi = float((f1.get('momentum', {}) or {}).get('rsi', 50.0))
        if rsi >= 55.0: votes += 1.0; brk['rsi_bias'] = +1.0
        elif rsi <= 45.0: votes -= 1.0; brk['rsi_bias'] = -1.0
    except Exception:
        pass
    try:
        # volume as booster theo hướng break nếu có
        vol_ok = bool(getattr(e, 'volume', None) and getattr(e.volume, 'ok', False))
        if vol_ok and 'price_breakout' in brk and brk['price_breakout'] > 0:
            votes += 1.0; brk['vol_boost'] = +1.0
        if vol_ok and 'price_breakdown' in brk and brk['price_breakdown'] < 0:
            votes -= 1.0; brk['vol_boost'] = -1.0
    except Exception:
        pass
    try:
        div = (f1.get('momentum', {}) or {}).get('divergence', 'none')
        if div == 'bearish' and votes > 0: votes -= 1.0; brk['divergence_against'] = -1.0
        if div == 'bullish' and votes < 0: votes += 1.0; brk['divergence_against'] = +1.0
    except Exception:
        pass

    # (D) Slow-market guards
    try:
        adaptive = getattr(e, 'adaptive', None) or {}
        is_slow = bool(adaptive.get('is_slow', False))
        liq_floor = bool(adaptive.get('liquidity_floor', False))
        if is_slow:
            votes *= 0.8; brk['slow_market_damp'] = -0.2*abs(votes)
        if liq_floor:
            # phạt nhẹ tín hiệu theo-trend (dựa trên dấu hiện tại)
            if votes > 0: votes -= 1.0; brk['liquidity_floor_pen'] = -1.0
            elif votes < 0: votes += 1.0; brk['liquidity_floor_pen'] = +1.0
    except Exception:
        pass

    side = 'long' if votes >= 3.0 else ('short' if votes <= -3.0 else None)
    return side, float(votes), brk



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

def _band_overlap(band: Tuple[float,float], bands4h: List[Dict[str, Any]], tol: float) -> bool:
    """Kiểm tra band 1H có nằm trong/đụng band 4H (có nới tol) không."""
    if not bands4h:
        return True
    lo, hi = float(band[0]), float(band[1])
    lo -= tol; hi += tol
    for b in bands4h:
        blo, bhi = float(b['band'][0]), float(b['band'][1])
        inter = max(0.0, min(hi, bhi) - max(lo, blo))
        if inter > 0:
            return True
    return False

def _filter_forward_tps_by_4h(tps: List[float], side: str, levels4h: Dict[str, Any], atr: float) -> List[float]:
    """Giữ TP1H nào có nằm trong/sát band 4H; nếu thiếu thì trả lại danh sách gốc."""
    try:
        bands4h = levels4h.get('bands_up' if side == 'long' else 'bands_down') or []
        if not bands4h:
            return tps
        kept: List[float] = []
        tol = 0.2 * max(1e-9, atr)
        for tp in tps:
            for b in bands4h:
                blo, bhi = float(b['band'][0]), float(b['band'][1])
                if (blo - tol) <= tp <= (bhi + tol):
                    kept.append(tp); break
        return kept if kept else tps
    except Exception:
        return tps

def _protective_sl_confluence(levels1h: Dict[str, Any], levels4h: Optional[Dict[str, Any]], ref_level: float, atr: float, side: str, pad_atr: float = 0.3) -> Optional[float]:
    """Đặt SL dựa trên band chứa ref_level (ưu tiên 1H) nhưng tránh xuyên band 4H."""
    base = _protective_sl(levels1h, ref_level, atr, side, pad_atr)
    if base is None or not levels4h:
        return base
    # nếu SL vẫn nằm trong band 4H cùng phía thì đẩy qua mép đối diện một chút
    pad = float(pad_atr) * atr
    for b in (levels4h.get('bands_up', []) + levels4h.get('bands_down', [])):
        lo, hi = float(b['band'][0]), float(b['band'][1])
        if lo <= ref_level <= hi:
            if side == 'long':
                return float(min(lo, ref_level) - pad)
            else:
                return float(max(hi, ref_level) + pad)
    return base

def _layered_tps(levels: Dict[str, Any], side: str, ref_price: float, entry: float, atr: float) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Chọn tối đa 3 TP theo hướng (forward) tính từ ENTRY.
    Ưu tiên band sẵn có; nếu thiếu thì fallback về bội số ATR [1.0, 2.0, 3.0] từ ENTRY.
    Trả về (tp1, tp2, tp3).
    """
    try:
        buckets = levels.get('bands_up' if side == 'long' else 'bands_down') or []
        forward = []
        for b in buckets:
            tpv = float(b.get('tp', float('nan')))
            if not np.isfinite(tpv):
                continue
            if (side == 'long' and tpv > entry) or (side == 'short' and tpv < entry):
                forward.append(tpv)
        forward = sorted(set(forward), key=lambda x: abs(x - entry))
        chosen = forward[:3]

        # fallback nếu thiếu band
        multiples = [1.0, 2.0, 3.0]
        i = 0
        while len(chosen) < 3 and atr > 0 and i < len(multiples):
            m = multiples[i]
            if side == 'long':
                cand = float(entry + m * atr)
                chosen.append(cand)
            else:
                cand = float(entry - m * atr)
                chosen.append(cand)
            i += 1

        # sắp theo thứ tự thực thi (tăng dần cho long, giảm dần cho short)
        if side == 'long':
            chosen = sorted(chosen)[:3]
        else:
            chosen = sorted(chosen, reverse=True)[:3]

        # pad về đủ 3 phần tử
        while len(chosen) < 3:
            chosen.append(None)
        return tuple(chosen[:3])
    except Exception:
        return None, None, None

def _protective_sl(levels: Dict[str, Any], ref_level: float, atr: float, side: str, pad_atr: float = 0.3) -> Optional[float]:
    pad = float(pad_atr) * atr
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
    try:
        if direction == 'long':
            risk = max(1e-9, entry - sl)
            reward = max(0.0, tp - entry)
        elif direction == 'short':
            risk = max(1e-9, sl - entry)
            reward = max(0.0, entry - tp)
        else:
            return 0.0
        return float(reward / risk) if risk > 0 else 0.0
    except Exception:
        return 0.0


def _price(df: pd.DataFrame) -> float:
    return float(df['close'].iloc[-1])

# --- Helpers: enforce SL gap & build TP ladder ---
def _ensure_sl_gap(entry: float, sl: float, atr: float, side: str, rules: DecisionRules) -> float:
    """Đảm bảo khoảng cách SL tối thiểu theo ATR; giữ đúng phía."""
    min_gap = max(1e-9, rules.sl_min_atr * atr)
    if not np.isfinite(entry) or not np.isfinite(sl) or atr <= 0:
        return sl
    if side == 'long':
        gap = entry - sl
        return float(entry - min_gap) if gap < min_gap else float(sl)
    else:
        gap = sl - entry
        return float(entry + min_gap) if gap < min_gap else float(sl)

def _tp_ladder(levels: Dict[str, Any], entry: float, side: str, atr: float, n: int = 3) -> List[float]:
    """Chọn n TP: ưu tiên band forward; nếu thiếu, bù bằng ATR multiples."""
    bands = levels.get('bands_up' if side == 'long' else 'bands_down') or []
    fwd = []
    for b in bands:
        tp = float(b.get('tp', np.nan))
        if np.isfinite(tp):
            if (side == 'long' and tp > entry) or (side == 'short' and tp < entry):
                fwd.append(tp)
    fwd = sorted(fwd, key=lambda x: abs(x - entry))
    tps = fwd[:n]
    # backfill bằng ATR nếu thiếu
    if atr > 0 and len(tps) < n:
        mults = [1.0, 1.5, 2.0, 2.5]
        for m in mults:
            tp = entry + (m * atr if side == 'long' else -m * atr)
            if all(abs(tp - x) > 1e-6 for x in tps):
                tps.append(float(tp))
            if len(tps) >= n:
                break
    # đảm bảo đơn điệu theo hướng
    tps = sorted(tps, reverse=(side == 'short'))
    return tps[:n]

def _tp_ladder_confluence(levels1h: Dict[str, Any], levels4h: Optional[Dict[str, Any]], entry: float, side: str, atr: float, n: int = 3) -> List[float]:
    """TP ladder 1H nhưng lọc theo band 4H; thiếu thì fallback ATR như cũ."""
    tps = _tp_ladder(levels1h, entry, side, atr, n)
    if not levels4h:
        return tps
    kept = _filter_forward_tps_by_4h(tps, side, levels4h, atr)
    return kept if kept else tps

def _tp_ladder_rr(levels1h: Dict[str, Any],
                  levels4h: Optional[Dict[str, Any]],
                  entry: float, sl: float, side: str, atr: float,
                  rr_targets: Tuple[float, float, float] = (1.2, 2.0, 3.0),
                  snap_tol_atr: float = 0.2) -> List[float]:
    """Build TP ladder from RR targets, then snap to bands within ±snap_tol_atr*ATR, and filter by 4H bands."""
    try:
        if not (np.isfinite(entry) and np.isfinite(sl) and atr > 0):
            return _tp_ladder_confluence(levels1h, levels4h, entry, side, atr, 3)
        risk = (entry - sl) if side == 'long' else (sl - entry)
        if risk <= 0:
            return _tp_ladder_confluence(levels1h, levels4h, entry, side, atr, 3)
        # RR targets → raw
        raw = [entry + rr*risk if side=='long' else entry - rr*risk for rr in rr_targets]
        # snap vào band 1H
        tol = snap_tol_atr * max(atr, 1e-9)
        bands = levels1h.get('bands_up' if side=='long' else 'bands_down') or []
        snapped = []
        for t in raw:
            cands = [float(b['tp']) for b in bands
                     if np.isfinite(b.get('tp', np.nan))
                     and ((side=='long' and b['tp']>entry) or (side=='short' and b['tp']<entry))
                     and abs(float(b['tp'])-t) <= tol]
            snapped.append(min(cands, key=lambda x: abs(x-t)) if cands else float(t))
        # lọc theo 4H + đảm bảo đơn điệu
        snapped = _filter_forward_tps_by_4h(snapped, side, levels4h, atr)
        return sorted(set(snapped), reverse=(side=='short'))[:3]
    except Exception:
        return _tp_ladder_confluence(levels1h, levels4h, entry, side, atr, 3)


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
    # 4H & 1D cho confluence
    f4 = features_by_tf.get('4H', {}) or {}
    d4: Optional[pd.DataFrame] = f4.get('df')
    levels4h = f4.get('levels', {}) or {}
    fD = features_by_tf.get('1D', {}) or {}
    # Determine side by state
    orig_state = eb.state
    confidence = float(eb.confidence)
    coarse_state, coarse_tags = classify_state_coarse(eb)
    state = coarse_state  # use coarse state for decision
    direction, vote_sum, vote_breakdown = infer_side_vote(features_by_tf, eb)
    
    # Adaptive regime
    adaptive = getattr(eb.evidence, 'adaptive', None) or {}
    regime = adaptive.get('regime', 'normal')
    is_slow = bool(adaptive.get('is_slow', False))
    liquidity_floor = bool(adaptive.get('liquidity_floor', False))# Required confirmations per state
    retest_pad_local = getattr(rules, "retest_pad_atr", 0.05) * (1.4 if regime == "high" else 1.0)
             
    # Required confirmations
    req_ok: List[bool] = []
    miss_reasons: List[str] = []

    # Helper lấy evidence theo tên; chịu lỗi an toàn
    def _ev(name: str) -> Any:
        try:
            return getattr(eb.evidence, name)
        except Exception:
            return None

    if direction is None and 'direction_undecided' not in miss_reasons:
        miss_reasons.append('direction_undecided')
             
    # Lấy danh sách required theo state (ưu tiên nhóm coarse 4-state)
    if state in REQUIRED_BY_GROUP:
        req_keys = REQUIRED_BY_GROUP[state]
    else:
        req_keys = REQUIRED_BY_STATE.get(state, [])
    # Nếu vẫn rỗng, fallback an toàn theo nhóm hoặc theo direction
    if not req_keys:
        if state == 'TREND_RETEST':
            req_keys = REQUIRED_BY_GROUP['TREND_RETEST']
        elif state == 'TREND_BREAK':
            req_keys = REQUIRED_BY_GROUP['TREND_BREAK']
        elif state == 'RANGE':
            req_keys = REQUIRED_BY_GROUP['RANGE']
        elif direction == 'long':
            req_keys = ['price_breakout', 'volume', 'trend']
        elif direction == 'short':
            req_keys = ['price_breakdown', 'volume', 'trend']
    # Chỉ giữ các key thực sự tồn tại trong EvidenceIn hiện tại
    # Hỗ trợ OR group dạng 'a|b' (nếu một trong các key con tồn tại thì giữ)
    req_keys = [
        k for k in req_keys
        if (
            ('|' in k and any(getattr(eb.evidence, ak.strip(), None) is not None for ak in k.split('|')))
            or (getattr(eb.evidence, k, None) is not None)
        )
    ]

    # Special-case: for 'reclaim' (SIẾT LẠI): price_reclaim AND volume; momentum tùy theo 4H alignment
    if state == 'reclaim':
        pr_ok = bool(getattr(_ev('price_reclaim'), 'ok', False))
        vol_ok_tmp = bool(getattr(_ev('volume'), 'ok', False))
        # momentum bắt buộc nếu 4H đối hướng; nếu 4H cùng/side thì không bắt buộc
        mom_ev = _ev('momentum')
        mom_ok_tmp = bool(getattr(mom_ev, 'ok', False)) if mom_ev else False
        req_ok = [pr_ok, vol_ok_tmp]  # AND
        if not pr_ok: miss_reasons.append('price_reclaim')
        if not vol_ok_tmp: miss_reasons.append('volume')
    for k in req_keys:
            # Support OR groups encoded as 'a|b' (any one OK)
            if '|' in k:
                alts = [ak.strip() for ak in k.split('|') if ak.strip()]
                alt_found = False
                alt_ok = False
                for ak in alts:
                    ev = _ev(ak)
                    if ev is not None:
                        alt_found = True
                        if getattr(ev, 'ok', False):
                            alt_ok = True
                req_ok.append(alt_ok if alt_found else False)
                if not alt_ok:
                    miss_reasons.append('(' + ' OR '.join(alts) + ')')
            else:
                ev = _ev(k)
                if ev is None:
                    miss_reasons.append(k)
                    req_ok.append(False)
                    continue
                ok = getattr(ev, 'ok', False)
                if not ok:
                    miss_reasons.append(k)
                req_ok.append(ok)

    # Optional/guards (không phải required)
    vol_ok = bool(getattr(eb.evidence.volume, 'ok', False))
    tr_ok = bool(getattr(eb.evidence.trend, 'ok', False))
    mom_bundle = getattr(eb.evidence, 'momentum', None)
    mom_ok = bool(mom_bundle and getattr(mom_bundle, 'primary', None) and getattr(mom_bundle.primary, 'ok', False))
    cdl_ok = bool(getattr(eb.evidence.candles, 'ok', False))
    liq_ok = bool(getattr(eb.evidence.liquidity, 'ok', False))

    # Candidate entry plan (revised setups)
    note = ""
    # --- init plan vars early to avoid UnboundLocalError
    entry: Optional[float] = None
    entry2: Optional[float] = None
    sl: Optional[float] = None
    tp: Optional[float] = None
    rr: Optional[float] = None
    rr_entry2: Optional[float] = None
    proximity_ok = False
    proximity_ok2 = False

# ---------- Multi-TF Confluence (EMA/RSI/Volume + 1D context) ----------
    def _trend_state(tf: str) -> Optional[str]:
        try:
            return (features_by_tf.get(tf, {}).get('trend', {}) or {}).get('state')
        except Exception:
            return None
    def _rsi(tf: str) -> float:
        try:
            return float(features_by_tf.get(tf, {}).get('momentum', {}).get('rsi', 50.0))
        except Exception:
            return 50.0
    def _volume_ok(evb: EvidenceBundleIn) -> Tuple[bool, bool]:
        try:
            v = evb.evidence.volume
            p_ok = bool(getattr(v, 'primary', None) and v.primary.ok)
            c_ok = bool(getattr(v, 'confirm', None) and v.confirm and v.confirm.ok)
            return p_ok, c_ok
        except Exception:
            return False, False

    now_tr, tr4, trD = _trend_state('1H'), _trend_state('4H'), _trend_state('1D')
    r1, r4 = _rsi('1H'), _rsi('4H')
    vol1_ok, vol4_ok = _volume_ok(eb)

    # align score: mạnh khi 1H==4H; trung tính khi 4H=side; yếu khi đối hướng
    align = 0.0
    if now_tr in ('up','down'):
        if tr4 == now_tr: align = 1.0
        elif tr4 in (None, 'side'): align = 0.6
        else: align = 0.2
    # rsi gate theo direction sau khi suy ra
    def _rsi_gate(dirn: Optional[str]) -> float:
        if dirn == 'long':
            ok_base = (r1 >= rules.rsi1h_long) and (r4 >= rules.rsi4h_long_soft or tr4 in (None,'side') or tr4=='up')
            ok_ctr  = (r1 >= rules.rsi1h_long_ctr) if (tr4=='down') else True
            return 1.0 if (ok_base and ok_ctr) else (0.5 if r1 >= (rules.rsi1h_long-2) else 0.0)
        if dirn == 'short':
            ok_base = (r1 <= rules.rsi1h_short) and (r4 <= rules.rsi4h_short_soft or tr4 in (None,'side') or tr4=='down')
            ok_ctr  = (r1 <= rules.rsi1h_short_ctr) if (tr4=='up') else True
            return 1.0 if (ok_base and ok_ctr) else (0.5 if r1 <= (rules.rsi1h_short+2) else 0.0)
        return 0.0
    vol_score = (1.0 if vol1_ok else 0.0) + (0.5 if vol4_ok else 0.0)
    # context 1D bonus/penalty áp vào confidence về sau

# Helper to build a retest entry around a reference level
    def _retest_entry(side: str, ref: float, sl_pad_atr: float) -> Tuple[Optional[float], Optional[float], Optional[float], str]:
        if not np.isfinite(ref) or atr <= 0:
            return None, None, None, ""
        pad = retest_pad_local * atr
        # Adaptive retest pad: dùng retest_pad_local nếu có (được set bởi lớp adaptive),
        # nếu không thì fallback về rules.retest_pad_atr để giữ hành vi cũ.
        try:
            pad_k = retest_pad_local  # có thể được định nghĩa ở scope ngoài (adaptive layer)
        except NameError:
            pad_k = getattr(rules, "retest_pad_atr", 0.05)
        pad = float(pad_k) * float(atr)
        if side == 'long':
            e = float(ref + pad)
            s = _protective_sl_confluence(levels, levels4h, ref_level=ref, atr=atr, side='long', pad_atr=sl_pad_atr)
            # chọn TP theo ENTRY pivot (đúng hướng RR ngay cả khi chưa layer)
            t = _nearest_band_tp(levels, e, side='long')
            return e, s, t, "retest_of_level"
        else:
            e = float(ref - pad)
            s = _protective_sl_confluence(levels, levels4h, ref_level=ref, atr=atr, side='short', pad_atr=sl_pad_atr)
            t = _nearest_band_tp(levels, e, side='short')
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

    # use original granular state for plan building
    state = orig_state

    # Build plan according to state with new setups
    if state == 'throwback_long' and direction == 'long':
        hh = (eb.evidence.price_breakout.ref or {}).get('hh') if hasattr(eb.evidence, 'price_breakout') else None
        ref_value = float(hh) if hh is not None else price_now
        entry, sl, tp, note = _retest_entry('long', ref_value, rules.sl_pad_breakout_atr)
    elif state == 'trend_follow_pullback' and direction == 'long':
        # Prefer EMA20/BB mid zone from pullback evidence if available
        z = ((eb.evidence.__dict__.get('pullback') or {}).get('zone') if hasattr(eb.evidence, '__dict__') else None)
        if z and isinstance(z, (list, tuple)):
            entry = float((z[0] + z[1]) / 2.0); sl = _protective_sl(levels, ref_level=z[0], atr=atr, side='long', pad_atr=rules.sl_pad_trend_follow_atr)
            piv = float(entry) if entry is not None else price_now
            tp = _nearest_band_tp(levels, piv, side='long')
            note = 'pullback_zone_entry'
        else:
            e1, e2 = _trend_follow_entries('long'); entry, entry2 = e1, e2; sl = _protective_sl(levels, ref_level=(entry - rules.trend_break_buf_atr*atr) if entry else price_now, atr=atr, side='long', pad_atr=rules.sl_pad_trend_follow_atr)
            piv = float(entry) if entry is not None else price_now
            tp  = _nearest_band_tp(levels, piv, side='long')
            note = 'trend_follow_pullback_fallback'
    elif state == 'false_breakout' and direction == 'short':
        ll = (eb.evidence.price_breakdown.ref or {}).get('ll') if hasattr(eb.evidence, 'price_breakdown') else None
        ref = float(ll) if ll is not None else float(df1['low'].iloc[-2])
        entry, sl, tp, note = _retest_entry('short', ref, rules.sl_pad_breakout_atr)
    elif state == 'mean_reversion' and direction == 'long':
        ref = float(df1['low'].iloc[-2]); entry = price_now
        sl = _protective_sl_confluence(levels, levels4h, ref_level=ref, atr=atr, side='long', pad_atr=rules.sl_pad_mean_reversion_atr)
        piv = float(entry) if entry is not None else price_now
        tp = _nearest_band_tp(levels, piv, side='long')
        note = 'mean_reversion_rebound'
    elif state == 'rejection' and direction == 'long':
        ref = float(df1['low'].iloc[-2]); entry = price_now
        sl = _protective_sl_confluence(levels, levels4h, ref_level=ref, atr=atr, side='long', pad_atr=rules.sl_pad_mean_reversion_atr)
        piv = float(entry) if entry is not None else price_now
        tp = _nearest_band_tp(levels, piv, side='long')
        note = 'rejection_long'
    elif state == 'divergence_up' and direction == 'long':
        e1, e2 = _trend_follow_entries('long'); entry, entry2 = e1, e2
        sl = _protective_sl_confluence(levels, levels4h, ref_level=(entry - rules.trend_break_buf_atr*atr) if entry else price_now, atr=atr, side='long', pad_atr=rules.sl_pad_trend_follow_atr)
        piv = float(entry) if entry is not None else price_now
        tp = _nearest_band_tp(levels, piv, side='long')
        note = 'divergence_break_entry'
    elif state == 'volatility_breakout' and direction == 'long':
        hh = (eb.evidence.price_breakout.ref or {}).get('hh')
        entry, sl, tp, note = _retest_entry('long', float(hh) if hh is not None else price_now, rules.sl_pad_breakout_atr)
    elif direction == 'long':
        hh = (eb.evidence.price_breakout.ref or {}).get('hh')
        if state == 'breakout' and hh is not None:
            entry, sl, tp, note = _retest_entry('long', float(hh), rules.sl_pad_breakout_atr)
        elif state == 'reclaim':
            lvl = (eb.evidence.price_reclaim.ref or {}).get('level')
            entry, sl, tp, note = _retest_entry('long', float(lvl) if lvl is not None else float('nan'), rules.sl_pad_reclaim_atr)
        else:
            # trend-follow entries
            e1, e2 = _trend_follow_entries('long')
            entry, entry2 = e1, e2
            if entry is not None:
                sl = _protective_sl_confluence(levels, levels4h, ref_level=(entry - rules.trend_break_buf_atr*atr), atr=atr, side='long', pad_atr=rules.sl_pad_trend_follow_atr)
                piv = float(entry) if entry is not None else price_now
                tp = _nearest_band_tp(levels, piv, side='long')
                note = "trend_follow: break + ema20/bb_mid"
    elif state == 'throwback_short' and direction == 'short':
        ll = (eb.evidence.price_breakdown.ref or {}).get('ll') if hasattr(eb.evidence, 'price_breakdown') else None
        entry, sl, tp, note = _retest_entry('short', float(ll) if ll is not None else price_now, rules.sl_pad_breakout_atr)
    elif state == 'trend_follow_pullback' and direction == 'short':
        z = ((eb.evidence.__dict__.get('pullback') or {}).get('zone') if hasattr(eb.evidence, '__dict__') else None)
        if z and isinstance(z, (list, tuple)):
            entry = float((z[0] + z[1]) / 2.0)
            sl = _protective_sl_confluence(levels, levels4h, ref_level=z[1], atr=atr, side='short', pad_atr=rules.sl_pad_trend_follow_atr)
            piv = float(entry) if entry is not None else price_now
            tp  = _nearest_band_tp(levels, piv, side='short')
            note = 'pullback_zone_entry'
        else:
            e1, e2 = _trend_follow_entries('short'); entry, entry2 = e1, e2
            sl = _protective_sl_confluence(levels, levels4h, ref_level=(entry + rules.trend_break_buf_atr*atr) if entry else price_now, atr=atr, side='short', pad_atr=rules.sl_pad_trend_follow_atr)
            piv = float(entry) if entry is not None else price_now
            tp = _nearest_band_tp(levels, piv, side='short')
            note = 'trend_follow_pullback_fallback'
    elif state == 'false_breakdown' and direction == 'long':
        hh = (eb.evidence.price_breakout.ref or {}).get('hh') if hasattr(eb.evidence, 'price_breakout') else None
        ref = float(hh) if hh is not None else float(df1['high'].iloc[-2])
        entry, sl, tp, note = _retest_entry('long', float(ref), rules.sl_pad_breakout_atr)
    elif state == 'mean_reversion' and direction == 'short':
        ref = float(df1['high'].iloc[-2]); entry = price_now
        sl = _protective_sl_confluence(levels, levels4h, ref_level=ref, atr=atr, side='short', pad_atr=rules.sl_pad_mean_reversion_atr)
        piv = float(entry) if entry is not None else price_now
        tp  = _nearest_band_tp(levels, piv, side='short')
        note = 'mean_reversion_snapback'
    elif state == 'rejection' and direction == 'short':
        ref = float(df1['high'].iloc[-2]); entry = price_now
        sl = _protective_sl_confluence(levels, levels4h, ref_level=ref, atr=atr, side='short', pad_atr=rules.sl_pad_mean_reversion_atr)
        piv = float(entry) if entry is not None else price_now
        tp  = _nearest_band_tp(levels, piv, side='short')
        note = 'rejection_short'
    elif state == 'divergence_down' and direction == 'short':
        e1, e2 = _trend_follow_entries('short'); entry, entry2 = e1, e2
        sl = _protective_sl_confluence(levels, levels4h, ref_level=(entry + rules.trend_break_buf_atr*atr) if entry else price_now, atr=atr, side='short', pad_atr=rules.sl_pad_trend_follow_atr) 
        piv = float(entry) if entry is not None else price_now
        tp  = _nearest_band_tp(levels, piv, side='short')
        note = 'divergence_break_entry'
    elif state == 'volatility_breakout' and direction == 'short':
        ll = (eb.evidence.price_breakdown.ref or {}).get('ll')
        entry, sl, tp, note = _retest_entry('short', float(ll) if ll is not None else price_now, rules.sl_pad_breakout_atr)
    elif direction == 'short':
        ll = (eb.evidence.price_breakdown.ref or {}).get('ll')
        if state == 'breakdown' and ll is not None:
            entry, sl, tp, note = _retest_entry('short', float(ll), rules.sl_pad_breakout_atr)
        elif state == 'reclaim':
            lvl = (eb.evidence.price_reclaim.ref or {}).get('level')
            entry, sl, tp, note = _retest_entry('short', float(lvl) if lvl is not None else float('nan'), rules.sl_pad_reclaim_atr)
        else:
            e1, e2 = _trend_follow_entries('short')
            entry, entry2 = e1, e2
            if entry is not None:
                sl = _protective_sl_confluence(levels, levels4h, ref_level=(entry + rules.trend_break_buf_atr*atr), atr=atr, side='short', pad_atr=rules.sl_pad_trend_follow_atr)
                piv = float(entry) if entry is not None else price_now
                tp  = _nearest_band_tp(levels, piv, side='short')
                note = "trend_follow: break + ema20/bb_mid"
    else:
        # state undefined/sideways -> no explicit plan
        pass
      
        # --- post-build: proximity checks, layered TP, RR, decision ---

    # 1) Enforce minimal SL gap before RR calc
    if direction and entry is not None and sl is not None and atr > 0:
        sl = _ensure_sl_gap(entry, sl, atr, side=direction, rules=rules)

    # 2) Layered TP: RR-first then snap to bands (1H filtered by 4H). Pivot = ENTRY
    tp1 = tp2 = tp3 = None
    if direction in ('long','short') and atr > 0 and entry is not None and sl is not None:
        tps = _tp_ladder_rr(levels, levels4h, float(entry), float(sl), direction, atr)
        tp1 = float(tps[0]) if len(tps)>0 else None
        tp2 = float(tps[1]) if len(tps)>1 else None
        tp3 = float(tps[2]) if len(tps)>2 else None
        if tp is None: tp = tp2 or tp1 or tp3

    # switch back to coarse state for gating & proximity
    state = coarse_state

    # 3) RR & proximity cho entry chính
    rr = None
    rr1 = rr2 = rr3 = None
    proximity_ok = False
    if direction and isinstance(entry, (int, float)) and sl is not None and atr > 0:
        e1f = float(entry)
        if tp1 is not None: rr1 = _rr(direction, e1f, sl, float(tp1))
        if tp2 is not None: rr2 = _rr(direction, e1f, sl, float(tp2))
        if tp3 is not None: rr3 = _rr(direction, e1f, sl, float(tp3))
        # cap RR theo rr_max
        if isinstance(rr1, (int, float)) and rr1 > rules.rr_max: rr1 = float(rules.rr_max)
        if isinstance(rr2, (int, float)) and rr2 > rules.rr_max: rr2 = float(rules.rr_max)
        if isinstance(rr3, (int, float)) and rr3 > rules.rr_max: rr3 = float(rules.rr_max)
        # legacy rr = TP2 ưu tiên, rồi TP1, rồi TP3
        rr = rr2 if rr2 is not None else (rr1 if rr1 is not None else rr3)

        prox_thr = (rules.retest_zone_atr_reclaim if state == 'TREND_RETEST' else rules.retest_zone_atr)
        # regime-adaptive proximity
        prox_thr = (prox_thr * (0.8 if regime=='low' else (1.2 if regime=='high' else 1.0)))
        proximity_ok = (abs(price_now - e1f) <= prox_thr * atr)

   # Soft-gate: nếu RR1 quá thấp thì bỏ TP1 và backfill bằng RR cao hơn
    rr1_soft_min = max(1.0, rules.rr_min * 0.8)
    if rr1 is not None and rr1 < rr1_soft_min:
        kept = [tp for tp in [tp2, tp3] if tp is not None]
        extra = _tp_ladder_rr(levels, levels4h, float(entry), float(sl), direction, atr, rr_targets=(4.0,))
        if extra: kept.append(float(extra[0]))
        kept = sorted(set(kept), reverse=(direction=='short'))[:3]
        tp1, tp2, tp3 = (kept + [None, None, None])[:3]


    # 4) trend-follow secondary entry (EMA20/BB mid) nếu có
    rr_entry2 = None
    proximity_ok2 = False
    e2 = locals().get('entry2', None)
    if direction and isinstance(e2, (int, float)) and sl is not None and tp is not None and atr > 0:
        e2f = float(e2)
        rr_entry2 = _rr(direction, e2f, sl, float(tp))
        if rr_entry2 is not None and rr_entry2 > rules.rr_max:
            rr_entry2 = float(rules.rr_max)
        proximity_ok2 = (abs(price_now - e2f) <= (rules.proximity_atr * (0.8 if regime=='low' else (1.2 if regime=='high' else 1.0))) * atr)

    # 5) Multi-TF confluence (EMA/RSI/Volume đã tính trước đó)
    #    - align, vol_score, _rsi_gate(direction), trD đã được chuẩn bị ở phần trên
    rsi_score = _rsi_gate(direction)
    confluence_score = min(1.0, 0.45*align + 0.35*rsi_score + 0.20*vol_score)

    # 6) 1D context: bonus/penalty confidence
    if direction in ('long','short') and trD in ('up','down'):
        if (direction == 'long' and trD == 'up') or (direction == 'short' and trD == 'down'):
            confidence = min(1.0, confidence + rules.confluence_bonus_ctx)
        elif (direction == 'long' and trD == 'down') or (direction == 'short' and trD == 'up'):
            confidence = max(0.0, confidence - rules.confluence_penalty_ctx)

    # 7) AVOID conditions (giữ nguyên triết lý cũ) + liquidity guard
    rsi_now = float(f1.get('momentum', {}).get('rsi', 50.0))
    div = (f1.get('momentum', {}).get('divergence') or 'none')
    vz = float(f1.get('volume', {}).get('vol_z20', 0.0))

    rsi_extreme = (rsi_now >= rules.rsi_overbought) or (rsi_now <= rules.rsi_oversold)
    div_against = (direction == 'long' and div == 'bearish') or (direction == 'short' and div == 'bullish')
    vol_blowoff = (vz >= rules.vol_z_hot)
    # robust: đọc cờ gần HVN theo evidence
    liq_heavy_close = (not liq_ok) and bool(getattr(eb.evidence.liquidity, 'near_heavy_zone', False))

    # AVOID theo ngữ cảnh setup & hướng
    trendish_states = {'breakout','breakdown','trend_follow_up','trend_follow_down',
                       'volatility_breakout','throwback_long','throwback_short','reclaim'}
    is_trendish = (state in trendish_states)
    try:
        ret_last = float(df1['close'].iloc[-1] - df1['close'].iloc[-2])
    except Exception:
        ret_last = 0.0
    blowoff_up = vol_blowoff and (ret_last > 0)
    blowoff_down = vol_blowoff and (ret_last < 0)
    # chỉ AVOID blowoff khi nó "cùng chiều kèo" (đu sóng) và chỉ với setup bám trend
    vol_blowoff_avoid = (is_trendish and ((direction == 'long' and blowoff_up) or
                                          (direction == 'short' and blowoff_down)))
    # RSI cực trị: AVOID khi là kèo bám trend; với kèo đảo chiều thì không
    rsi_extreme_avoid = (is_trendish and rsi_extreme)

    avoid_reasons = []
    if rsi_extreme_avoid: avoid_reasons.append('rsi_extreme')
    if vol_blowoff_avoid: avoid_reasons.append('volume_blowoff')
    if liq_heavy_close: avoid_reasons.append('heavy_liquidity_ahead')
    # Divergence ngược chiều: giảm confluence thay vì AVOID cứng
    if div_against:
        confluence_score = max(0.0, confluence_score - 0.15)
        miss_reasons.append('divergence_against')

    # 8) Quyết định ENTER/WAIT/AVOID (bỏ confluence gate; dùng micro-gates & adaptive)
    decision = 'WAIT'
    # regime-adaptive thresholds
    rr_min_eff = rules.rr_min * (0.87 if regime=='low' else (1.13 if regime=='high' else 1.0))
    liq_guard_fail = not liq_ok
    if avoid_reasons:
        decision = 'AVOID'
    else:
        any_prox = proximity_ok or proximity_ok2
        rr_ok_any = (rr is not None and rr >= rr_min_eff) or (rr_entry2 is not None and rr_entry2 >= rr_min_eff)
    
        # Micro-gates per coarse state
        micro_wait = False
        if state == 'TREND_BREAK':
            # in slow/liquidity-floor sessions, yêu cầu vol ok hoăc bb expanding
            need_boost = is_slow or liquidity_floor
            have_boost = bool(vol_ok) or bool(getattr(eb.evidence, 'bb', None) and getattr(eb.evidence.bb, 'ok', False))
            if need_boost and not have_boost:
                micro_wait = True
        elif state == 'TREND_RETEST':
            # cần align 1H~4H cùng hướng để vào
            tr1 = (f1.get('trend', {}) or {}).get('state')
            tr4 = (f4.get('trend', {}) or {}).get('state')
            if not (tr1 == tr4 and tr1 in ('up','down')):
                micro_wait = True
        elif state == 'REVERSAL':
            # tránh khi BB đang bung mạnh
            bb_ok = bool(getattr(eb.evidence, 'bb', None) and getattr(eb.evidence.bb, 'ok', False))
            if bb_ok:
                micro_wait = True
        elif state == 'RANGE':
            # Range mode: không vào lệnh. Chỉ giao dịch khi có REVERSAL hoặc TREND_BREAK xuất hiện.
            micro_wait = True
            if 'range_mode' not in miss_reasons:
                miss_reasons.append('range_mode')
        gates_ok = all(req_ok) and any_prox and rr_ok_any and (not liq_guard_fail) and (not micro_wait)
        decision = 'ENTER' if gates_ok else 'WAIT'
    
    # 9) Build plan out (giữ style PlanOut/LogsOut cũ, thêm confluence vào note)
    note_str = (note or "")
    note_str += f"|confluence={confluence_score:.2f}"

    plan = PlanOut(
        direction=direction,
        entry=_smart_round(entry) if isinstance(entry, (int,float)) else None,
        sl=_smart_round(sl) if isinstance(sl, (int,float)) else None,
        tp=_smart_round(tp) if isinstance(tp, (int,float)) else None,     # legacy = TP2
        rr=round(rr, 3) if isinstance(rr, (int,float)) else None,         # legacy RR at TP (TP2)
        tp1=_smart_round(tp1) if isinstance(tp1, (int,float)) else None,
        tp2=_smart_round(tp2) if isinstance(tp2, (int,float)) else None,
        tp3=_smart_round(tp3) if isinstance(tp3, (int,float)) else None,
        rr1=round(rr1, 3) if 'rr1' in locals() and isinstance(rr1, (int,float)) else None,
        rr2=round(rr2, 3) if isinstance(rr2, (int,float)) else None,
        rr3=round(rr3, 3) if 'rr3' in locals() and isinstance(rr3, (int,float)) else None,
        entry2=_smart_round(entry2) if 'entry2' in locals() and isinstance(entry2, (int,float)) else None,
        note=note_str or None,
    )

    logs = LogsOut(
        ENTER={
            'required_ok': all(req_ok),
            'proximity_ok_primary': proximity_ok,
            'proximity_ok_secondary': proximity_ok2,
            'rr_ok_primary': (rr is not None and rr >= rr_min_eff),
            'rr_ok_secondary': (rr_entry2 is not None and rr_entry2 >= rr_min_eff),
            'confluence_score': confluence_score,
            'state_group': state,
            'orig_state': orig_state,
            'tags': coarse_tags,
            'vote_sum': vote_sum,
            'vote_breakdown': vote_breakdown,
            'plan': plan.model_dump() if hasattr(plan, 'model_dump') else plan.__dict__,
        },
        WAIT={
            'missing': sorted(set(miss_reasons)),
            'vol_ok': vol_ok,
            'momentum_ok': mom_ok,
            'candles_ok': cdl_ok,
            'liquidity_ok': liq_ok,
            'confluence_score': confluence_score,
            'state_group': state,
            'orig_state': orig_state,
            'tags': coarse_tags,
            'vote_sum': vote_sum,
            'plan_preview': {
                'direction': plan.direction,
                'entry': plan.entry, 'sl': plan.sl,
                'tp1': plan.tp1, 'tp2': plan.tp2, 'tp3': plan.tp3,
                'rr1': plan.rr1, 'rr2': plan.rr2, 'rr3': plan.rr3
            },
        },
        AVOID={
            'reasons': avoid_reasons,
            'rsi': rsi_now,
            'divergence': div,
            'vol_z20': vz,
            'rr': rr,
            'rr_entry2': rr_entry2,
        },
    )

    out = DecisionOut(
        symbol=symbol,
        timeframe=timeframe,
        asof=eb.asof,
        state=state,
        confidence=float(confidence),
        decision=decision,
        plan=plan,
        logs=logs,
        telegram_signal=None,
        headline=None,
    )

    # Telegram signal when ENTER (include both entries if applicable)
    telegram_signal = None
    if decision == 'ENTER' and direction and plan.sl is not None and (plan.entry is not None or plan.entry2 is not None):
        # Đưa đủ TP1/TP2/TP3 (fallback về tp legacy nếu thiếu)
        strategy = state.replace('_', ' ').title()
        entry_lines = []
        if plan.entry is not None:
            entry_lines.append(f"Entry: {plan.entry}")
        if plan.entry2 is not None:
            entry_lines.append(f"Entry2: {plan.entry2}")
        tp_lines = []
        if plan.tp1 is not None: tp_lines.append(f"TP1: {plan.tp1}")
        if plan.tp2 is not None: tp_lines.append(f"TP2: {plan.tp2}")
        if plan.tp3 is not None: tp_lines.append(f"TP3: {plan.tp3}")
        if not tp_lines and plan.tp is not None:
            tp_lines.append(f"TP: {plan.tp}")
        entries_text = "\n".join(entry_lines) if entry_lines else ""
        tps_text = "\n".join(tp_lines) if tp_lines else ""
        telegram_signal = (
            f"{direction.upper()} | {symbol}\n"
            f"Strategy: {strategy} ({timeframe})\n"
            f"{entries_text}\n"
            f"SL: {plan.sl}\n"
            f"{tps_text}"
        )
    
    
    # Compose a short headline for logging (includes direction and TP ladder)
    _tp_parts = []
    if plan.tp1 is not None: _tp_parts.append(f"TP1={plan.tp1}")
    if plan.tp2 is not None: _tp_parts.append(f"TP2={plan.tp2}")
    if plan.tp3 is not None: _tp_parts.append(f"TP3={plan.tp3}")
    _tp_text = " ".join(_tp_parts) if _tp_parts else (f"TP={plan.tp}" if plan.tp is not None else "")
    headline = (
        f"DECISION={decision} | STATE={state} | DIR={(plan.direction or '-').upper()} | "
        f"entry={plan.entry} entry2={plan.entry2} sl={plan.sl} "
        f"{_tp_text} rr={(plan.rr if plan.rr is not None else plan.rr2)}"
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
