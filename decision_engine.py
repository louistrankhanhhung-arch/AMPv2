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
    retest_zone_atr_reclaim: float = 0.50  
    trend_break_buf_atr: float = 0.10 # trend-follow Entry1 uses break of nearest swing with this buffer (sửa 0.2 -> 0.1)
    sl_min_atr: float = 0.5       # SL tối thiểu = 0.5*ATR
    tp_ladder_n: int = 3           # số bậc TP
    # SL pads by setup type (A/B testable)
    sl_pad_breakout_atr: float = 0.5
    sl_pad_reclaim_atr: float = 0.8
    sl_pad_trend_follow_atr: float = 0.6
    sl_pad_mean_reversion_atr: float = 1.2
    # --- NEW: multi-TF confluence gates ---
    rsi1h_long: float = 55.0
    rsi4h_long_soft: float = 50.0
    rsi1h_long_ctr: float = 60.0    # yêu cầu khi countertrend 4H
    rsi1h_short: float = 45.0
    rsi4h_short_soft: float = 50.0
    rsi1h_short_ctr: float = 40.0   # yêu cầu khi countertrend 4H
    confluence_enter_thr: float = 0.5
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
    req_keys = [k for k in req_keys if getattr(eb.evidence, k, None) is not None]

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
    else:
        for k in req_keys:
            ev = _ev(k)
            if ev is None:
                miss_reasons.append(k)
                continue
            if not getattr(ev, 'ok', False):
                miss_reasons.append(k)

    # Optional/guards (không phải required)
    vol_ok = bool(getattr(eb.evidence.volume, 'ok', False))
    tr_ok = bool(getattr(eb.evidence.trend, 'ok', False))
    mom_ok = bool(getattr(getattr(eb.evidence, 'momentum', None), 'primary', None).ok) if getattr(eb.evidence, 'momentum', None) else False
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
        pad = rules.retest_pad_atr * atr
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

    # 2) Layered TP: dùng 1H nhưng lọc theo band 4H (nếu có). Pivot = ENTRY (không dùng price_now)
    tp1 = tp2 = tp3 = None
    if direction in ('long','short') and atr > 0 and entry is not None:
        piv = float(entry)
        tps = _tp_ladder_confluence(levels, levels4h, piv, direction, atr, rules.tp_ladder_n)
        if tps:
            if len(tps) >= 1: tp1 = float(tps[0])
            if len(tps) >= 2: tp2 = float(tps[1])
            if len(tps) >= 3: tp3 = float(tps[2])
        # legacy tp ưu tiên TP2, fallback TP1 rồi TP3
        if tp is None:
            if tp2 is not None: tp = float(tp2)
            elif tp1 is not None: tp = float(tp1)
            elif tp3 is not None: tp = float(tp3)

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

        prox_thr = (rules.retest_zone_atr_reclaim if state == 'reclaim' else rules.retest_zone_atr)
        proximity_ok = (abs(price_now - e1f) <= prox_thr * atr)

    # 4) trend-follow secondary entry (EMA20/BB mid) nếu có
    rr_entry2 = None
    proximity_ok2 = False
    e2 = locals().get('entry2', None)
    if direction and isinstance(e2, (int, float)) and sl is not None and tp is not None and atr > 0:
        e2f = float(e2)
        rr_entry2 = _rr(direction, e2f, sl, float(tp))
        if rr_entry2 is not None and rr_entry2 > rules.rr_max:
            rr_entry2 = float(rules.rr_max)
        proximity_ok2 = (abs(price_now - e2f) <= rules.proximity_atr * atr)

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
    rr_bad = (rr is not None and rr < rules.rr_avoid)
    liq_heavy_close = (not liq_ok) and (getattr(eb.evidence.liquidity, 'why', '') or '').startswith('heavy_zone')

    avoid_reasons = []
    if rsi_extreme: avoid_reasons.append('rsi_extreme')
    if div_against: avoid_reasons.append('rsi_divergence_against')
    if vol_blowoff: avoid_reasons.append('volume_blowoff')
    if rr_bad: avoid_reasons.append('rr_bad')
    if liq_heavy_close: avoid_reasons.append('heavy_liquidity_ahead')

    # 8) Quyết định ENTER/WAIT/AVOID (thêm gate confluence + liquidity_ok)
    decision = 'WAIT'
    if avoid_reasons:
        decision = 'AVOID'
    else:
        any_prox = proximity_ok or proximity_ok2
        rr_ok_any = (rr is not None and rr >= rules.rr_min) or (rr_entry2 is not None and rr_entry2 >= rules.rr_min)
        confluence_ok = (confluence_score >= rules.confluence_enter_thr)
        liq_guard_fail = not liq_ok

        gates_ok = all(req_ok) and any_prox and rr_ok_any and confluence_ok and (not liq_guard_fail)

        # riêng RECLAIM: nếu 4H đối hướng bắt buộc momentum.ok
        if state == 'reclaim' and ((direction == 'long' and tr4 == 'down') or (direction == 'short' and tr4 == 'up')):
            gates_ok = gates_ok and mom_ok

        if not all(req_ok):
            pass
        if not any_prox:
            miss_reasons.append('price_far_from_entry')
        if not rr_ok_any:
            miss_reasons.append('rr_min')
        if not confluence_ok:
            miss_reasons.append('confluence_low')
        if liq_guard_fail:
            miss_reasons.append('liquidity_guard')

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
            'rr_ok_primary': (rr is not None and rr >= rules.rr_min),
            'rr_ok_secondary': (rr_entry2 is not None and rr_entry2 >= rules.rr_min),
            'confluence_score': confluence_score,
            'plan': plan.model_dump() if hasattr(plan, 'model_dump') else plan.__dict__,
        },
        WAIT={
            'missing': sorted(set(miss_reasons)),
            'vol_ok': vol_ok,
            'momentum_ok': mom_ok,
            'candles_ok': cdl_ok,
            'liquidity_ok': liq_ok,
            'confluence_score': confluence_score,
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
    return out.model_dump()

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



