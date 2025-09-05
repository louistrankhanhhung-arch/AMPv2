
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List
import math


# ===============================================
# Tiny-Core (Side-Aware State)  — dict-safe evidence
# ===============================================
@dataclass
class SideCfg:
    # regime thresholds (you can tune for your market)
    bbw_squeeze_thr: float = 0.06         # range-like when below
    adx_trend_thr: float = 18.0           # range-like when below

    # tie handling
    tie_eps: float = 1e-6                 # absolute tie tolerance
    side_margin: float = 0.05             # require this margin to choose a side
    allow_slope_fallback: bool = True  # mặc định KHÔNG fallback, chỉ dùng slope trực tiếp
    
    # volatility/momentum thresholds
    natr_lo: float = 0.015                # low-vol regime (<= 1.5%)
    natr_hi: float = 0.04                 # high-vol regime (>= 4%)
    rsi_long_thr: float = 55.0
    rsi_short_thr: float = 45.0

    # setup/entry rules
    max_entry_dist_atr: float = 0.6
    rr_min_enter: float = 1.2
    sl_min_atr: float = 0.6

    # direction scoring within retest
    retest_long_threshold: float = 0.6
    retest_short_threshold: float = 0.6

    # timeframes
    tf_primary: str = "1H"
    tf_confirm: str = "4H"
@staticmethod
def _get_env_float(name: str, default: float) -> float:
    import os
    try:
        v = os.getenv(name)
        return float(v) if v is not None and v != "" else default
    except Exception:
        return default
@staticmethod
def _get_env_bool(name: str, default: bool) -> bool:
    import os
    v = os.getenv(name)
    if v is None or v == "":
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")

@classmethod
def from_env(cls) -> "SideCfg":
    """
    Load SideCfg from environment variables (optional overrides).
    Variables:
      SIDE_MARGIN, RETEST_LONG_THRESHOLD, RETEST_SHORT_THRESHOLD,
      MAX_ENTRY_DIST_ATR, RR_MIN_ENTER, ALLOW_SLOPE_FALLBACK
    """
    return cls(
        side_margin=cls._get_env_float("SIDE_MARGIN", cls.side_margin),
        retest_long_threshold=cls._get_env_float("RETEST_LONG_THRESHOLD", cls.retest_long_threshold),
        retest_short_threshold=cls._get_env_float("RETEST_SHORT_THRESHOLD", cls.retest_short_threshold),
        max_entry_dist_atr=cls._get_env_float("MAX_ENTRY_DIST_ATR", cls.max_entry_dist_atr),
        rr_min_enter=cls._get_env_float("RR_MIN_ENTER", cls.rr_min_enter),
        allow_slope_fallback=cls._get_env_bool("ALLOW_SLOPE_FALLBACK", cls.allow_slope_fallback),
    )

@dataclass
class SI:  # Side Indicators (pulled from features/evidence)
    price: Optional[float] = None
    atr: Optional[float] = None
    natr: Optional[float] = None  # ATR / price

    ema_slope_primary: Optional[float] = None
    ema_slope_confirm: Optional[float] = None
    rsi_primary: Optional[float] = None
    rsi_confirm: Optional[float] = None
    bbw_primary: Optional[float] = None
    adx_primary: Optional[float] = None

    vol_impulse_up: bool = False
    vol_impulse_down: bool = False

    # structure events
    breakout_ok: bool = False
    breakout_side: Optional[str] = None  # "long"/"short"
    last_break_level: Optional[float] = None

    retest_ok: bool = False
    retest_zone_mid: Optional[float] = None
    reclaim_ok: bool = False
    reclaim_side: Optional[str] = None

    # mean-reversion (optional)
    meanrev_ok: bool = False
    meanrev_side: Optional[str] = None

    # new: reversal/fake & boosters
    false_break_long: bool = False   # false_breakdown => long bias
    false_break_short: bool = False  # false_breakout  => short bias
    rejection_side: Optional[str] = None  # 'long'/'short'
    div_side: Optional[str] = None        # 'long'/'short'
    tf_ready_long: bool = False
    tf_ready_short: bool = False


def _get(path: List[str], d: Dict[str, Any], default=None):
    cur = d
    for p in path:
        cur = (cur or {}).get(p, {} if p != path[-1] else None)
    return default if cur is None else cur

def _gat(obj, name, default=None):
    """get attribute or dict key"""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)

def _as_evidence_dict(eb):
    """Return an object (dict-like or attr) that holds evidence items directly.
    Accept eb, or {"evidence": {...}}, or pydantic models with attributes.
    """
    if eb is None:
        return {}
    # unwrap .evidence if present
    ev = _gat(eb, "evidence", None)
    if ev is not None:
        return ev
    return eb

def collect_side_indicators(features_by_tf: Dict[str, Dict[str, Any]], eb: Any, cfg: SideCfg) -> SI:
    P = features_by_tf.get(cfg.tf_primary, {}) or {}
    C = features_by_tf.get(cfg.tf_confirm, {}) or {}

    price = P.get("close") or P.get("price") or _get(["price", "close"], P, None)
    atr = P.get("atr") or _get(["volatility", "atr"], P, None)
    natr = None
    if atr and price:
        natr = atr / price

    # Unwrap evidence container
    EV = _as_evidence_dict(eb)

    # helpers
    def _ev_item(key):
        return _gat(EV, key, None)

    def _ok(key):
        itm = _ev_item(key)
        return bool(_gat(itm, "ok", False))

    def _level(key):
        itm = _ev_item(key)
        return _gat(itm, "level", None)

    def _mid(key):
        itm = _ev_item(key)
        return _gat(itm, "mid", None)

    def _side(key):
        itm = _ev_item(key)
        return _gat(itm, "side", None)

    def _reclaim_side():
        itm = _ev_item("price_reclaim")
        ref = _gat(itm, "ref", None)
        return _gat(ref, "side", None)

    breakout_ok = bool(_ok("price_breakout") or _ok("price_breakdown"))
    breakout_side = "long" if _ok("price_breakout") else ("short" if _ok("price_breakdown") else None)
    last_break_level = _level("price_breakout") or _level("price_breakdown")

    retest_ok = bool(_ok("pullback_throwback") or _ok("pullback"))
    retest_zone_mid = _mid("pullback_throwback") or _mid("pullback")
    reclaim_ok = bool(_ok("price_reclaim"))
    reclaim_side = _reclaim_side()

    meanrev_ok = bool(_ok("mean_reversion"))
    meanrev_side = _side("mean_reversion")

    # ---- new evidences ----
    fb_out_ok = _ok("false_breakout")     # poke trên HH fail → nghiêng short
    fb_dn_ok  = _ok("false_breakdown")    # poke dưới LL fail → nghiêng long
    rjt_side  = _side("rejection")        # ev_rejection.side ∈ {'long','short'}
    div_side  = _side("divergence")       # ev_divergence_updown.side
    # guard: tuyệt đối không shadow tên hàm _ev_item
    _tfr = _ev_item("trend_follow_ready") or {}
    tf_long  = bool(_gat(_tfr, "long", {}).get("ok", False))
    tf_short = bool(_gat(_tfr, "short", {}).get("ok", False))

    vol_impulse_up = bool(_ok("volume_impulse_up"))
    vol_impulse_down = bool(_ok("volume_impulse_down"))

    # lấy trực tiếp ema50_slope
    ema_slope_primary = P.get("ema50_slope") or _get(["trend","ema50_slope"], P, None)
    ema_slope_confirm = C.get("ema50_slope") or _get(["trend","ema50_slope"], C, None)
    # optional fallback nếu bật cờ
    if ema_slope_primary is None and cfg.allow_slope_fallback:
        st = _get(["trend","state"], P, None)
        ema_slope_primary = 1.0 if st == "up" else (-1.0 if st == "down" else 0.0)
    if ema_slope_confirm is None and cfg.allow_slope_fallback:
        st = _get(["trend","state"], C, None)
        ema_slope_confirm = 1.0 if st == "up" else (-1.0 if st == "down" else 0.0)   
        
    si = SI(
        price = price,
        atr = atr,
        natr = natr,
        ema_slope_primary = ema_slope_primary,
        ema_slope_confirm = ema_slope_confirm,
        rsi_primary = P.get("rsi") or _get(["momentum","rsi"], P, None),
        rsi_confirm = C.get("rsi") or _get(["momentum","rsi"], C, None),
        bbw_primary = (
            P.get("bb_width")
            or _get(["volatility","bb_width"], P, None)
            or P.get("bbw_last")
            or _get(["volatility","bbw_last"], P, None)
        ),
        adx_primary = P.get("adx") or _get(["trend","adx"], P, None),

        vol_impulse_up   = vol_impulse_up,
        vol_impulse_down = vol_impulse_down,

        breakout_ok      = breakout_ok,
        breakout_side    = breakout_side,
        last_break_level = last_break_level,

        retest_ok        = retest_ok,
        retest_zone_mid  = retest_zone_mid,
        reclaim_ok       = reclaim_ok,
        reclaim_side     = reclaim_side,

        meanrev_ok       = meanrev_ok,
        meanrev_side     = meanrev_side,

        false_break_long = bool(fb_dn_ok),
        false_break_short= bool(fb_out_ok),
        rejection_side   = rjt_side if rjt_side in ("long","short") else None,
        div_side         = div_side if div_side in ("long","short") else None,
        tf_ready_long    = tf_long,
        tf_ready_short   = tf_short,
    )
    return si


# ============== State with Side =================
def _range_like(si: SI, cfg: SideCfg) -> bool:
    bbw_ok = (si.bbw_primary is not None) and (si.bbw_primary <= cfg.bbw_squeeze_thr)
    adx_ok = (si.adx_primary is None) or (si.adx_primary < cfg.adx_trend_thr)  # allow missing ADX
    return bbw_ok and adx_ok

def _break_distance_atr(si: SI) -> float:
    if not si.price or not si.atr or not si.last_break_level or si.atr == 0:
        return 0.0
    return abs(si.price - si.last_break_level) / si.atr

def _trend_sign(si: SI) -> int:
    if si.ema_slope_primary is None or si.ema_slope_confirm is None:
        return 0
    if si.ema_slope_primary > 0 and si.ema_slope_confirm > 0:
        return 1
    if si.ema_slope_primary < 0 and si.ema_slope_confirm < 0:
        return -1
    return 0

def _momo_sign(si: SI, cfg: SideCfg) -> int:
    if si.rsi_primary is None or si.rsi_confirm is None:
        return 0
    if si.rsi_primary >= cfg.rsi_long_thr and si.rsi_confirm >= 50:
        return 1
    if si.rsi_primary <= cfg.rsi_short_thr and si.rsi_confirm <= 50:
        return -1
    return 0

def _volume_dir(si: SI) -> int:
    if si.vol_impulse_up and not si.vol_impulse_down:
        return 1
    if si.vol_impulse_down and not si.vol_impulse_up:
        return -1
    return 0

def classify_state_with_side(si: SI, cfg: SideCfg) -> Tuple[str, Optional[str], Dict[str, Any]]:
    """
    Returns (state_name, side, meta)
    States: breakout, breakdown, retest_support, retest_resistance, none_state
    Volume & NATR regimes adjust breakout acceptance.
    Range & reverse are folded into retest_* when possible.
    """
    meta: Dict[str, Any] = {}
    natr = si.natr if si.natr is not None else 0.0
    dist_atr = _break_distance_atr(si)
    tr = _trend_sign(si)
    momo = _momo_sign(si, cfg)
    vdir = _volume_dir(si)
    meta.update(dict(natr=natr, dist_atr=dist_atr, trend=tr, momo=momo, v=vdir))

    # --- 1) Breakout/breakdown regime-aware ---
    if si.breakout_ok and si.breakout_side in ("long","short"):
        # NATR regime: in high vol, require matching volume impulse to confirm; in low vol, relax
        if natr >= cfg.natr_hi:
            vol_ok = (si.breakout_side == "long" and si.vol_impulse_up) or (si.breakout_side == "short" and si.vol_impulse_down)
        elif natr <= cfg.natr_lo:
            vol_ok = True
        else:
            # mid regime: prefer matching volume, but not mandatory
            vol_ok = True if vdir == 0 else ((si.breakout_side == "long" and vdir >= 0) or (si.breakout_side == "short" and vdir <= 0))

        # distance beyond level (avoid micro breaks in choppy)
        dist_ok = (dist_atr >= cfg.break_buffer_atr) if cfg.break_buffer_atr > 0 else True

        # --- new: boosters & blockers ---
        if si.breakout_side == "long" and si.tf_ready_long:
            vol_ok = True  # trend-follow booster for breakout long
        if si.breakout_side == "short" and si.tf_ready_short:
            vol_ok = True  # booster for breakdown short

        # block if recent false_* chống hướng
        if (si.breakout_side == "long" and si.false_break_short) or (si.breakout_side == "short" and si.false_break_long):
            vol_ok = False  # force re-evaluate as retest

            if si.breakout_side == "long":
                return "breakout", "long", meta
            else:
                return "breakdown", "short", meta

    # --- 2) Retest (includes reverse & range) ---
    is_range_like = _range_like(si, cfg)

    # If no explicit retest signal, in range we still try classify to support/resistance by location and soft context
    retest_signal = si.retest_ok or is_range_like or si.meanrev_ok or si.false_break_long or si.false_break_short or (si.rejection_side in ("long","short"))

    if retest_signal:
        # Soft scoring to decide support(long) vs resistance(short)
        long_score = 0.0
        short_score = 0.0

        # context signals
        long_score += 1.0 if tr > 0 else 0.0
        short_score += 1.0 if tr < 0 else 0.0

        long_score += 0.5 if momo > 0 else 0.0
        short_score += 0.5 if momo < 0 else 0.0

        long_score += 0.75 if si.reclaim_ok and si.reclaim_side == "long" else 0.0
        short_score += 0.75 if si.reclaim_ok and si.reclaim_side == "short" else 0.0

        # location vs zone
        if si.retest_zone_mid is not None and si.price is not None:
            if si.price <= si.retest_zone_mid:
                long_score += 0.5  # near support/mean
            if si.price >= si.retest_zone_mid:
                short_score += 0.5 # near resistance/mean

        # volume tilt (bonus if aligned)
        if vdir > 0:
            long_score += 0.25
        elif vdir < 0:
            short_score += 0.25

        # reverse hint (mean-rev side)
        if si.meanrev_ok and si.meanrev_side in ("long","short"):
            if si.meanrev_side == "long":
                long_score += 0.5
            else:
                short_score += 0.5

        # new: false_* nghiêng mạnh vào retest ngược hướng break
        if si.false_break_long:
            long_score += 0.75
        if si.false_break_short:
            short_score += 0.75

        # new: rejection & divergence nghiêng nhẹ
        if si.rejection_side == "long":
            long_score += 0.5
        elif si.rejection_side == "short":
            short_score += 0.5
        if si.div_side == "long":
            long_score += 0.25
        elif si.div_side == "short":
            short_score += 0.25
        meta.update(dict(long_score=long_score, short_score=short_score))

        # Tie & margin policy:
        # - If both sides close (|diff| <= tie_eps or |diff| < side_margin), return none_state -> WAIT.
        # - Else require the winner to pass its threshold.
        diff = long_score - short_score
        if abs(diff) <= cfg.tie_eps or abs(diff) < cfg.side_margin:
            return "none_state", None, meta

        if diff > 0 and long_score >= cfg.retest_long_threshold:
            return "retest_support", "long", meta
        if diff < 0 and short_score >= cfg.retest_short_threshold:
            return "retest_resistance", "short", meta

    # --- 3) None ---
    return "none_state", None, meta


# ============== Setup Builder ===================
@dataclass
class Setup:
    state: str
    side: Optional[str]
    entry: Optional[float]
    sl: Optional[float]
    tps: List[float]
    proximity_ok: bool
    rr_any_ok: bool
    meta: Dict[str, Any]

def _tp_ladder(entry: float, sl: float, side: str, rr_targets: tuple) -> List[float]:
    tps = []
    for rr in rr_targets:
        if side == "long":
            tps.append(entry + rr * (entry - sl))
        else:
            tps.append(entry - rr * (sl - entry))
    return tps

def _rr(price: float, entry: float, sl: float, tp: float, side: str) -> float:
    risk = (entry - sl) if side == "long" else (sl - entry)
    reward = (tp - entry) if side == "long" else (entry - tp)
    if risk <= 0:
        return 0.0
    return reward / risk

def build_setup(si: SI, state: str, side: Optional[str], cfg: SideCfg) -> Setup:
    price = si.price
    atr = si.atr or 0.0
    if price is None or atr == 0 or side is None:
        return Setup(state, side, None, None, [], False, False, dict())

    entry = None
    sl = None
    meta: Dict[str, Any] = {}

    if state in ("breakout","breakdown"):
        base = si.last_break_level if si.last_break_level is not None else price
        if side == "long":  # breakout
            entry = max(price, base)
            sl = entry - max(cfg.sl_min_atr*atr, 0.8*atr)
        else:              # breakdown
            entry = min(price, base)
            sl = entry + max(cfg.sl_min_atr*atr, 0.8*atr)

    elif state in ("retest_support","retest_resistance"):
        base = si.retest_zone_mid if si.retest_zone_mid is not None else price
        entry = base
        if side == "long":
            sl = entry - max(cfg.sl_min_atr*atr, 0.8*atr)
        else:
            sl = entry + max(cfg.sl_min_atr*atr, 0.8*atr)

    else:  # none_state
        return Setup(state, side, None, None, [], False, False, dict())

    tps = _tp_ladder(entry, sl, side, cfg.rr_targets)
    # proximity & RR gates data
    dist_atr = abs(price - entry)/atr if atr > 0 else math.inf
    proximity_ok = (dist_atr <= cfg.max_entry_dist_atr)
    rr_any_ok = any(_rr(price, entry, sl, tp, side) >= cfg.rr_min_enter for tp in tps)

    meta.update(dict(dist_atr=dist_atr))
    return Setup(state, side, entry, sl, tps, proximity_ok, rr_any_ok, meta)


# ============== 5 Gates Decision =================
@dataclass
class Decision:
    decision: str                  # ENTER / WAIT / AVOID
    state: str
    side: Optional[str]
    setup: Setup
    reasons: List[str]
    meta: Dict[str, Any]

def decide_5_gates(state: str, side: Optional[str], setup: Setup, si: SI, cfg: SideCfg, meta: Dict[str, Any]) -> Decision:
    reasons: List[str] = []
    # Gate 1: state != none_state
    if state == "none_state":
        reasons.append("none_state")
    # Gate 2: must have side
    if side is None:
        reasons.append("no_side")
    # Gate 3: proximity
    if not setup.proximity_ok:
        reasons.append("far_from_entry")
    # Gate 4: RR
    if not setup.rr_any_ok:
        reasons.append("rr_too_low")
    # Gate 5: liquidity/volume guard (simple contra check)
    if side == "long" and si.vol_impulse_down:
        reasons.append("volume_contra")
    if side == "short" and si.vol_impulse_up:
        reasons.append("volume_contra")

    decision = "ENTER" if len(reasons) == 0 else "WAIT"
    return Decision(decision, state, side, setup, reasons, meta)


# ============== Orchestrator =====================
def run_side_state_core(features_by_tf: Dict[str, Dict[str, Any]], eb: Any, cfg: Optional[SideCfg] = None) -> Decision:
    cfg = cfg or SideCfg()
    si = collect_side_indicators(features_by_tf, eb, cfg)
    state, side, meta = classify_state_with_side(si, cfg)
    setup = build_setup(si, state, side, cfg)
    dec = decide_5_gates(state, side, setup, si, cfg, meta)
    return dec
