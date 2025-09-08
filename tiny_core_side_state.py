from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, List
import math

# ===============================================
# Tiny-Core (Side-Aware State) — dict-safe evidence
# ===============================================

@dataclass
class SideCfg:
    """Config cho phân loại state có xét side (long/short)."""

    # Regime thresholds (tinh chỉnh theo thị trường bạn chạy)
    bbw_squeeze_thr: float = 0.06       # range-like khi BBW dưới ngưỡng
    adx_trend_thr: float = 25.0         # range-like khi ADX dưới ngưỡng
    break_buffer_atr: float = 0.3      # buffer tính theo ATR quanh mốc break

    # Tie handling
    tie_eps: float = 1e-6               # sai số tuyệt đối để coi như hoà
    side_margin: float = 1.0           # yêu cầu chênh tối thiểu để chọn side

    # Retest score gates
    retest_long_threshold: float = 1.25
    retest_short_threshold: float = 1.25

    # TP ladder mặc định cho tính RR (fallback khi thiếu band)
    rr_targets: Tuple[float, float, float] = (1.2, 2.0, 3.0)

    # Timeframes
    tf_primary: str = "1H"
    tf_confirm: str = "4H"

# Kết quả setup/decision để tương thích engine_adapter.py
@dataclass
class Setup:
    entry: Optional[float] = None
    sl: Optional[float] = None
    tps: List[float] = field(default_factory=list)

@dataclass
class Decision:
    decision: str = "WAIT"                 # ENTER / WAIT / AVOID
    state: Optional[str] = None
    side: Optional[str] = None
    setup: Setup = field(default_factory=Setup)
    meta: Dict[str, Any] = field(default_factory=dict)
    reasons: List[str] = field(default_factory=list)

# Stub/alias kiểu dữ liệu
SI = Any


# -----------------------------
# Helpers (an toàn, tái sử dụng)
# -----------------------------
def _safe_get(obj: Any, name: str, default: Any = None) -> Any:
    try:
        return getattr(obj, name, default)
    except Exception:
        try:
            return obj.get(name, default)  # dict-style
        except Exception:
            return default

def _get_ev(eb: Dict[str, Any], key: str) -> Dict[str, Any]:
    if eb is None:
        return {}
    # eb có thể là {'evidence': {...}} hoặc đã là dict evidence thẳng
    evs = eb.get("evidence", eb) if isinstance(eb, dict) else {}
    x = evs.get(key, {}) if isinstance(evs, dict) else {}
    return x or {}

def _price(df) -> Optional[float]:
    try:
        return float(df["close"].iloc[-1])
    except Exception:
        return None

def _rr(direction: Optional[str], entry: Optional[float], sl: Optional[float], tp: Optional[float]) -> float:
    try:
        if direction == 'long':
            risk = max(1e-9, (entry or 0) - (sl or 0))
            reward = max(0.0, (tp or 0) - (entry or 0))
        elif direction == 'short':
            risk = max(1e-9, (sl or 0) - (entry or 0))
            reward = max(0.0, (entry or 0) - (tp or 0))
        else:
            return 0.0
        return float(reward / risk) if risk > 0 else 0.0
    except Exception:
        return 0.0

def _ensure_sl_gap(entry: float, sl: float, atr: float, side: str, min_atr: float = 0.6) -> float:
    """Đảm bảo khoảng cách SL tối thiểu theo ATR; giữ đúng phía. (ý tưởng từ decision_engine) :contentReference[oaicite:1]{index=1}"""
    min_gap = max(1e-9, min_atr * max(atr, 0.0))
    if atr <= 0 or entry is None or sl is None or side not in ("long", "short"):
        return sl
    if side == 'long':
        return float(entry - min_gap) if (entry - sl) < min_gap else float(sl)
    else:
        return float(entry + min_gap) if (sl - entry) < min_gap else float(sl)

def _tp_by_rr(entry: float, sl: float, side: str, targets: Tuple[float, ...]) -> List[float]:
    """
    Tạo TP theo RR tuyệt đối từ cấu hình rr_targets (ví dụ 1.2, 2.0, 3.0).
    RR = |TP - entry| / |entry - SL|  (tùy theo side).
    """
    try:
        if side not in ("long", "short"):
            return []
        risk = (entry - sl) if side == "long" else (sl - entry)
        if risk <= 0:
            return []
        tps: List[float] = []
        for r in targets:
            r = float(r)
            tp = entry + r * risk if side == "long" else entry - r * risk
            tps.append(float(tp))
        # đảm bảo thứ tự hợp lý theo side
        tps.sort(reverse=(side == "short"))
        return tps
    except Exception:
        return []


# -------------------------------------------------------
# Collect side indicators từ features + evidence bundle
# -------------------------------------------------------
def collect_side_indicators(features_by_tf: Dict[str, Dict[str, Any]], eb: Dict[str, Any], cfg: SideCfg) -> Any:
    # Respect SideCfg timeframes: 1H trigger, 4H execution
    tf_primary = getattr(cfg, "tf_primary", "1H")
    tf_confirm = getattr(cfg, "tf_confirm", "4H")   # dùng như execution
    f1 = features_by_tf.get(tf_primary, {}) or {}
    f4 = features_by_tf.get(tf_confirm, {}) or {}

    # Price theo trigger; ATR/NATR theo execution để dựng SL/TP
    df_trigger = f1.get('df')
    price = _price(df_trigger)
    atr  = float((f4.get('volatility', {}) or {}).get('atr', 0.0) or 0.0)
    natr = float((f4.get('volatility', {}) or {}).get('natr', 0.0) or 0.0)

    # trend/momentum/volume phía 1H (đưa về sign)
    def _trend_dir_from_features(ff) -> int:
        st = (ff.get('trend', {}) or {}).get('state')
        if st == 'up': return +1
        if st == 'down': return -1
        return 0

    def _momo_dir_from_features(ff) -> int:
        rsi = float((ff.get('momentum', {}) or {}).get('rsi', 50.0) or 50.0)
        return +1 if rsi > 50 else (-1 if rsi < 50 else 0)

    def _vol_dir_from_features(ff) -> int:
        vz = float((ff.get('volume', {}) or {}).get('vol_z20', 0.0) or 0.0)
        vr = float((ff.get('volume', {}) or {}).get('vol_ratio', 1.0) or 1.0)
        s = 0
        if vz > 0: s += 1
        if vr > 1.0: s += 1
        if vz < 0: s -= 1
        if vr < 1.0: s -= 1
        return 1 if s > 0 else (-1 if s < 0 else 0)

    trend_strength = _trend_dir_from_features(f1)
    momo_strength = _momo_dir_from_features(f1)
    volume_tilt = _vol_dir_from_features(f1)

    # levels để dựng TP ladder/SL confluence
    levels1h = f1.get('levels', {}) or {}
    levels4h = f4.get('levels', {}) or {}

    # Lấy evidences chính
    ev_pb  = _get_ev(eb, 'price_breakout')
    ev_pdn = _get_ev(eb, 'price_breakdown')
    ev_prc = _get_ev(eb, 'price_reclaim')
    ev_mr  = _get_ev(eb, 'mean_reversion')
    ev_div = _get_ev(eb, 'divergence')
    ev_rjt = _get_ev(eb, 'rejection')
    ev_tb  = _get_ev(eb, 'throwback')
    ev_pbk = _get_ev(eb, 'pullback')
    ev_fb_out = _get_ev(eb, 'false_breakout')
    ev_fb_dn  = _get_ev(eb, 'false_breakdown')
    ev_adapt  = _get_ev(eb, 'adaptive')  # meta: is_slow, liquidity_floor, regime ...

    # breakout flags
    breakout_ok = bool(ev_pb.get('ok') or ev_pdn.get('ok'))
    breakout_side = 'long' if ev_pb.get('ok') else ('short' if ev_pdn.get('ok') else None)

    # reclaim
    reclaim_ok = bool(ev_prc.get('ok'))
    reclaim_side = None
    try:
        ref = ev_prc.get('ref') or {}
        _s = ref.get('side')
        if _s in ('long','short'):
            reclaim_side = _s
    except Exception:
        pass

    # mean-reversion
    meanrev_ok = bool(ev_mr.get('ok'))
    meanrev_side = ev_mr.get('side') if ev_mr.get('side') in ('long','short') else None

    # divergence & rejection side
    div_side = ev_div.get('side') if ev_div.get('side') in ('long','short') else None
    rejection_side = ev_rjt.get('side') if ev_rjt.get('side') in ('long','short') else None

    # false-break (đảo hướng)
    false_break_long  = bool(ev_fb_dn.get('ok'))   # false breakdown → long bias
    false_break_short = bool(ev_fb_out.get('ok'))  # false breakout  → short bias

    # Retest zone (ưu tiên pullback/throwback; đã được normalize 'mid' trong build_evidence_bundle) :contentReference[oaicite:3]{index=3}
    def _zone_fields(ev: Dict[str, Any]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        try:
            z = ev.get('zone')
            mid = float(ev.get('mid')) if ev.get('mid') is not None else None
            if isinstance(z, (list, tuple)) and len(z) == 2:
                lo, hi = float(z[0]), float(z[1])
                return lo, hi, mid
        except Exception:
            pass
        return None, None, None

    lo1, hi1, mid1 = _zone_fields(ev_pbk)
    lo2, hi2, mid2 = _zone_fields(ev_tb)
    retest_zone_lo = lo1 if lo1 is not None else lo2
    retest_zone_hi = hi1 if hi1 is not None else hi2
    retest_zone_mid = mid1 if mid1 is not None else mid2

    # khoảng cách hiện tại đến mid theo ATR (để guard proximity)
    dist_atr = abs(((price or 0.0) - (retest_zone_mid or (price or 0.0))) / max(atr, 1e-9)) if retest_zone_mid is not None and price is not None else 0.0

    # đóng gói SI đơn giản bằng object kiểu dict
    class SIObj:
        pass

    si = SIObj()
    si.price = price
    si.atr = atr
    si.natr = natr
    si.dist_atr = dist_atr

    si.trend_strength = trend_strength
    si.momo_strength = momo_strength
    si.volume_tilt = volume_tilt

    si.levels1h = levels1h
    si.levels4h = levels4h

    si.breakout_ok = breakout_ok
    si.breakout_side = breakout_side

    si.reclaim_ok = reclaim_ok
    si.reclaim_side = reclaim_side

    si.retest_ok = True
    si.retest_zone_lo = retest_zone_lo
    si.retest_zone_hi = retest_zone_hi
    si.retest_zone_mid = retest_zone_mid

    si.meanrev_ok = meanrev_ok
    si.meanrev_side = meanrev_side

    si.div_side = div_side
    si.rejection_side = rejection_side

    si.false_break_long = false_break_long
    si.false_break_short = false_break_short

    # adaptive guards
    si.is_slow = bool(ev_adapt.get('is_slow', False)) if isinstance(ev_adapt, dict) else False
    si.liquidity_floor = bool(ev_adapt.get('liquidity_floor', False)) if isinstance(ev_adapt, dict) else False
    si.regime = (ev_adapt.get('regime') if isinstance(ev_adapt, dict) else None) or 'normal'

    return si


# -------------------------------
# Phân loại state + side (giữ nguyên)
# -------------------------------
def classify_state_with_side(si: SI, cfg: SideCfg) -> Tuple[str, Optional[str], Dict[str, Any]]:
    """
    Xác định state và side từ bộ chỉ báo 'si'.
    Trả về: (state, side, meta)
    """
    meta: Dict[str, Any] = {}

    # Helper an toàn (nếu field không tồn tại thì trả mặc định)
    def _safe_get_local(obj: Any, name: str, default: Any = None) -> Any:
        return getattr(obj, name, default)

    # Hướng trend/momentum/volume: quy về {-1, 0, +1}
    def _trend_dir(x: SI) -> int:
        val = float(_safe_get_local(x, "trend_strength", 0.0))
        return 0 if val == 0 else int(math.copysign(1, val))

    def _momo_dir(x: SI) -> int:
        val = float(_safe_get_local(x, "momo_strength", 0.0))
        return 0 if val == 0 else int(math.copysign(1, val))

    def _volume_dir(x: SI) -> int:
        val = float(_safe_get_local(x, "volume_tilt", 0.0))
        return 0 if val == 0 else int(math.copysign(1, val))

    # Meta context
    natr = float(_safe_get_local(si, "natr", float("nan")))
    dist_atr = float(_safe_get_local(si, "dist_atr", float("nan")))
    tr = _trend_dir(si)
    momo = _momo_dir(si)
    vdir = _volume_dir(si)
    meta.update(dict(natr=natr, dist_atr=dist_atr, trend=tr, momo=momo, v=vdir))

    # --- 1) Breakout/breakdown regime-aware ---
    if _safe_get_local(si, "breakout_ok", False) and _safe_get_local(si, "breakout_side") in ("long", "short"):
        pass  # placeholder (điểm phân xử chính ở retest/score)

    # --- 2) Retest regime (support vs resistance) với soft scoring ---
    if _safe_get_local(si, "retest_ok", True):
        long_score = 0.0
        short_score = 0.0

        # Context signals
        if tr > 0:
            long_score += 1.0
        elif tr < 0:
            short_score += 1.0

        if momo > 0:
            long_score += 0.5
        elif momo < 0:
            short_score += 0.5

        if _safe_get_local(si, "reclaim_ok", False):
            if _safe_get_local(si, "reclaim_side") == "long":
                long_score += 0.75
            elif _safe_get_local(si, "reclaim_side") == "short":
                short_score += 0.75

        # Vị trí so với zone
        zone_mid = _safe_get_local(si, "retest_zone_mid")
        price = _safe_get_local(si, "price")
        if zone_mid is not None and price is not None:
            if price <= zone_mid:
                long_score += 0.5  # gần hỗ trợ/mean
            if price >= zone_mid:
                short_score += 0.5  # gần kháng cự/mean

        # Volume tilt (bonus nếu cùng hướng)
        if vdir > 0:
            long_score += 0.25
        elif vdir < 0:
            short_score += 0.25

        # Mean-reversion hint
        if _safe_get_local(si, "meanrev_ok", False) and _safe_get_local(si, "meanrev_side") in ("long", "short"):
            if si.meanrev_side == "long":
                long_score += 0.5
            else:
                short_score += 0.5

        # False-break nghiêng mạnh về retest ngược hướng break
        if _safe_get_local(si, "false_break_long", False):
            long_score += 0.75
        if _safe_get_local(si, "false_break_short", False):
            short_score += 0.75

        # Rejection & divergence (nghiêng nhẹ)
        rej_side = _safe_get_local(si, "rejection_side")
        if rej_side == "long":
            long_score += 0.5
        elif rej_side == "short":
            short_score += 0.5

        div_side = _safe_get_local(si, "div_side")
        if div_side == "long":
            long_score += 0.25
        elif div_side == "short":
            short_score += 0.25

        meta.update(dict(long_score=long_score, short_score=short_score))

        # Tie & margin policy:
        diff = long_score - short_score
        if abs(diff) <= cfg.tie_eps or abs(diff) < cfg.side_margin:
            return "none_state", None, meta

        if diff > 0 and long_score >= cfg.retest_long_threshold:
            return "retest_support", "long", meta

        if diff < 0 and short_score >= cfg.retest_short_threshold:
            return "retest_resistance", "short", meta

    # --- 3) None ---
    return "none_state", None, meta


# ---------------------------------------
# Build setup (entry/SL/TP) theo side/zone
# ---------------------------------------
def build_setup(si: SI, state: str, side: Optional[str], cfg: SideCfg) -> Setup:
    st = Setup()
    price = _safe_get(si, "price")
    atr = float(_safe_get(si, "atr", 0.0) or 0.0)

    if price is None or atr <= 0 or side not in ("long","short"):
        return st  # thiếu dữ liệu → setup rỗng

    # Entry: ưu tiên mid của retest zone nếu có, else giá hiện tại
    z_mid = _safe_get(si, "retest_zone_mid")
    st.entry = float(z_mid if z_mid is not None else price)

    # SL: nếu có zone_lo/hi dùng làm mốc, có pad nhỏ; else dùng ATR
    z_lo = _safe_get(si, "retest_zone_lo")
    z_hi = _safe_get(si, "retest_zone_hi")
    pad = 0.1 * atr
    if side == "long":
        if z_lo is not None:
            st.sl = float(z_lo - pad)
        else:
            st.sl = float(st.entry - 0.8 * atr)
    else:
        if z_hi is not None:
            st.sl = float(z_hi + pad)
        else:
            st.sl = float(st.entry + 0.8 * atr)

    # enforce SL gap theo ATR
    st.sl = _ensure_sl_gap(st.entry, st.sl, atr, side, min_atr=0.6)

    # TP theo RR targets (ví dụ 1.2, 2.0, 3.0) — yêu cầu của bạn
    st.tps = _tp_by_rr(st.entry, st.sl, side, cfg.rr_targets)

    return st


# ---------------------------------------
# Quyết định 5-gates (tối giản, thực dụng)
# ---------------------------------------
def decide_5_gates(state: str, side: Optional[str], setup: Setup, si: SI, cfg: SideCfg, meta: Dict[str, Any]) -> Decision:
    dec = Decision(state=state, side=side, setup=setup, meta=dict(meta))
    reasons: List[str] = []

    # Guards theo adaptive
    if _safe_get(si, "liquidity_floor", False):
        reasons.append("liquidity_floor")
    if _safe_get(si, "is_slow", False):
        reasons.append("slow_market")

    # Thiếu dữ liệu thiết yếu?
    price = _safe_get(si, "price")
    atr = float(_safe_get(si, "atr", 0.0) or 0.0)
    if price is None or atr <= 0 or side not in ("long","short") or setup.entry is None or setup.sl is None:
        dec.decision = "WAIT"
        if side is None:
            reasons.append("no_side")
        if price is None:
            reasons.append("no_price")
        if atr <= 0:
            reasons.append("no_atr")
        dec.reasons = sorted(set(reasons))
        return dec

    # Proximity guard: không vào nếu quá xa entry (>0.75 ATR)
    if abs(price - setup.entry) > (0.75 * atr):
        reasons.append("far_from_entry")

    # RR guard: RR tới TP1 tối thiểu 1.0
    tp1 = setup.tps[0] if setup.tps else None
    rr1 = _rr(side, setup.entry, setup.sl, tp1) if tp1 is not None else 0.0
    if rr1 < 1.0:
        reasons.append("rr_too_low")

    # Tổng hợp quyết định
    if not reasons:
        dec.decision = "ENTER"
    else:
        dec.decision = "WAIT"
    dec.reasons = sorted(set(reasons))
    return dec


# ============== Orchestrator =====================
def run_side_state_core(
    features_by_tf: Dict[str, Dict[str, Any]],
    eb: Any,
    cfg: Optional[SideCfg] = None,
) -> Decision:
    """
    Orchestrator:
    - Thu thập side-indicators
    - Phân loại state/side
    - Build setup
    - Ra quyết định (5 cổng)
    """
    cfg = cfg or SideCfg()

    si: SI = collect_side_indicators(features_by_tf, eb, cfg)
    state, side, meta = classify_state_with_side(si, cfg)
    setup: Setup = build_setup(si, state, side, cfg)
    dec: Decision = decide_5_gates(state, side, setup, si, cfg, meta)

    return dec
