from __future__ import annotations

from dataclasses import dataclass
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
    adx_trend_thr: float = 18.0         # range-like khi ADX dưới ngưỡng
    break_buffer_atr: float = 0.20      # buffer tính theo ATR quanh mốc break

    # Tie handling
    tie_eps: float = 1e-6               # sai số tuyệt đối để coi như hoà
    side_margin: float = 0.05           # yêu cầu chênh tối thiểu để chọn side

    # Retest score gates
    retest_long_threshold: float = 0.6
    retest_short_threshold: float = 0.6

    # TP ladder mặc định cho tính RR
    rr_targets: Tuple[float, float, float] = (1.2, 2.0, 3.0)

    # Timeframes
    tf_primary: str = "1H"
    tf_confirm: str = "4H"


# Các kiểu dữ liệu/stub để type-hint (thực tế có thể đã được định nghĩa nơi khác)
SI = Any
Setup = Any
Decision = Dict[str, Any]


def classify_state_with_side(si: SI, cfg: SideCfg) -> Tuple[str, Optional[str], Dict[str, Any]]:
    """
    Xác định state và side từ bộ chỉ báo 'si'.
    Trả về: (state, side, meta)
    """
    meta: Dict[str, Any] = {}

    # Helper an toàn (nếu field không tồn tại thì trả mặc định)
    def _safe_get(obj: Any, name: str, default: Any = None) -> Any:
        return getattr(obj, name, default)

    # Hướng trend/momentum/volume: quy về {-1, 0, +1}
    def _trend_dir(x: SI) -> int:
        val = float(_safe_get(x, "trend_strength", 0.0))
        return 0 if val == 0 else int(math.copysign(1, val))

    def _momo_dir(x: SI) -> int:
        val = float(_safe_get(x, "momo_strength", 0.0))
        return 0 if val == 0 else int(math.copysign(1, val))

    def _volume_dir(x: SI) -> int:
        val = float(_safe_get(x, "volume_tilt", 0.0))
        return 0 if val == 0 else int(math.copysign(1, val))

    # Meta context
    natr = float(_safe_get(si, "natr", float("nan")))
    dist_atr = float(_safe_get(si, "dist_atr", float("nan")))
    tr = _trend_dir(si)
    momo = _momo_dir(si)
    vdir = _volume_dir(si)
    meta.update(dict(natr=natr, dist_atr=dist_atr, trend=tr, momo=momo, v=vdir))

    # --- 1) Breakout/breakdown regime-aware ---
    # (Giữ chỗ theo mô tả, phần phân xử chính nằm trong khối retest/score bên dưới)
    if _safe_get(si, "breakout_ok", False) and _safe_get(si, "breakout_side") in ("long", "short"):
        pass

    # --- 2) Retest regime (support vs resistance) với soft scoring ---
    if _safe_get(si, "retest_ok", True):
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

        if _safe_get(si, "reclaim_ok", False):
            if _safe_get(si, "reclaim_side") == "long":
                long_score += 0.75
            elif _safe_get(si, "reclaim_side") == "short":
                short_score += 0.75

        # Vị trí so với zone
        zone_mid = _safe_get(si, "retest_zone_mid")
        price = _safe_get(si, "price")
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
        if _safe_get(si, "meanrev_ok", False) and _safe_get(si, "meanrev_side") in ("long", "short"):
            if si.meanrev_side == "long":
                long_score += 0.5
            else:
                short_score += 0.5

        # False-break nghiêng mạnh về retest ngược hướng break
        if _safe_get(si, "false_break_long", False):
            long_score += 0.75
        if _safe_get(si, "false_break_short", False):
            short_score += 0.75

        # Rejection & divergence (nghiêng nhẹ)
        rej_side = _safe_get(si, "rejection_side")
        if rej_side == "long":
            long_score += 0.5
        elif rej_side == "short":
            short_score += 0.5

        div_side = _safe_get(si, "div_side")
        if div_side == "long":
            long_score += 0.25
        elif div_side == "short":
            short_score += 0.25

        meta.update(dict(long_score=long_score, short_score=short_score))

        # Tie & margin policy:
        # - Nếu |diff| <= tie_eps hoặc |diff| < side_margin -> none_state (WAIT)
        # - Ngược lại, cần vượt ngưỡng retest_*_threshold tương ứng
        diff = long_score - short_score
        if abs(diff) <= cfg.tie_eps or abs(diff) < cfg.side_margin:
            return "none_state", None, meta

        if diff > 0 and long_score >= cfg.retest_long_threshold:
            return "retest_support", "long", meta

        if diff < 0 and short_score >= cfg.retest_short_threshold:
            return "retest_resistance", "short", meta

    # --- 3) None ---
    return "none_state", None, meta

def collect_side_indicators(
    features_by_tf: Dict[str, Dict[str, Any]],
    eb: Any,
    cfg: SideCfg,
) -> Any:
    """
    Gom các chỉ báo side-aware từ evidence/features (dict-safe).
    Trả về một đối tượng có thuộc tính như si.xxx mà classifier kỳ vọng,
    kèm các flag phụ: sideways, bb_expanding, volatility_breakout, trend_follow_ready,
    candles, liquidity, adaptive...
    """
    # Lấy evidence dict-safe
    E = eb.get("evidence", eb) if isinstance(eb, dict) else {}
    if not isinstance(E, dict):
        E = {}

    def _ev(name):
        v = E.get(name, {})
        return v if isinstance(v, dict) else {}

    def _ok(name):
        return bool(_ev(name).get("ok", False))

    def _side_from(name):
        s = _ev(name).get("side")
        return s if s in ("long", "short") else None

    def _num_from(name, key, default=None):
        try:
            v = _ev(name).get(key, default)
            return float(v) if v is not None else default
        except Exception:
            return default

    # Giá hiện tại từ features (tf_primary)
    price = None
    try:
        df = (features_by_tf or {}).get(cfg.tf_primary, {}).get("df")
        if df is not None and len(df) > 0:
            price = float(df.iloc[-1]["close"])
    except Exception:
        price = None

    # natr từ df nếu có, không thì giữ NaN
    natr = float("nan")
    try:
        dfp = (features_by_tf or {}).get(cfg.tf_primary, {}).get("df")
        if dfp is not None and len(dfp) > 0 and "natr" in dfp.columns:
            natr = float(dfp.iloc[-1]["natr"])
    except Exception:
        pass

    # dist_atr: giữ NaN nếu bạn không tính ở nơi khác
    dist_atr = float("nan")

    # trend_strength từ trend_alignment.side
    trend_strength = 0.0
    ta = _side_from("trend_alignment")
    if ta == "long":
        trend_strength = 1.0
    elif ta == "short":
        trend_strength = -1.0

    # momentum: dùng hint từ breakout + momentum.primary.ok nếu có
    side_hint = "long" if _ok("price_breakout") else ("short" if _ok("price_breakdown") else None)
    mom_ok = bool((_ev("momentum").get("primary") or {}).get("ok", False))
    momo_strength = 1.0 if (side_hint == "long" and mom_ok) else (
        -1.0 if (side_hint == "short" and mom_ok) else 0.0
    )

    # volume tilt: nếu bundle có side cho 'volume' thì dùng, không có thì để 0.0
    volume_tilt = 0.0
    vside = _side_from("volume")
    if vside == "long":
        volume_tilt = 1.0
    elif vside == "short":
        volume_tilt = -1.0

    # breakout/breakdown
    breakout_ok = bool(_ok("price_breakout") or _ok("price_breakdown"))
    breakout_side = "long" if _ok("price_breakout") else ("short" if _ok("price_breakdown") else None)

    # pullback/throwback -> retest
    retest_ok = bool(_ok("pullback") or _ok("throwback"))
    retest_zone_mid = None
    for name in ("pullback", "throwback"):
        mid = _num_from(name, "mid", None)
        if mid is None:
            z = _ev(name).get("zone")
            if isinstance(z, (list, tuple)) and len(z) == 2:
                try:
                    mid = (float(z[0]) + float(z[1])) / 2.0
                except Exception:
                    mid = None
        if mid is not None:
            retest_zone_mid = float(mid)
            break

    # reclaim
    rc = _ev("price_reclaim")
    reclaim_ok = bool(rc.get("ok", False))
    rside = rc.get("ref", {}).get("side") if isinstance(rc.get("ref"), dict) else rc.get("side")
    reclaim_side = rside if rside in ("long", "short") else None

    # mean reversion
    meanrev_ok = _ok("mean_reversion")
    meanrev_side = _side_from("mean_reversion")

    # false break:
    # - false_breakout: fail break up -> nghiêng short
    # - false_breakdown: fail break down -> nghiêng long
    false_break_short = bool(_ok("false_breakout"))
    false_break_long = bool(_ok("false_breakdown"))

    # rejection & divergence
    rejection_side = _side_from("rejection")
    div_side = _side_from("divergence")

    # ====== CÁC FLAG PHỤ BỔ SUNG ======

    # 1) sideways / bb_expanding / volatility_breakout
    sideways_ok = _ok("sideways")
    bb_expand_ok = _ok("bb_expanding")
    vol_breakout_ok = _ok("volatility_breakout")

    # 2) trend_follow_ready.{long, short}
    tfr = _ev("trend_follow_ready")
    trend_follow_ready_long = bool((tfr.get("long") or {}).get("ok", False))
    trend_follow_ready_short = bool((tfr.get("short") or {}).get("ok", False))

    # 3) candles (hướng nếu có)
    cdl = _ev("candles")
    candle_ok = bool(cdl.get("ok", False))
    candle_side = cdl.get("side") if cdl.get("side") in ("long", "short") else None

    # 4) liquidity meta
    liq = _ev("liquidity")
    near_heavy_zone = bool(liq.get("near_heavy_zone", False))
    heavy_zone_mid = liq.get("nearest_zone_mid")
    try:
        heavy_zone_mid = float(heavy_zone_mid) if heavy_zone_mid is not None else None
    except Exception:
        heavy_zone_mid = None

    # 5) adaptive meta (regime + slow-market guards)
    adp = _ev("adaptive")
    market_regime = adp.get("regime")
    is_slow = bool(adp.get("is_slow", False))
    liquidity_floor = bool(adp.get("liquidity_floor", False))

    # Trả đối tượng có thuộc tính như si.xxx; dùng SimpleNamespace để không đổi định nghĩa SI
    from types import SimpleNamespace
    return SimpleNamespace(
        natr=natr,
        dist_atr=dist_atr,
        trend_strength=trend_strength,
        momo_strength=momo_strength,
        volume_tilt=volume_tilt,
        price=price,
        breakout_ok=breakout_ok,
        breakout_side=breakout_side,
        retest_ok=retest_ok,
        retest_zone_mid=retest_zone_mid,
        reclaim_ok=reclaim_ok,
        reclaim_side=reclaim_side,
        meanrev_ok=meanrev_ok,
        meanrev_side=meanrev_side,
        false_break_long=false_break_long,
        false_break_short=false_break_short,
        rejection_side=rejection_side,
        div_side=div_side,
        # Phụ:
        sideways_ok=sideways_ok,
        bb_expand_ok=bb_expand_ok,
        vol_breakout_ok=vol_breakout_ok,
        trend_follow_ready_long=trend_follow_ready_long,
        trend_follow_ready_short=trend_follow_ready_short,
        candle_ok=candle_ok,
        candle_side=candle_side,
        near_heavy_zone=near_heavy_zone,
        heavy_zone_mid=heavy_zone_mid,
        market_regime=market_regime,
        is_slow=is_slow,
        liquidity_floor=liquidity_floor,
    )


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
