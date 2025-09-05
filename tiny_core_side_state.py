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
