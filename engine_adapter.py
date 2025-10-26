"""
engine_adapter.py
- Thin wrapper so main.py can call decide(...) without decision_engine.py.
- Uses tiny_core_side_state to compute decision and formats a legacy-compatible dict.
"""
from typing import Dict, Any, List, Optional, Iterable
from tiny_core_side_state import SideCfg, run_side_state_core
import os
from evidence_evaluators import _reversal_signal
import math

# ===============================================================
# Adaptive Enhancements Patch (Range + Early + AutoConfig)
# ===============================================================
from dataclasses import dataclass, field
from typing import Tuple

# ===============================================================
# Adaptive Timeframe Switch (4H↔1H Execution)
# ===============================================================

def _adaptive_timeframe(features_by_tf: Dict[str, Any]) -> str:
    """
    Tự động chọn TF thực thi (execution) giữa 4H và 1H.
    - 4H: dùng khi thị trường có trend rõ, biến động mạnh.
    - 1H: dùng khi sideway, BB co, ATR nhỏ, ADX yếu.
    """
    try:
        df4 = features_by_tf.get("4H", {}).get("df")
        if df4 is None or len(df4) < 10:
            return "4H"
        import math
        atr = float(df4["atr14"].iloc[-2]) if "atr14" in df4.columns else float("nan")
        price = float(df4["close"].iloc[-2]) if "close" in df4.columns else float("nan")
        bbw = float(df4["bb_width_pct"].iloc[-2]) if "bb_width_pct" in df4.columns else float("nan")
        adx = float(df4["adx"].iloc[-2]) if "adx" in df4.columns else float("nan")
        natr = (atr / price) * 100 if (price and price > 0) else 0

        # --- Logic chọn TF ---
        if (bbw < 1.5 and adx < 25) or natr < 0.03:
            return "1H"   # thị trường chậm / sideway
        return "4H"
    except Exception:
        return "4H"

# ---------- RANGE DETECTION ----------
def _detect_ranging_market(features_by_tf: Dict[str, Any]) -> bool:
    """Detect sustained ranging conditions via BB width + ADX + range size"""
    try:
        df4 = features_by_tf.get("4H", {}).get("df")
        if df4 is None or len(df4) < 20:
            return False
        bb_width = float(df4["bb_width_pct"].iloc[-2]) if "bb_width_pct" in df4.columns else float("nan")
        adx = float(df4["adx"].iloc[-2]) if "adx" in df4.columns else float("nan")
        recent_high = df4["high"].tail(10).max()
        recent_low = df4["low"].tail(10).min()
        range_pct = (recent_high - recent_low) / recent_low * 100
        return (bb_width < 2.0 and (adx < 25 or not adx == adx) and range_pct < 8.0)
    except Exception:
        return False

# ===============================================================
#  LOW-VOL / BB-AWARE POST PROCESSING
# ===============================================================

def _bb_pack(features_by_tf: Dict[str, Any], tf: str = "1H") -> Tuple[float,float,float,float,float]:
    """Trả (price, atr, bb_lower, bb_mid, bb_upper) của TF (mặc định 1H)."""
    import math
    df = ((features_by_tf or {}).get(tf) or {}).get("df")
    if df is None or len(df) < 5:
        return float("nan"), float("nan"), float("nan"), float("nan"), float("nan")
    def _safe(name, idx=-2):
        try:
            return float(df[name].iloc[idx])
        except Exception:
            return float("nan")
    price = _safe("close")
    atr   = _safe("atr14")
    bl    = _safe("bb_lower")
    bm    = _safe("bb_mid")
    bu    = _safe("bb_upper")
    return price, atr, bl, bm, bu

def _natr_pct(features_by_tf: Dict[str, Any], tf: str = "1H") -> float:
    """NATR (%) ~ ATR/price*100"""
    p, atr, *_ = _bb_pack(features_by_tf, tf=tf)
    if not (p and atr and p>0): return 0.0
    return (atr/p)*100

def _cap_sl_in_low_vol(entry: float, sl: float, side: str, atr: float, natr_pct: float) -> float:
    """
    Giới hạn SL khi NATR thấp:
      - NATR < 2%  → |SL-entry| ≤ 0.60×ATR
      - NATR < 3.5%→ |SL-entry| ≤ 0.80×ATR
    """
    import math
    if not (math.isfinite(entry) and math.isfinite(sl) and math.isfinite(atr)): return sl
    max_mult = None
    if natr_pct < 2.0: max_mult = 0.6
    elif natr_pct < 3.5: max_mult = 0.8
    if max_mult is None: return sl
    gap = abs(sl - entry)
    cap_gap = max_mult * atr
    if gap <= cap_gap: return sl
    if side == "short": return float(entry + cap_gap)
    if side == "long":  return float(entry - cap_gap)
    return sl

def _pull_tp1_inside_band(tp1: float, side: str, bb_lower: float, bb_upper: float, atr: float) -> float:
    """Kéo TP1 “lọt vào trong dải BB” ~ 0.15×ATR."""
    import math
    if not math.isfinite(tp1) or not math.isfinite(atr): return tp1
    pad = max(0.0, 0.15 * atr)
    if side == "short" and math.isfinite(bb_lower): return max(tp1, bb_lower + pad)
    if side == "long"  and math.isfinite(bb_upper): return min(tp1, bb_upper - pad)
    return tp1

def _rescale_tps_after_tp1_shift(entry: float, tps: list[float], side: str,
                                 tp1_old: float, tp1_new: float) -> list[float]:
    """
    Khi TP1 bị kéo vào trong dải BB → scale lại toàn bộ TP2–TP5 để giữ nhịp RR.
    """
    import math
    if not (tps and math.isfinite(entry) and math.isfinite(tp1_old) and math.isfinite(tp1_new)): return tps
    if tp1_new == tp1_old or len(tps) < 2: return tps
    try:
        if side == "short":
            scale = (entry - tp1_new) / max(1e-9, (entry - tp1_old))
        elif side == "long":
            scale = (tp1_new - entry) / max(1e-9, (tp1_old - entry))
        else:
            return tps
    except Exception:
        return tps
    if not math.isfinite(scale): return tps

    new_tps = [tp1_new]
    for tp in tps[1:]:
        if side == "short":
            new_tp = entry - (entry - tp) * scale
        else:
            new_tp = entry + (tp - entry) * scale
        new_tps.append(new_tp)
    return new_tps

def _range_trading_setup(features_by_tf: Dict[str, Any], side: str) -> Optional[Dict[str, Any]]:
    """Construct a bounce/rejection setup when market is in range."""
    try:
        df1 = features_by_tf.get("1H", {}).get("df")
        if df1 is None or len(df1) < 20:
            return None
        price = float(df1["close"].iloc[-1])
        atr = float(df1["atr14"].iloc[-2]) if "atr14" in df1.columns else 0
        hi = df1["high"].tail(20).max()
        lo = df1["low"].tail(20).min()
        mid = (hi + lo) / 2

        if side == "long" and price <= lo + 0.3 * atr:
            return {"entry": price, "sl": lo - 0.5 * atr, "tp1": mid, "tp2": hi - 0.5 * atr, "strategy": "range_bounce"}
        elif side == "short" and price >= hi - 0.3 * atr:
            return {"entry": price, "sl": hi + 0.5 * atr, "tp1": mid, "tp2": lo + 0.5 * atr, "strategy": "range_reject"}
    except Exception:
        return None
    return None


# ---------- EARLY TREND DETECTION ----------
def _early_trend_detection(features_by_tf: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    """Detect early momentum breakout before confirmation."""
    try:
        df1 = features_by_tf.get("1H", {}).get("df")
        if df1 is None or len(df1) < 5:
            return None
        rsi = float(df1["rsi14"].iloc[-2]) if "rsi14" in df1.columns else 50
        vol = float(df1["vol_ratio"].iloc[-2]) if "vol_ratio" in df1.columns else 1.0
        close = float(df1["close"].iloc[-1])
        ema20 = float(df1["ema20"].iloc[-1]) if "ema20" in df1.columns else close
        atr = float(df1["atr14"].iloc[-2]) if "atr14" in df1.columns else 0.0

        if vol > 1.8:
            if close > ema20 and 45 < rsi < 70:
                prev_low = float(df1["low"].iloc[-3])
                if close > prev_low + 0.2 * atr:
                    return ("early_trend", "long")
            if close < ema20 and 30 < rsi < 55:
                prev_high = float(df1["high"].iloc[-3])
                if close < prev_high - 0.2 * atr:
                    return ("early_trend", "short")
    except Exception:
        return None
    return None


def _relax_guards_for_early_entries(decision: Dict[str, Any], features_by_tf: Dict[str, Any]) -> Dict[str, Any]:
    """Loosen WAIT → ENTER for strong early-trend conditions."""
    try:
        if decision.get("decision") != "WAIT":
            return decision
        early = _early_trend_detection(features_by_tf)
        if not early:
            return decision
        state, side = early
        decision["decision"] = "ENTER"
        decision["side"] = side
        decision["state"] = state
        meta = dict(decision.get("meta") or {})
        meta["early_entry"] = True
        decision["meta"] = meta
        decision["reasons"] = [r for r in decision.get("reasons", []) if "guard" not in r]
    except Exception:
        pass
    return decision


# ---------- ADAPTIVE PRESET CONFIG ----------
def _auto_select_preset(features_by_tf: Dict[str, Any]) -> str:
    """Auto choose mode: aggressive / conservative / range_specialist / balanced."""
    try:
        df4 = features_by_tf.get("4H", {}).get("df")
        if df4 is None:
            return "balanced"
        atr = float(df4["atr14"].iloc[-2]) if "atr14" in df4.columns else 0
        price = float(df4["close"].iloc[-1])
        vol_pct = atr / price * 100 if price > 0 else 0
        adx = float(df4["adx"].iloc[-2]) if "adx" in df4.columns else 0
        if vol_pct > 0.1 and adx > 30:
            return "aggressive"
        if vol_pct < 0.03:
            return "range_specialist"
        if vol_pct < 0.06:
            return "conservative"
    except Exception:
        pass
    return "balanced"


def enhanced_decide(symbol: str, timeframe: str, features_by_tf: Dict[str, Any], evidence_bundle: Dict[str, Any]) -> Dict[str, Any]:
    """
    Wrapper around decide() that adds:
    - Range trading overlay
    - Early entry relaxation
    - Auto-configuration metadata
    """
    from engine_adapter import decide as _base_decide
    base = _base_decide(symbol, timeframe, features_by_tf, evidence_bundle)
    base = _relax_guards_for_early_entries(base, features_by_tf)

    # 0️⃣ Auto switch TF execution based on regime
    exec_tf = _adaptive_timeframe(features_by_tf)
    try:
        # Đồng bộ config để các hàm tính ATR/SL/TP sử dụng đúng TF
        from tiny_core_side_state import AdaptiveConfig
        cfg = AdaptiveConfig().adjust(features_by_tf)
        cfg.tf_primary = exec_tf
        cfg.tf_confirm = exec_tf
        meta = dict(base.get("meta") or {})
        meta["exec_tf_auto"] = exec_tf
        meta["exec_tf_cfg"] = {"tf_primary": cfg.tf_primary, "tf_confirm": cfg.tf_confirm}
        base["meta"] = meta
    except Exception:
        meta = dict(base.get("meta") or {})
        meta["exec_tf_auto"] = exec_tf
        base["meta"] = meta

    # Gắn nhãn vào log để tiện theo dõi hiệu suất 1H/4H
    logs = dict(base.get("logs") or {})
    logs["adaptive_tf"] = {"selected": exec_tf}
    base["logs"] = logs

    # 1️⃣ Nếu WAIT + thị trường đang range → dựng setup bounce/reject
    if base.get("decision") == "WAIT" and _detect_ranging_market(features_by_tf):
        rng_long = _range_trading_setup(features_by_tf, "long")
        rng_short = _range_trading_setup(features_by_tf, "short")
        chosen = rng_long or rng_short
        if chosen:
            base["decision"] = "ENTER"
            base["side"] = chosen["strategy"].split("_")[-1]
            base["setup"] = chosen
            base["meta"]["range_mode"] = True

    # 2️⃣ Hậu xử lý cho range/ATR thấp → thu SL, kéo TP1 vào dải, scale TP2–TP5
    try:
        if base.get("decision") == "ENTER":
            side = base.get("side") or (base.get("setup") or {}).get("side")
            stp  = base.get("setup") or {}
            entry = float(stp.get("entry") or 0.0)
            sl    = float(stp.get("sl") or 0.0)
            tps   = list(stp.get("tps") or [])
            p, atr, bb_l, bb_m, bb_u = _bb_pack(features_by_tf, "1H")
            natr = _natr_pct(features_by_tf, "1H")

            # --- Cap SL trong low-vol regime ---
            if entry and sl and atr and side in ("long","short"):
                sl_new = _cap_sl_in_low_vol(entry, sl, side, atr, natr)
                if sl_new != sl:
                    sl = sl_new

            # --- TP1 band-snap + rescale ladder ---
            if tps and atr and side in ("long","short"):
                tp1_old = float(tps[0])
                tp1_new = _pull_tp1_inside_band(tp1_old, side, bb_l, bb_u, atr)
                if tp1_new != tp1_old:
                    tps = _rescale_tps_after_tp1_shift(entry, tps, side, tp1_old, tp1_new)

            # --- Ghi kết quả ---
            stp["sl"] = sl
            stp["tps"] = tps
            base["setup"] = stp
            meta = dict(base.get("meta") or {})
            meta.update({
                "natr_pct_1h": natr,
                "sl_capped_lowvol": bool(natr and natr<3.5),
                "tp1_inside_band": True if (tps and tps[0]) else False,
                "tp_rescaled_after_band_snap": (True if (tps and len(tps)>1) else False),
            })
            base["meta"] = meta
    except Exception:
        pass

    base["meta"]["strategy_mode"] = _auto_select_preset(features_by_tf)
    base["meta"]["auto_optimized"] = True
    return base

def _last_closed_bar(df):
    """
    Return the last *closed* bar for streaming safety:
    use df.iloc[-2] if available, else the last row.
    """
    try:
        n = len(df)
        if n >= 2:
            return df.iloc[-2]
        elif n == 1:
            return df.iloc[-1]
    except Exception:
        pass
    return None

def _guard_recent_1h_opposite(feats: Dict[str, Any], side: str) -> Dict[str, Any]:
    """
    Guard: Trong N (=3) nến 1H *đã đóng* gần nhất, nếu xuất hiện:
      - Engulfing NGƯỢC CHIỀU so với 'side', HOẶC
      - Marubozu/WRB ngược chiều (thân rất lớn, râu ngắn, range ≥ k*ATR)
      - (Tùy chọn) tần suất nến ngược chiều cao (>= FREQ_MIN) với thân đủ lớn (BODY_MIN%)
        và/hoặc vol_ratio spike
        Và tại THỜI ĐIỂM QUÉT, nến 1H đang chạy (chưa đóng) cũng KHÔNG được là Marubozu/WRB ngược chiều.
    → Ép WAIT để tránh vào lệnh sau cú impulse ngược chiều.
    ENV:
      OPP_1H_LOOKBACK (3)       – số nến 1H đã đóng để xét
      OPP_1H_BODY_MIN (30)      – % thân tối thiểu để tính là nến ngược chiều
      OPP_1H_FREQ_MIN (2)       – số lượng tối thiểu nến ngược chiều trong N nến
      OPP_1H_USE_VOL  (1)       – bật lọc volume cho nến ngược chiều
      OPP_1H_VOLR_THR (1.4)     – ngưỡng vol_ratio
      MARU_BODY_MIN   (70)      – % thân coi là marubozu
      MARU_WICK_MAX   (15)      – % mỗi râu tối đa
      MARU_ATR_MULT   (1.2)     – (high-low) ≥ k*ATR
      OPP_4H_CONFIRM  (0/1)     – nếu 1: yêu cầu nến 4H vừa đóng cũng là impulse cùng chiều để block mạnh
      OPP_CHECK_PARTIAL_1H (1)  – nếu 1: kiểm tra thêm nến 1H chưa đóng có phải marubozu/WRB ngược chiều không
    """
    out = {"block": False, "why": ""}
    try:
        if side not in ("long", "short"):
            return out
        df1 = ((feats or {}).get("1H") or {}).get("df")
        if df1 is None or len(df1) < 4:
            return out

        # Adaptive skip cho 1H regime hoặc NATR thấp
        try:
            df4 = ((feats or {}).get("4H") or {}).get("df")
            atr = float(df4["atr14"].iloc[-2]) if "atr14" in df4.columns else None
            price = float(df4["close"].iloc[-2]) if "close" in df4.columns else None
            natr = (atr / price) * 100 if atr and price and price > 0 else 0
            if natr < 0.04:
                return {"block": False, "why": "skip_guard_1h_opposite_lowvol"}
            meta_tf = str(((feats.get("meta") or {}).get("exec_tf_auto")) or "")
            if meta_tf == "1H":
                return {"block": False, "why": "skip_guard_1h_opposite_exec_1H"}
        except Exception:
            pass
        import os as _os
        N        = int(_os.getenv("OPP_1H_LOOKBACK", "3"))
        BODYMIN  = float(_os.getenv("OPP_1H_BODY_MIN", "30"))
        FREQMIN  = int(_os.getenv("OPP_1H_FREQ_MIN", "2"))
        USEVOL   = str(_os.getenv("OPP_1H_USE_VOL", "1")).lower() in ("1","true","yes")
        VOLTHR   = float(_os.getenv("OPP_1H_VOLR_THR", "1.4"))
        MARU_BODY= float(_os.getenv("MARU_BODY_MIN", "70"))
        MARU_WICK= float(_os.getenv("MARU_WICK_MAX", "15"))
        MARU_K   = float(_os.getenv("MARU_ATR_MULT", "1.2"))
        CONFIRM4H= str(_os.getenv("OPP_4H_CONFIRM", "0")).lower() in ("1","true","yes")
        CHECK_PARTIAL = str(_os.getenv("OPP_CHECK_PARTIAL_1H", "1")).lower() in ("1","true","yes")

        # Lấy N nến 1H đã đóng: [-2-N+1 : -1]
        end_idx   = -1
        start_idx = max(-1 - N, -len(df1))
        window = df1.iloc[start_idx:end_idx]
        if window is None or len(window) < max(2, N):
            return out

        def _engulf(curr, prev, want_bearish: bool) -> bool:
            try:
                co, cc = float(curr["open"]), float(curr["close"])
                po, pc = float(prev["open"]), float(prev["close"])
                if want_bearish:
                    return (cc < co) and (co >= pc) and (cc <= po)
                else:
                    return (cc > co) and (co <= pc) and (cc >= po)
            except Exception:
                return False

        def _is_marubozu(cur, want_bearish: bool, atr_val: float) -> bool:
            try:
                o, c = float(cur.open), float(cur.close)
                h, l = float(cur.high), float(cur.low)
                rng = max(1e-12, h - l)
                body = abs(c - o)
                body_pct = body / rng * 100.0
                upper = (h - max(o, c)) / rng * 100.0
                lower = (min(o, c) - l) / rng * 100.0
                dir_ok = (c < o) if want_bearish else (c > o)
                atr_ok = True
                if atr_val and atr_val > 0:
                    atr_ok = (rng >= MARU_K * atr_val)
                return dir_ok and (body_pct >= MARU_BODY) and (upper <= MARU_WICK) and (lower <= MARU_WICK) and atr_ok
            except Exception:
                return False

        opp_hits = 0
        has_opp_engulf = False
        has_opp_maru   = False
        for i in range(len(window)):
            cur = window.iloc[i]
            prev = window.iloc[i - 1] if i - 1 >= 0 else None
            try:
                o, c = float(cur.open), float(cur.close)
                h, l = float(cur.high), float(cur.low)
            except Exception:
                continue
            rng = max(1e-12, h - l)
            # body %
            try:
                body_pct = float(cur.body_pct) if "body_pct" in window.columns else abs(c - o) / rng * 100.0
            except Exception:
                body_pct = abs(c - o) / rng * 100.0
            is_red, is_green = (c < o), (c > o)
            vol_ok = True
            if USEVOL:
                try:
                    vr = float(cur.vol_ratio) if "vol_ratio" in window.columns else 1.0
                except Exception:
                    vr = 1.0
                vol_ok = (vr >= VOLTHR)
            # nến ngược chiều đủ lớn
            opp_candle = (is_red if side == "long" else is_green) and (body_pct >= BODYMIN) and vol_ok
            if opp_candle:
                opp_hits += 1
            # engulf ngược chiều
            if prev is not None:
                if side == "long" and _engulf(cur, prev, want_bearish=True):
                    has_opp_engulf = True
                if side == "short" and _engulf(cur, prev, want_bearish=False):
                    has_opp_engulf = True
            # marubozu/WRB ngược chiều
            try:
                atr_val = float(cur.atr14) if "atr14" in window.columns else (float(cur.atr) if "atr" in window.columns else None)
            except Exception:
                atr_val = None
            if side == "long" and _is_marubozu(cur, want_bearish=True, atr_val=atr_val):
                has_opp_maru = True
            if side == "short" and _is_marubozu(cur, want_bearish=False, atr_val=atr_val):
                has_opp_maru = True

        # Tùy chọn xác nhận 4H
        confirm4h_ok = True
        if CONFIRM4H:
            try:
                df4 = ((feats or {}).get("4H") or {}).get("df")
                if df4 is not None and len(df4) >= 2:
                    last4 = df4.iloc[-2]
                    try:
                        atr4 = float(last4.atr14 if "atr14" in df4.columns else last4.atr)
                    except Exception:
                        atr4 = None
                    if side == "long":
                        confirm4h_ok = _is_marubozu(last4, want_bearish=True, atr_val=atr4)
                    else:
                        confirm4h_ok = _is_marubozu(last4, want_bearish=False, atr_val=atr4)
                else:
                    confirm4h_ok = False
            except Exception:
                confirm4h_ok = False

        if has_opp_engulf or has_opp_maru or (opp_hits >= FREQMIN):
            reason = []
            if has_opp_engulf: reason.append("opp_engulf_1H")
            if has_opp_maru:   reason.append("opp_marubozu_1H")
            if opp_hits >= FREQMIN: reason.append(f"opp_freq_1H={opp_hits}/{N}(body>={BODYMIN}%)")
            if CONFIRM4H:
                out["block"] = bool(confirm4h_ok)
                if not out["block"]:
                    reason.append("4H_no_confirm")
            else:
                out["block"] = True
            out["why"] = ", ".join(reason)
            return out

        # -------------------------------
        # (MỚI) Kiểm tra nến 1H hiện tại
        # -------------------------------
        if CHECK_PARTIAL and len(df1) >= 2:
            try:
                cur_partial = df1.iloc[-1]          # nến đang chạy
                last_closed = df1.iloc[-2]          # nến đã đóng gần nhất
                # Tránh double-check khi data source không stream (2 cái trùng nhau)
                is_same_bar = bool(cur_partial.name == last_closed.name)
                if not is_same_bar:
                    # ATR ưu tiên ngay trên nến hiện tại; fallback về cột khác nếu thiếu
                    try:
                        atr_p = float(cur_partial.atr14 if "atr14" in df1.columns else cur_partial.atr)
                    except Exception:
                        # fallback nhẹ: dùng ATR của nến đã đóng nếu hiện tại thiếu
                        try:
                            atr_p = float(last_closed.atr14 if "atr14" in df1.columns else last_closed.atr)
                        except Exception:
                            atr_p = None
                    want_bearish = (side == "long")
                    if _is_marubozu(cur_partial, want_bearish=want_bearish, atr_val=atr_p):
                        # Nếu yêu cầu xác nhận 4H, tôn trọng confirm4h_ok
                        if "confirm4h_ok" not in locals():
                            confirm4h_ok = True
                            if CONFIRM4H:
                                try:
                                    df4 = ((feats or {}).get("4H") or {}).get("df")
                                    if df4 is not None and len(df4) >= 2:
                                        last4 = df4.iloc[-2]
                                        try:
                                            atr4 = float(last4.atr14 if "atr14" in df4.columns else last4.atr)
                                        except Exception:
                                            atr4 = None
                                        confirm4h_ok = _is_marubozu(last4, want_bearish=want_bearish, atr_val=atr4)
                                    else:
                                        confirm4h_ok = False
                                except Exception:
                                    confirm4h_ok = False
                        out["block"] = True if not CONFIRM4H else bool(confirm4h_ok)
                        out["why"] = "opp_marubozu_1H_partial" + ("" if out["block"] else ", 4H_no_confirm")
                        return out
            except Exception:
                # im lặng nếu không đủ dữ liệu của cột anatomy / atr
                pass
    except Exception:
        pass
    return out

def _apply_recent_1h_guard(bundle: Dict[str, Any], side: str, decision_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Nếu decision là ENTER thì chạy guard 1H 3 nến ngược chiều.
    Khớp → chuyển quyết định thành WAIT và gắn lý do vào meta/reasons/logs.
    """
    try:
        if (decision_dict or {}).get("decision") != "ENTER":
            return decision_dict
        feats = bundle.get("features_by_tf") or {}
        # Kiểm tra skip guard trong 1H regime
        meta = (decision_dict.get("meta") or {})
        exec_tf = str(meta.get("exec_tf_auto", "")) or ""
        if exec_tf == "1H":
            return decision_dict

        g = _guard_recent_1h_opposite(feats, side)
        if g.get("block"):
            decision_dict["decision"] = "WAIT"
            meta = dict(decision_dict.get("meta") or {})
            meta.update({"guard": "opp_candle_1h_recent", "guard_detail": g.get("why","")})
            decision_dict["meta"] = meta
            rs = list(decision_dict.get("reasons") or [])
            rs.append(f"guard_1h_recent_opposite: {g.get('why','')}")
            decision_dict["reasons"] = rs
            lg = dict(decision_dict.get("logs") or {})
            lg["WAIT"] = {"reasons": ["recent_1h_opposite"], "details": g.get("why","")}
            decision_dict["logs"] = lg
        return decision_dict
    except Exception:
        return decision_dict

def _guard_intraday_reversal_shock(feats: Dict[str, Any], side: str) -> Dict[str, Any]:
    """
    Chặn ENTER khi nến 1H vừa đóng có 'volatility shock' mang tính đảo chiều:
      - range nến 1H >= SHOCK_RANGE_ATR * ATR1H
      - và (bearish/bullish engulfing hoặc wick dài theo hướng đảo chiều hoặc ΔRSI mạnh)
      - và yếu thêm: close cắt qua EMA20 theo chiều bất lợi hoặc vol_ratio spike
    Có van an toàn cho continuation mạnh thật (marubozu/thân lớn + RSI tăng/giảm mạnh).
    ENV:
      SHOCK_RANGE_ATR(1.3) SHOCK_WICK_FRAC(0.6) SHOCK_VOLR(1.8) SHOCK_RSI_DROP(8)
      SHOCK_ALLOW_STRONG(1)
    """
    try:
        if side not in ("long","short"):
            return {"block": False, "why": ""}
        df1 = ((feats or {}).get("1H") or {}).get("df")
        if df1 is None or len(df1) < 3:
            return {"block": False, "why": ""}
        # Adaptive skip khi NATR thấp hoặc đang dùng 1H regime
        try:
            df4 = ((feats or {}).get("4H") or {}).get("df")
            atr = float(df4["atr14"].iloc[-2]) if "atr14" in df4.columns else None
            price = float(df4["close"].iloc[-2]) if "close" in df4.columns else None
            natr = (atr / price) * 100 if atr and price and price > 0 else 0
            meta_tf = str(((feats.get("meta") or {}).get("exec_tf_auto")) or "")
            if natr < 0.04 or meta_tf == "1H":
                return {"block": False, "why": "skip_guard_shock_lowvol_or_exec1H"}
        except Exception:
            pass
        last = _last_closed_bar(df1)
        prev = df1.iloc[-3] if len(df1) >= 3 else None
        if last is None or prev is None:
            return {"block": False, "why": ""}
        o, h, l, c = float(last["open"]), float(last["high"]), float(last["low"]), float(last["close"])
        atr1 = float(df1.loc[last.name, "atr14"]) if "atr14" in df1.columns else float("nan")
        e20  = float(last.get("ema20", float("nan")))
        e50  = float(last.get("ema50", float("nan"))) if "ema50" in last.index else float("nan")
        volr = float(last.get("vol_ratio", 1.0)) if "vol_ratio" in df1.columns else 1.0
        rsi  = None; rsi_prev = None
        for col in ("rsi14","rsi","RSI"):
            if col in df1.columns:
                rsi      = float(df1.loc[last.name, col])
                rsi_prev = float(prev[col])
                break
        rng = max(1e-12, h - l)
        up_wick   = max(0.0, h - max(o, c))
        down_wick = max(0.0, min(o, c) - l)
        body = abs(c - o)
        body_frac = body / rng if rng > 0 else 0.0
        # ENV thresholds
        import os as _os
        K_atr    = float(_os.getenv("SHOCK_RANGE_ATR",   "1.3"))
        K_wick   = float(_os.getenv("SHOCK_WICK_FRAC",   "0.6"))
        K_volr   = float(_os.getenv("SHOCK_VOLR",        "1.8"))
        K_rsidd  = float(_os.getenv("SHOCK_RSI_DROP",    "8"))
        ALLOW_ST = str(_os.getenv("SHOCK_ALLOW_STRONG",  "1")).lower() in ("1","true","yes")
        # patterns
        bearish_engulf = (c < o) and (float(prev["close"]) > float(prev["open"])) and (o >= float(prev["close"])) and (c <= float(prev["open"]))
        bull_engulf    = (c > o) and (float(prev["close"]) < float(prev["open"])) and (o <= float(prev["close"])) and (c >= float(prev["open"]))
        big_range      = (atr1 == atr1) and (rng >= K_atr * atr1)
        vol_spike      = (volr is not None) and (volr >= K_volr)
        rsi_drop       = (rsi is not None and rsi_prev is not None and (rsi_prev - rsi) >= K_rsidd)
        rsi_pop        = (rsi is not None and rsi_prev is not None and (rsi - rsi_prev) >= K_rsidd)
        marubozu_up    = (c > o) and (up_wick/rng <= 0.20) and (down_wick/rng <= 0.10)
        marubozu_down  = (c < o) and (down_wick/rng <= 0.20) and (up_wick/rng <= 0.10)
        if side == "long":
            wick_reversal = (rng > 0 and (up_wick / rng) >= K_wick)
            weak_filter   = ((e20 == e20) and (c < e20)) or vol_spike
            # Allow continuation mạnh thật
            if ALLOW_ST:
                strong_trend = (c > e20 > e50) and rsi_pop and (body_frac >= 0.55)
                if big_range and (marubozu_up or strong_trend):
                    return {"block": False, "why": "allow_strong_trend_after_shock"}
            if big_range and weak_filter and (bearish_engulf or wick_reversal or rsi_drop):
                return {"block": True, "why": f"1H shock-down: rng>={K_atr}*ATR, wick≥{K_wick}, volr≥{K_volr} or ΔRSI≤-{K_rsidd}"}
        else:
            wick_reversal = (rng > 0 and (down_wick / rng) >= K_wick)
            weak_filter   = ((e20 == e20) and (c > e20)) or vol_spike
            if ALLOW_ST:
                strong_trend = (c < e20 < e50) and rsi_drop and (body_frac >= 0.55)
                if big_range and (marubozu_down or strong_trend):
                    return {"block": False, "why": "allow_strong_trend_after_shock"}
            if big_range and weak_filter and (bull_engulf or wick_reversal or rsi_pop):
                return {"block": True, "why": f"1H shock-up: rng>={K_atr}*ATR, wick≥{K_wick}, volr≥{K_volr} or ΔRSI≥{K_rsidd}"}
    except Exception:
        pass
    return {"block": False, "why": ""}

def _atr_from_features_tf(features_by_tf: Dict[str, Any], tf: str = "4H") -> float:
    """Use ATR at the last *closed* bar to avoid partial-candle drift."""
    try:
        df = (features_by_tf or {}).get(tf, {}).get("df")
        if df is not None and len(df) > 0:
            last = _last_closed_bar(df)
            if last is not None:
                # read ATR value at the same index as the closed bar
                return float(df.loc[last.name, "atr14"])
    except Exception:
        pass
    return 0.0

def _regime_from_feats(features_by_tf: Dict[str, Any], tf: str = "4H") -> str:
    try:
        return str(((features_by_tf or {}).get(tf) or {}).get("meta", {}) or {}).get("regime","normal")
    except Exception:
        return "normal"

def _soft_levels_by_tf(features_by_tf: Dict[str, Any], tf: str = "4H") -> Dict[str, float]:
    """
    Lấy các mức mềm ở nến đã đóng gần nhất của TF chỉ định: BB upper/mid/lower, EMA20/50, Close.
    """
    out = {}
    try:
        df = (features_by_tf or {}).get(tf, {}).get("df")
        if df is not None and len(df) > 0:
            last = _last_closed_bar(df)
            if last is None:
                return out
            for k in ("bb_upper","bb_mid","bb_lower","ema20","ema50","close"):
                if k in last and last[k] == last[k]:  # not NaN
                    out[k] = float(last[k])
    except Exception:
        pass
    return out

def _rsi_from_features_tf(features_by_tf: Dict[str, Any], tf: str = "1H") -> Optional[float]:
    """
    Lấy RSI (mặc định cột rsi14) tại nến *đã đóng* gần nhất của TF cho trước.
    """
    try:
        df = (features_by_tf or {}).get(tf, {}).get("df")
        if df is not None and len(df) > 0:
            last = _last_closed_bar(df)
            if last is not None:
                for col in ("rsi14", "rsi", "RSI"):
                    if col in df.columns:
                        return float(df.loc[last.name, col])
    except Exception:
        pass
    return None

def _is_pullback_in_progress(feats: Dict[str, Any], side: str) -> tuple[bool, str]:
    """
    Phát hiện *pullback đang diễn tiến* để chặn continuation bắt dao rơi.
    Long: 4H còn nằm dưới BB-mid/EMA20 và EMA20<EMA50; 1H còn yếu (close<EMA50, RSI<52).
    Short: đối xứng ngược lại.
    """
    try:
        if side not in ("long","short"):
            return False, ""
        # 4H levels
        df4 = ((feats or {}).get("4H") or {}).get("df")
        if df4 is None or len(df4) < 3:
            return False, ""
        last4 = _last_closed_bar(df4)
        c4   = float(last4["close"])
        e20  = float(last4.get("ema20", float("nan")))
        e50  = float(last4.get("ema50", float("nan")))
        bmid = float(last4.get("bb_mid", float("nan")))
        # 1H momentum/structure
        df1 = ((feats or {}).get("1H") or {}).get("df")
        last1 = _last_closed_bar(df1) if (df1 is not None and len(df1)) else None
        rsi1 = None; c1=None; e50_1h=None; 
        if last1 is not None:
            c1 = float(last1.get("close", float("nan")))
            e50_1h = float(last1.get("ema50", float("nan"))) if "ema50" in last1.index else float("nan")
            for col in ("rsi14","rsi","RSI"):
                if col in df1.columns:
                    rsi1 = float(df1.loc[last1.name, col]); break
        # Long: pullback còn chạy nếu (c4 < bmid or c4 < e20) & (e20 < e50) & (c1<e50_1h or rsi1<52)
        if side == "long":
            cond4 = ((c4 < bmid) or (c4 < e20)) and (e20 == e20 and e50 == e50 and e20 < e50)
            cond1 = ((c1 is not None and e50_1h == e50_1h and c1 < e50_1h) or (rsi1 is not None and rsi1 < 52.0))
            if cond4 and cond1:
                return True, "4H below mid/ema20 & ema20<ema50; 1H weak (below ema50 / rsi<52)"
        # Short: pullback còn chạy nếu (c4 > bmid or c4 > e20) & (e20>e50) & (c1>e50_1h or rsi1>48)
        if side == "short":
            cond4 = ((c4 > bmid) or (c4 > e20)) and (e20 == e20 and e50 == e50 and e20 > e50)
            cond1 = ((c1 is not None and e50_1h == e50_1h and c1 > e50_1h) or (rsi1 is not None and rsi1 > 48.0))
            if cond4 and cond1:
                return True, "4H above mid/ema20 & ema20>ema50; 1H strong (above ema50 / rsi>48)"
    except Exception:
        pass
    return False, ""

def _momentum_grade_1h(features_by_tf: Dict[str, Any]) -> str:
    """
    Đánh giá nhanh động lượng 1H dựa trên RSI, BB width, và vol_ratio.
    - strong: rsi>=60, bb_width_pct tăng và vol_ratio>=1.2
    - weak:   rsi<=45 hoặc bb_width_pct rất hẹp
    - normal: còn lại
    (bb_width_pct & vol_ratio được enrich trong indicators.py)
    """
    try:
        df1 = ((features_by_tf or {}).get("1H") or {}).get("df")
        if df1 is None or len(df1) < 5:
            return "normal"
        rsi = float(df1["rsi14"].iloc[-2]) if "rsi14" in df1.columns else float("nan")
        vw  = float(df1["bb_width_pct"].iloc[-2]) if "bb_width_pct" in df1.columns else float("nan")
        vw_prev = float(df1["bb_width_pct"].iloc[-4]) if "bb_width_pct" in df1.columns and len(df1)>=4 else vw
        volr = float(df1["vol_ratio"].iloc[-2]) if "vol_ratio" in df1.columns else 1.0
        widening = (vw == vw) and (vw_prev == vw_prev) and (vw > vw_prev)
        if (rsi == rsi) and rsi >= 60.0 and widening and volr >= 1.2:
            return "strong"
        if (rsi == rsi) and rsi <= 45.0:
            return "weak"
        if (vw == vw) and vw < 1.0:
            return "weak"
        return "normal"
    except Exception:
        return "normal"

def _scale_out_weights_for_profile(profile: str) -> Dict[str, float]:
    """
    Trả về weights scale-out theo profile.
    Lưu ý: TP0 (nếu có) dùng weight riêng 'tp0_weight' đã set trong meta.
    Tổng (TP1..TP4 + TP0 nếu có) ≈ 1.0 là khuyến nghị; không ép buộc trong code.
    """
    p = (profile or "").strip().lower()
    if p == "1h-ladder":
        # Gợi ý: TP0=0.20 (đã set ở meta), TP1=0.20, TP2=0.25, TP3=0.20, TP4=0.15  ⇒ tổng ≈ 1.0
        return {"tp1": 0.20, "tp2": 0.25, "tp3": 0.20, "tp4": 0.15, "tp5": 0.0}
    if p == "momentum-back":
        # Động lượng mạnh → back-load (giữ phần lớn cho TP xa)
        return {"tp1": 0.10, "tp2": 0.15, "tp3": 0.25, "tp4": 0.25, "tp5": 0.25}
    if p == "range-front":
        # Sideways/range → front-load để hiện thực hóa sớm
        return {"tp1": 0.35, "tp2": 0.25, "tp3": 0.20, "tp4": 0.15, "tp5": 0.05}
    # fallback chung (không có TP0)
    return {"tp1": 0.30, "tp2": 0.30, "tp3": 0.20, "tp4": 0.20, "tp5": 0.0}
    
def _risk_unit(entry: float, sl: float, side: str) -> float:
    try:
        return (entry - sl) if side == "long" else (sl - entry)
    except Exception:
        return 0.0

def _entry_cushion_k_from_env(side: str, regime: str) -> float:
    """
    Thứ tự ưu tiên ENV:
      ENTRY_CUSHION_ATR_{SIDE} > ENTRY_CUSHION_ATR_{REGIME} > ENTRY_CUSHION_ATR
    """
    def _f(name, default=None):
        try:
            v = os.getenv(name)
            return float(v) if v is not None and v != "" else default
        except Exception:
            return default
    side_u = (side or "").upper()
    reg_u  = (regime or "normal").upper()
    for key in (f"ENTRY_CUSHION_ATR_{side_u}", f"ENTRY_CUSHION_ATR_{reg_u}", "ENTRY_CUSHION_ATR"):
        k = _f(key, None)
        if k is not None:
            return max(0.0, float(k))
    return 0.0

def _apply_entry_cushion(entry: Optional[float], side: Optional[str], atr4: float, k: float) -> Optional[float]:
    """LONG: entry += k*ATR ; SHORT: entry -= k*ATR"""
    try:
        if entry is None or side not in ("long","short") or atr4 <= 0 or k <= 0:
            return entry
        return float(entry + k*atr4) if side == "long" else float(entry - k*atr4)
    except Exception:
        return entry

def _momentum_grade_1h(features_by_tf: Dict[str, Any]) -> str:
    """
    Đánh giá nhanh động lượng 1H dựa trên RSI, BB width, và vol_ratio.
    - strong: rsi>=60, bb_width_pct tăng và vol_ratio>=1.2
    - weak:   rsi<=45 hoặc bb_width_pct rất hẹp
    - normal: còn lại
    (bb_width_pct & vol_ratio được enrich trong indicators.py)
    """
    try:
        df1 = ((features_by_tf or {}).get("1H") or {}).get("df")
        if df1 is None or len(df1) < 5:
            return "normal"
        rsi = float(df1["rsi14"].iloc[-2]) if "rsi14" in df1.columns else float("nan")
        vw  = float(df1["bb_width_pct"].iloc[-2]) if "bb_width_pct" in df1.columns else float("nan")
        vw_prev = float(df1["bb_width_pct"].iloc[-4]) if "bb_width_pct" in df1.columns and len(df1)>=4 else vw
        volr = float(df1["vol_ratio"].iloc[-2]) if "vol_ratio" in df1.columns else 1.0
        widening = (vw == vw) and (vw_prev == vw_prev) and (vw > vw_prev)
        if (rsi == rsi) and rsi >= 60.0 and widening and volr >= 1.2:
            return "strong"
        if (rsi == rsi) and rsi <= 45.0:
            return "weak"
        if (vw == vw) and vw < 1.0:
            return "weak"
        return "normal"
    except Exception:
        return "normal"

def _choose_scaleout_profile(features_by_tf: Dict[str, Any], side: Optional[str]) -> str:
    """
    Quy tắc chọn profile:
      - momentum-back: 1H momentum strong và 4H không ở 'high' regime
      - range-front:   1H momentum weak hoặc 4H regime='low'
      - 1h-ladder:     nếu meta/profile đã gợi ý trước đó
      - fallback:      balanced
    """
    try:
        regime4 = _regime_from_feats(features_by_tf, "4H")
        mom1 = _momentum_grade_1h(features_by_tf)
        if mom1 == "strong" and regime4 != "high":
            return "momentum-back"
        if mom1 == "weak" or regime4 == "low":
            return "range-front"
        # nếu core đã set profile (ví dụ 1h-ladder), giữ nguyên
        meta = ((features_by_tf or {}).get("4H") or {}).get("meta", {}) or {}
        prof = str(meta.get("profile") or "").strip().lower()
        if prof:
            return prof
        return "balanced"
    except Exception:
        return "balanced"

def _guard_near_bb_low_4h_and_rsi1h_extreme(
    side: Optional[str],
    entry: Optional[float],
    feats: Dict[str, Any],
    *,
    state: Optional[str] = None,
) -> Dict[str, Any]:
    """
    WAIT guard khi ở gần mép BB 4H và RSI(1H) cực trị ĐÚNG CHIỀU RỦI RO:
      - Long: gần BB-lower (<= 0.30 * ATR_4H) & RSI1H <= 20  → dễ rơi tiếp
      - Short: gần BB-upper (<= 0.30 * ATR_4H) & RSI1H >= 80 → dễ bật tiếp
    Tránh ENTER kể cả khi state=trend_break.
    """
    try:
        if side not in ("long","short") or entry is None:
            return {"block": False, "why": ""}
        # C: Skip proximity cho BREAK (đã có breakout/early logic ở core)
        if state == "trend_break":
            return {"block": False, "why": ""}
        atr4 = _atr_from_features_tf(feats, "4H")
        if atr4 <= 0:
            return {"block": False, "why": ""}
        lv = _soft_levels_by_tf(feats, "4H")
        bb_l = lv.get("bb_lower")
        bb_u = lv.get("bb_upper")
        if bb_l is None and bb_u is None:
            return {"block": False, "why": ""}
        rsi1 = _rsi_from_features_tf(feats, "1H")
        if rsi1 is None:
            return {"block": False, "why": ""}
        thr = 0.30 * atr4
        near_bb_low = (bb_l is not None) and (entry >= bb_l) and (abs(entry - bb_l) <= thr)
        near_bb_up  = (bb_u is not None) and (entry <= bb_u) and (abs(entry - bb_u) <= thr)
        if side == "long" and near_bb_low and (rsi1 <= 20.0):
            return {"block": True, "why": f"long@near_bb_lower±{thr:.4f} & RSI1H={rsi1:.1f}"}
        if side == "short" and near_bb_up and (rsi1 >= 80.0):
            return {"block": True, "why": f"short@near_bb_upper±{thr:.4f} & RSI1H={rsi1:.1f}"}
    except Exception:
        pass
    return {"block": False, "why": ""}

# ============================================================
# Guard bổ sung: chặn continuation khi thị trường sideway
# ============================================================
def _guard_sideway_regime(
    features_by_tf: Dict[str, Any],
    side: Optional[str] = None,
    *,
    state: Optional[str] = None,
) -> Dict[str, Any]:
    """
    WAIT guard khi thị trường 4H ở regime='low' hoặc BB width hẹp.
    - Mục đích: tránh continuation trong range, dễ bị gập giá ngược.
    - Áp dụng cho cả long và short.
    """
    try:
        # Bỏ qua nếu chưa có features
        df4 = (features_by_tf or {}).get("4H", {}).get("df")
        meta4 = ((features_by_tf or {}).get("4H") or {}).get("meta", {}) or {}
        if df4 is None or len(df4) < 5:
            return {"block": False, "why": ""}

        # Đọc BB width % và regime
        bbw = float(df4["bb_width_pct"].iloc[-2]) if "bb_width_pct" in df4.columns else float("nan")
        regime = str(meta4.get("regime", "normal")).lower()

        # Nếu BB hẹp hoặc regime low → block
        if (regime == "low") or (bbw == bbw and bbw < 1.2):
            return {
                "block": True,
                "why": f"sideway regime (regime={regime}, BBW={bbw:.2f}%)"
            }
    except Exception:
        pass
    return {"block": False, "why": ""}
    
def _near_soft_level_guard_multi(
    side: Optional[str],
    entry: Optional[float],
    feats: Dict[str, Any],
    tfs: Iterable[str] = ("4H",),
    *,
    state: Optional[str] = None,
    rr_ok: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Trả về {"block": bool, "why": str} nếu entry quá gần BB/EMA (1H/4H…) theo hướng giao dịch.
    Nới cho crypto:
      - Bỏ guard cho state RETEST (ôm mid/EMA là hợp lệ).
      - Siết ngưỡng cấm: band 0.25*ATR, center 0.20*ATR.
      - Không block nếu RR1 đã đủ (>= 1.0).
    """
    if side not in ("long","short") or entry is None:
        return {"block": False, "why": ""}
    # Skip proximity cho RETEST (pullback/throwback cần gần mid/EMA)
    if state in ("retest_support", "retest_resistance"):
        return {"block": False, "why": ""}
    reasons: List[str] = []
    for tf in tfs:
        atr = _atr_from_features_tf(feats, tf)
        if atr <= 0:
            continue
        lv = _soft_levels_by_tf(feats, tf)
        if not lv:
            continue
        bb_u, bb_m, bb_l = lv.get("bb_upper"), lv.get("bb_mid"), lv.get("bb_lower")
        e20, e50 = lv.get("ema20"), lv.get("ema50")
        # regime-adaptive thresholds
        try:
            regime = (feats.get('4H', {}).get('meta', {}) or {}).get('regime', 'normal')
        except Exception:
            regime = 'normal'
        if regime == 'high':
            thr_band, thr_center = 0.40 * atr, 0.35 * atr
        elif regime == 'low':
            thr_band, thr_center = 0.30 * atr, 0.25 * atr
        else:
            thr_band, thr_center = 0.35 * atr, 0.30 * atr
        def _dist(a, b):
            try: return abs(float(a) - float(b))
            except Exception: return float("inf")
        if side == "long":
            if bb_u is not None and entry <= bb_u and _dist(entry, bb_u) <= thr_band:
                reasons.append(f"{tf}:near_BB_upper(<= {thr_band:.4f})")
            for nm, lvl in (("EMA20", e20), ("EMA50", e50), ("BB_mid", bb_m)):
                if lvl is not None and entry >= lvl and _dist(entry, lvl) <= thr_center:
                    reasons.append(f"{tf}:near_{nm}(<= {thr_center:.4f})")
        else:
            if bb_l is not None and entry >= bb_l and _dist(entry, bb_l) <= thr_band:
                reasons.append(f"{tf}:near_BB_lower(<= {thr_band:.4f})")
            for nm, lvl in (("EMA20", e20), ("EMA50", e50), ("BB_mid", bb_m)):
                if lvl is not None and entry <= lvl and _dist(entry, lvl) <= thr_center:
                    reasons.append(f"{tf}:near_{nm}(<= {thr_center:.4f})")
    # Nếu RR đủ gần (rr_ok) >= 1.0 thì không block dù gần soft-level
    if reasons and not (isinstance(rr_ok, (int, float)) and rr_ok >= 1.0):
        return {"block": True, "why": ";".join(reasons)}
    return {"block": False, "why": ""}

def _apply_1h_ladder(decision: Dict[str, Any], cfg: SideCfg) -> Dict[str, Any]:
    """
    Post-build transform: khi meta.profile == '1H-ladder' thì:
    - SL min-gap dựa ATR(1H) (cạn hơn) nhưng vẫn chắc.
    - TPs dùng rr_targets_1h + chèn TP0 ở +0.35R (mặc định).
    - Gắn meta để main.py biết kích hoạt BE Turbo sớm.
    """
    try:
        meta = decision.get("meta", {}) or {}
        setup = decision.get("setup", {}) or {}
        side  = (decision.get("side") or "").lower()
        if meta.get("profile") != "1H-ladder":
            return decision
        entry = float(setup.get("entry"))
        sl    = float(setup.get("sl"))
        if side not in ("long","short") or not (entry == entry) or not (sl == sl):
            return decision
        feats = meta.get("features_by_tf") or {}
        atr1h = _atr_from_features_tf(feats, "1H")
        atr4h = _atr_from_features_tf(feats, "4H")
        # đảm bảo có ATR (fallback 4H nếu 1H thiếu)
        atr = atr1h if atr1h > 0 else atr4h
        if atr <= 0:
            return decision
        # Bảo đảm SL gap tối thiểu theo regime (xấp xỉ logic core)
        regime = str(meta.get("regime", "normal"))
        if regime == "low":
            min_atr = getattr(cfg, "sl_min_atr_low", 0.60)
        elif regime == "high":
            min_atr = getattr(cfg, "sl_min_atr_high", 1.20)
        else:
            min_atr = getattr(cfg, "sl_min_atr_normal", 0.80)
        min_gap = max(1e-9, min_atr * atr)  # ATR(1H) ưu tiên
        if side == "long" and (entry - sl) < min_gap:
            sl = entry - min_gap
        elif side == "short" and (sl - entry) < min_gap:
            sl = entry + min_gap
        # RR ladder 1H
        r = _risk_unit(entry, sl, side)
        if r <= 0:
            return decision
        rr_targets_1h = tuple(getattr(cfg, "rr_targets_1h", (0.8, 1.3, 1.8)))
        tps = []
        for rr in rr_targets_1h:
            tp = entry + rr * r if side == "long" else entry - rr * r
            tps.append(float(tp))
        # TP0 ~ +0.35R
        tp0_frac = float(getattr(cfg, "tp0_frac", 0.35))
        tp0 = entry + tp0_frac * r if side == "long" else entry - tp0_frac * r
        tps = [tp0] + tps
        # cập nhật setup & meta
        meta["ladder_tf"] = "1H"
        meta["tp0_weight"] = float(getattr(cfg, "tp0_weight", 0.20))
        meta["be_turbo_profile"] = "1H-ladder"
        decision["meta"] = meta
        decision["setup"] = {"entry": entry, "sl": sl, "tps": tps}
    except Exception:
        pass
    return decision

def _rr_snapshot(entry: float, sl: float, side: str, tps: List[float]) -> Dict[str, float]:
    """Tính RR tới các TP hiện có để guard có thể 'bỏ qua' nếu RR đã đủ."""
    out = {}
    try:
        risk = abs(entry - sl)
        if risk <= 0: 
            return out
        for i, tp in enumerate(tps, start=0):
            # tp[0] có thể là TP0 — vẫn tính RR để dùng cho rr_ok
            rr = (tp - entry) / risk if side == "long" else (entry - tp) / risk
            out[f"tp{i}"] = float(rr)
    except Exception:
        pass
    return out

def _apply_proximity_guard(decision: Dict[str, Any]) -> Dict[str, Any]:
    """
    Chặn ENTER nếu entry quá gần soft-level/band ở 1H hoặc 4H (headroom kém),
    trừ khi RR tới TP2 đã đủ tốt (rr_ok ≥ 1.5, có thể chỉnh).
    """
    try:
        if (decision.get("decision") or "").upper() != "ENTER":
            return decision
        setup = decision.get("setup") or {}
        meta  = decision.get("meta") or {}
        entry = float(setup.get("entry"))
        sl    = float(setup.get("sl"))
        side  = (decision.get("side") or meta.get("side") or "").lower()
        if side not in ("long","short") or not (entry == entry) or not (sl == sl):
            return decision
        tps = list(setup.get("tps") or [])
        feats = meta.get("features_by_tf") or {}
        # RR tới TP2 (hoặc TP1 nếu thiếu) làm "cửa thoát"
        rrs = _rr_snapshot(entry, sl, side, tps)
        rr_ok = float(rrs.get("tp2") or rrs.get("tp1") or 0.0)
        # gọi guard đa-TF (1H + 4H)
        g = _near_soft_level_guard_multi(
            features_by_tf=feats,
            tfs=("1H", "4H"),
            entry=entry,
            side=side,
            rr_ok=rr_ok
        )
        if g.get("block"):
            # chuyển sang WAIT, nêu lý do
            new_logs = decision.get("logs") or {}
            new_logs["WAIT"] = {"reasons": ["near_soft_level"], "details": g.get("why")}
            decision.update({
                "decision": "WAIT",
                "logs": new_logs
            })
    except Exception:
        pass
    return decision

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

def _leverage_hint(side: Optional[str], entry: Optional[float], sl: Optional[float]) -> Optional[float]:
    """
    Leverage tối ưu để rủi ro thực ~ risk_pct (ENV RISK_PCT, mặc định 5%).
    """
    try:
        if side not in ("long","short") or entry is None or sl is None or entry <= 0:
            return None
        risk_raw = abs((entry - sl) / entry)
        if risk_raw <= 0:
            return None
        risk_pct = float(os.getenv("RISK_PCT", "0.05"))
        import math
        # Cốt lõi: tính leverage sao cho risk thực = risk_pct
        lev = risk_pct / max(1e-9, risk_raw)

        # Điều chỉnh theo ATR regime (nếu được truyền qua ENV hoặc meta)
        natr = float(os.getenv("NATR_PCT", "0"))
        if natr and natr < 2.0:
            lev *= 1.3   # ATR thấp: cho phép leverage cao hơn
        elif natr > 5.0:
            lev *= 0.8   # ATR cao: giảm leverage để an toàn

        # Clamp trong khung 1–8x và làm tròn xuống số nguyên
        lev_min = float(os.getenv("LEVERAGE_MIN", "1.0"))
        lev_max = float(os.getenv("LEVERAGE_MAX", "8.0"))
        lev = max(lev_min, min(lev, lev_max))
        lev_int = math.floor(lev)

        # Tính risk thực tế sau khi làm tròn
        real_risk_pct = risk_raw * lev_int
        os.environ["RISK_REAL_PCT"] = f"{real_risk_pct:.4f}"

        return float(lev_int)
    except Exception:
        return None

def _sanitize_meta_for_logs(meta_in) -> Dict[str, Any]:
    """
    Loại bỏ/thu gọn các trường không JSON-serializable (đặc biệt là DataFrame)
    trước khi đưa meta vào logs/JSON.
    - Bỏ hẳn 'features_by_tf'
    - Nếu gặp DataFrame ở mức 1, thay bằng string mô tả
    - Nếu gặp dict có key 'df', set 'df'=None để tránh serialize DataFrame
    """
    try:
        if not isinstance(meta_in, dict):
            return {}
        meta_out: Dict[str, Any] = {}
        # Bỏ hẳn features_by_tf (chứa df)
        items = {k: v for k, v in meta_in.items() if k != "features_by_tf"}
        for k, v in items.items():
            # xử lý dict có 'df'
            if isinstance(v, dict) and ("df" in v):
                vv = dict(v)
                try:
                    # nếu vv["df"] là DataFrame → None
                    import pandas as _pd  # lazy import chỉ để isinstance check
                    if isinstance(vv.get("df"), _pd.DataFrame):
                        vv["df"] = None
                except Exception:
                    vv["df"] = None
                meta_out[k] = vv
                continue
            # thay DataFrame mức 1 bằng mô tả string
            try:
                import pandas as _pd
                if isinstance(v, _pd.DataFrame):
                    meta_out[k] = f"DataFrame(shape={v.shape})"
                    continue
            except Exception:
                pass
            meta_out[k] = v
        return meta_out
    except Exception:
        return {}

def decide(symbol: str, timeframe: str, features_by_tf: Dict[str, Dict[str, Any]], evidence_bundle: Dict[str, Any]) -> Dict[str, Any]:
    # evidence_bundle expected to include 'evidence' object; pass through as eb-like
    eb = evidence_bundle.get("evidence") or evidence_bundle  # tolerate both shapes
    cfg = SideCfg()
    # ===== ATR-based TP ladder via ENV =====
    # ATR_TP_MODE: "1"/"true"/"True" để bật
    # ATR_TP_BASE: "1.0,1.5,2.5" (ít nhất 3 số) nếu muốn đổi base
    try:
        if str(os.getenv("ATR_TP_MODE", "0")).strip() in ("1", "true", "True"):
            cfg.atr_tp_mode = True
        _base = str(os.getenv("ATR_TP_BASE", "1.0,1.5,2.5")).strip()
        parts = [float(x) for x in _base.split(",") if x.strip()]
        if len(parts) >= 3:
            cfg.atr_tp_base = (parts[0], parts[1], parts[2])
    except Exception:
        pass
    dec = run_side_state_core(features_by_tf, eb, cfg)

    # -------- REVERSAL GUARD (filter before release) --------
    # Dùng style (side, df4, df1) để chắc chắn dùng đúng DataFrame đã đóng nến.
    try:
        df4 = (features_by_tf or {}).get("4H", {}).get("df")
        df1 = (features_by_tf or {}).get("1H", {}).get("df")
        if dec.side in ("long", "short") and (dec.state or "").lower() != "reversal":
            is_rev, why_rev = _reversal_signal(dec.side.upper(), df4, df1)
            if is_rev:
                dec.decision = "WAIT"
                reasons = list(dec.reasons or [])
                # ghi rõ lý do để trace trên log
                reasons.append("guard:reversal")
                if why_rev:
                    reasons.append(f"rev:{why_rev}")
                dec.reasons = reasons
    except Exception as e:
        # Không chặn nếu check reversal lỗi, chỉ log warning
        import logging
        logging.getLogger(__name__).warning(f"Reversal guard check failed for {symbol}: {e}")

    # -------- [NEW] GUARD: pullback-in-progress (tránh bắt dao rơi) --------
    # Nếu 4H còn đang pullback (đúng mẫu) thì không cho ENTER continuation/break trong chiều ngược pullback.
    try:
        if (dec.decision or "WAIT").upper() == "ENTER" and dec.side in ("long","short"):
            if str(dec.state or "").lower() in ("trend_break", "continuation"):
                _pb_block, _pb_why = _is_pullback_in_progress(features_by_tf, dec.side)
                if _pb_block:
                    dec.decision = "WAIT"
                    rs = list(dec.reasons or [])
                    rs.append("guard:pullback_in_progress")
                    if _pb_why:
                        rs.append(f"pb:{_pb_why}")
                    dec.reasons = rs
    except Exception:
        pass

    # -------- [NEW] GUARD: continuation cần 1H momentum không 'weak' --------
    try:
        if (dec.decision or "WAIT").upper() == "ENTER" and str(dec.state or "").lower() == "continuation":
            if _momentum_grade_1h(features_by_tf) == "weak":
                dec.decision = "WAIT"
                rs = list(dec.reasons or [])
                rs.append("guard:weak_1h_momentum")
                dec.reasons = rs
    except Exception:
        pass

    # ===== [NEW] HẬU XỬ LÝ PROFILE & PROXIMITY (dựa trên dec/setup hiện có) =====
    # Chuyển 'dec' về dict tạm để dùng lại các helper _apply_1h_ladder / _apply_proximity_guard
    try:
        tmp_decision = {
            "decision": (dec.decision or "WAIT"),
            "side": dec.side,
            "state": dec.state,
            "meta": {
                # truyền features để guard 1H+4H tra cứu BB/EMA/ATR
                "features_by_tf": features_by_tf,
                # giữ các thông tin có ích nếu core đã gắn
                "regime": ((dec.meta or {}).get("regime") if isinstance(dec.meta, dict) else None),
                "profile": ((dec.meta or {}).get("profile") if isinstance(dec.meta, dict) else None),
            },
            "setup": {
                "entry": dec.setup.entry,
                "sl": dec.setup.sl,
                "tps": list(dec.setup.tps or []),
            },
            # logs để gom lý do nếu guard chuyển sang WAIT
            "logs": {},
        }
        # 1) Áp ladder 1H nếu meta.profile='1H-ladder' (chỉnh SL/TP + TP0; gắn tp0_weight/weights trong meta)
        tmp_decision = _apply_1h_ladder(tmp_decision, cfg)
        # 1.1) ENTRY CUSHION THEO ATR_4H (áp trước proximity để guard dùng entry đã cushion)
        try:
            _side   = (tmp_decision.get("side") or "").lower()
            _entry  = tmp_decision.get("setup", {}).get("entry")
            _atr4   = _atr_from_features_tf(features_by_tf, "4H")
            _reg4   = _regime_from_feats(features_by_tf, "4H")
            _k      = _entry_cushion_k_from_env(_side, _reg4)
            _entry2 = _apply_entry_cushion(_entry, _side, _atr4, _k)
            if _entry2 is not None and _entry2 != _entry:
                tmp_decision["setup"]["entry"] = float(_entry2)
                # ghi reason để trace
                _logs = tmp_decision.get("logs") or {}
                _logs["CUSHION"] = {"k_atr": float(_k), "atr4": float(_atr4)}
                tmp_decision["logs"] = _logs
        except Exception:
            pass
        # 2) Guard proximity 1H+4H ngay trước khi phát hành; rr_ok được tính nội bộ theo TP list
        tmp_decision = _apply_proximity_guard(tmp_decision)
        # 2.1) Guard intraday reversal shock (đảo chiều 1H kiểu FET)
        try:
            _side  = (tmp_decision.get("side") or "").lower()
            _feats = (tmp_decision.get("meta") or {}).get("features_by_tf") or {}
            _g = _guard_intraday_reversal_shock(_feats, _side)
            if _g.get("block"):
                _logs = tmp_decision.get("logs") or {}
                _logs["WAIT"] = {"reasons": ["intraday_reversal_shock"], "details": _g.get("why")}
                tmp_decision["logs"] = _logs
                tmp_decision["decision"] = "WAIT"
        except Exception:
            pass
        # 2.2) Guard 1H: 3 nến gần nhất có engulfing/marubozu ngược chiều (tùy chọn xác nhận 4H)
        try:
            _side  = (tmp_decision.get("side") or "").lower()
            if _side in ("long","short"):
                tmp_decision = _apply_recent_1h_guard(
                    {"features_by_tf": features_by_tf},
                    _side,
                    tmp_decision
                )
        except Exception:
            pass
        # Ghi ngược lại về object 'dec'
        dec.decision = tmp_decision.get("decision", dec.decision)
        # cập nhật setup nếu có thay đổi từ 1H-ladder
        try:
            _st = tmp_decision.get("setup") or {}
            if _st:
                if _st.get("entry") is not None: dec.setup.entry = float(_st["entry"])
                if _st.get("sl")    is not None: dec.setup.sl    = float(_st["sl"])
                if isinstance(_st.get("tps"), list) and _st["tps"]:
                    dec.setup.tps = list(_st["tps"])
        except Exception:
            pass
        # bổ sung lý do nếu guard block
        if (tmp_decision.get("decision") or "").upper() != "ENTER":
            rs = list(dec.reasons or [])
            wlog = (tmp_decision.get("logs") or {}).get("WAIT") or {}
            det = wlog.get("details") or ""
            # Gộp toàn bộ reasons từ guard hậu xử lý
            for r in (wlog.get("reasons") or []):
                if r not in rs:
                    rs.append(r)
            # Giữ tương thích cũ: thêm soft_proximity nếu có proximity guard
            if "near_soft_level" in (wlog.get("reasons") or []) and "soft_proximity" not in rs:
                rs.append("soft_proximity")
            if det:
                rs.append(det)
            dec.reasons = rs
        # merge meta nhưng KHÔNG giữ features_by_tf (có DataFrame)
        try:
            _m = dict(tmp_decision.get("meta") or {})
            _m.pop("features_by_tf", None)
            if isinstance(dec.meta, dict):
                dec.meta.update(_m)
            else:
                dec.meta = _m
        except Exception:
            pass
    except Exception:
        # nếu có lỗi, bỏ qua hậu xử lý
        pass

    # Build plan (legacy fields)
    tps = dec.setup.tps or []
    tp1 = tps[0] if len(tps) > 0 else None
    tp2 = tps[1] if len(tps) > 1 else None
    tp3 = tps[2] if len(tps) > 2 else None

    # RR calculations
    # -------- Map original 3 TP -> 5 TP ladder --------
    # Ensure we have 3 base TP levels; if only 2 provided, synthesize mid as TP2
    def _mid(a,b):
        try: return (float(a)+float(b))/2.0
        except Exception: return None
    if tp3 is None and (tp1 is not None) and (tp2 is not None):
        _tp2_mid = _mid(tp1, tp2)
        if _tp2_mid is not None:
            tp3 = tp2
            tp2 = _tp2_mid
    # Expand to 5 levels: [mid(entry,tp1), tp1, tp2, mid(tp2,tp3), tp3]
    new_tp1 = _mid(dec.setup.entry, tp1) if (dec.setup.entry is not None and tp1 is not None) else tp1
    new_tp2 = tp1
    new_tp3 = tp2
    new_tp4 = _mid(tp2, tp3) if (tp2 is not None and tp3 is not None) else None
    new_tp5 = tp3
    tp1, tp2, tp3, tp4, tp5 = new_tp1, new_tp2, new_tp3, new_tp4, new_tp5
    rr1 = _rr(dec.setup.entry, dec.setup.sl, tp1, dec.side)
    rr2 = _rr(dec.setup.entry, dec.setup.sl, tp2, dec.side)
    rr3 = _rr(dec.setup.entry, dec.setup.sl, tp3, dec.side)
    rr4 = _rr(dec.setup.entry, dec.setup.sl, tp4, dec.side) if tp4 is not None else None
    rr5 = _rr(dec.setup.entry, dec.setup.sl, tp5, dec.side) if tp5 is not None else None

    # -------- SL risk guard (> 5% entry) --------
    try:
        thr = float(os.getenv("SL_MAX_RISK_PCT", "0.05"))
    except Exception:
        thr = 0.05
    try:
        if dec.setup.entry and dec.setup.sl and dec.side in ("long", "short"):
            risk_pct = abs(float(dec.setup.entry) - float(dec.setup.sl)) / max(float(dec.setup.entry), 1e-9)
            if risk_pct > thr:
                dec.decision = "WAIT"
                reasons = list(dec.reasons or [])
                if f"sl_risk>{thr:.3f}" not in reasons:
                    reasons.append(f"sl_risk>{thr:.3f}")
                dec.reasons = reasons
    except Exception:
        pass

    # -------- SOFT PROXIMITY GUARD (BB/EMA) --------
    # Truyền state & rr1 để nới hợp lý theo ngữ cảnh
    # Với 5TP: rr1 là mid(entry,tp1-old); rr2 mới là TP1-old.
    # Dùng rr_ok = max(rr1, rr2) để mềm hợp lý hơn.
    # C: dùng rr2/rr3 (TP1/TP2 sau expand) để đại diện tốt hơn cho setup 5TP
    _rr_ok_candidates = [x for x in (rr2, rr3) if isinstance(x,(int,float))]
    rr_ok = max(_rr_ok_candidates) if _rr_ok_candidates else None
    prox = _near_soft_level_guard_multi(
        dec.side, dec.setup.entry, features_by_tf,
        state=dec.state, rr_ok=rr_ok
    )
    if prox.get("block"):
        # Ép về WAIT + thêm lý do "soft_proximity"
        dec.decision = "WAIT"
        reasons = list(dec.reasons or [])
        reasons.append("soft_proximity")
        dec.reasons = reasons
        # Không đổi setup; chỉ cấm vào kèo lúc này

    # -------- BB-low(4H) + RSI(1H) extreme guard --------
    bb_rsi_guard = _guard_near_bb_low_4h_and_rsi1h_extreme(
        dec.side, dec.setup.entry, features_by_tf, state=dec.state
    )
    if bb_rsi_guard.get("block"):
        dec.decision = "WAIT"
        reasons = list(dec.reasons or [])
        reasons.append("guard:near_4h_bb_low_and_rsi1h_os")
        dec.reasons = reasons

    # -------- SIDEWAY regime guard (low volatility / narrow BB) --------
    try:
        sideway_guard = _guard_sideway_regime(features_by_tf, side=dec.side, state=dec.state)
        if sideway_guard.get("block"):
            dec.decision = "WAIT"
            reasons = list(dec.reasons or [])
            reasons.append("guard:sideway_regime")
            dec.reasons = reasons
            # Thêm chi tiết để dễ trace log
            if isinstance(dec.meta, dict):
                dec.meta["sideway_guard_why"] = sideway_guard.get("why")
    except Exception:
        pass

    # -------- RR floors sau khi mở rộng 3TP -> 5TP --------
    # Mapping:
    #   - TP2 (cũ) => TP3 (mới)  → dùng "RR2_FLOOR"
    #   - TP3 (cũ) => TP5 (mới)  → dùng "RR3_FLOOR"
    rr2_base = float(os.getenv("RR2_FLOOR", "1.30"))  # floor cho TP2 (cũ) -> TP3 (mới)
    rr3_base = float(os.getenv("RR3_FLOOR", "1.80"))  # floor cho TP3 (cũ) -> TP5 (mới)
    regime = (dec.meta or {}).get("regime", "normal") if isinstance(dec.meta, dict) else "normal"
    if regime == "high":
        rr_tp3_floor, rr_tp5_floor = 1.10, 1.60
    elif regime == "normal":
        rr_tp3_floor, rr_tp5_floor = 1.20, 1.70
    else:
        rr_tp3_floor, rr_tp5_floor = rr2_base, rr3_base

    def _suggest_entry2_for_floor(side: str, sl: float, tp: float, floor: float, cur_entry: float) -> Optional[float]:
        try:
            if side == "long":
                # (tp - e2) / (e2 - sl) >= floor  =>  e2 <= (tp + floor*sl) / (1 + floor)
                e2 = (tp + floor * sl) / (1.0 + floor)
                return float(e2) if e2 > 0 else None
            elif side == "short":
                # (e2 - tp) / (sl - e2) >= floor  =>  e2 >= (floor*sl + tp) / (1 + floor)
                e2 = (floor * sl + tp) / (1.0 + floor)
                return float(e2) if e2 > 0 else None
        except Exception:
            return None
        return None

    rr_floor_hit = False
    suggest_from = None  # ("TP3"/"TP5", tp_value, floor_value)
    # Sau expand: TP3 (mới) = TP2 (cũ), TP5 (mới) = TP3 (cũ)
    if tp3 is not None and rr3 is not None and rr3 < rr_tp3_floor:
        rr_floor_hit = True
        suggest_from = ("TP3", tp3, rr_tp3_floor)
    if tp5 is not None and rr5 is not None and rr5 < rr_tp5_floor:
        rr_floor_hit = True
        # nếu cả 2 dưới sàn, ưu tiên ràng buộc nghiêm hơn (TP5 xa hơn)
        suggest_from = ("TP5", tp5, rr_tp5_floor)

    if rr_floor_hit and dec.side in ("long","short") and dec.setup.entry is not None and dec.setup.sl is not None:
        # Soft rule sau expand:
        #   Nếu RR1>=0.8 và (RR@TP3 < sàn) nhưng (RR@TP5 >= sàn) => vẫn ENTER, chỉ log cảnh báo
        # Cho phép "allow_soft" nếu TP5 đạt floor dù TP3 chưa đạt; RR1 tối thiểu tuỳ regime
        allow_rr1_min = 0.4 if regime == "high" else 0.5
        allow_soft = (
            (rr1 is not None and rr1 >= allow_rr1_min) and
            (tp3 is not None and rr3 is not None and rr3 < rr_tp3_floor) and
            (tp5 is not None and rr5 is not None and rr5 >= rr_tp5_floor)
        )
        if not allow_soft:
            dec.decision = "WAIT"
        reasons = list(dec.reasons or [])
        if "rr_floor" not in reasons:
            reasons.append("rr_floor")
        dec.reasons = reasons
        _tpname, _tpval, _floor = suggest_from
        e2 = _suggest_entry2_for_floor(dec.side, dec.setup.sl, float(_tpval), float(_floor), dec.setup.entry)
        if e2 is not None:
            # long: entry2 thấp hơn; short: entry2 cao hơn
            dec.setup.entry2 = float(e2)
        # Log chi tiết RR sau mapping mới (TP3/TP5)
        try:
            _rr1 = f"{rr1:.2f}" if rr1 is not None else "nan"
            _rr3 = f"{rr3:.2f}" if rr3 is not None else "nan"  # TP3 (mới)
            _rr5 = f"{rr5:.2f}" if rr5 is not None else "nan"  # TP5 (mới)
            _gline = f"rr_floor_map(3/5TP): rr1={_rr1} rr@TP3={_rr3}/{rr_tp3_floor:.2f} rr@TP5={_rr5}/{rr_tp5_floor:.2f} allow_soft={allow_soft} regime={regime}"
        except Exception:
            pass

    # ---------- price formatting helpers ----------
    def _infer_dp(symbol: str, price: Optional[float], features_by_tf: Dict[str, Any], evidence_bundle: Dict[str, Any]) -> int:
        """
        Ưu tiên:
        1) meta.price_dp / meta.tick_size -> dp
        2) Heuristic theo giá (crypto)
        3) VN stock (không có '/') -> 0 lẻ
        """
        # 1) từ features/meta nếu có
        try:
            meta = (features_by_tf or {}).get("1H", {}).get("meta", {}) or {}
            dp = meta.get("price_dp")
            if isinstance(dp, int) and 0 <= dp <= 8:
                return dp
            tick = meta.get("tick_size") or evidence_bundle.get("meta", {}).get("tick_size")
            if tick:
                s = f"{tick}"
                if "." in s:
                    return min(8, max(0, len(s.split(".")[1].rstrip("0"))))
                # tick là số nguyên -> 0 lẻ
                return 0
        except Exception:
            pass
        # 2) Heuristic theo giá (crypto)
        if "/" in symbol:
            p = float(price or evidence_bundle.get("last_price") or 0.0)
            if p >= 1000: return 1
            if p >= 100:  return 2
            if p >= 1:    return 3
            if p >= 0.1:  return 4
            if p >= 0.01: return 5
            return 6
        # 3) VN stock (mã không có '/'): 0 lẻ (VND)
        return 0

    def _fmt(x: Optional[float], dp: int) -> Optional[str]:
        if x is None:
            return None
        try:
            return f"{float(x):.{dp}f}"
        except Exception:
            return f"{x}"

    # size hint (leverage) theo công thức risk_pct / risk_raw
    size_hint = _leverage_hint(dec.side, dec.setup.entry, dec.setup.sl)

    # ---------- end helpers ----------

    # === Chọn playbook scale-out theo động lượng & gắn weights vào plan ===
    profile = _choose_scaleout_profile(features_by_tf, dec.side)
    scale_weights = _scale_out_weights_for_profile(profile)

    plan = {
        "direction": dec.side.upper() if dec.side else None,
        "entry": dec.setup.entry,
        "entry2": None,               # kept for compatibility; tiny core emits single entry
        "sl": dec.setup.sl,
        "tp": tp1,                    # fallback single TP
        "tp1": tp1,
        "tp2": tp2,
        "tp3": tp3,
        "tp4": tp4,
        "tp5": tp5,
        "rr": rr1,                    # primary RR
        "rr2": rr2,
        "rr3": rr3,
        "rr4": rr4,
        "rr5": rr5,
        "risk_size_hint": size_hint,  # <— leverage đề xuất
        "profile": profile,
        "scale_out_weights": scale_weights,
    }

    # --- [NEW] scale-out theo động lượng (profile & weights) ---
    try:
        # Ưu tiên profile có sẵn trong meta/core; nếu chưa có thì chọn theo động lượng 1H & regime 4H
        _meta_in = dec.meta if isinstance(dec.meta, dict) else {}
        _profile = str((_meta_in.get("profile") or "")).strip().lower()
        if not _profile:
            _profile = _choose_scaleout_profile(features_by_tf, dec.side)
        weights = _scale_out_weights_for_profile(_profile)
        # TP0 weight (nếu ladder 1H chèn TP0)
        tp0_w = None
        try:
            if isinstance(_meta_in.get("tp0_weight"), (int, float)):
                tp0_w = float(_meta_in["tp0_weight"])
        except Exception:
            tp0_w = None
        # Lưu xuống plan/meta để main.py & storage.py sử dụng
        plan["scale_out_weights"] = weights
        if tp0_w is not None:
            plan["tp0_weight"] = tp0_w
        plan["profile"] = _profile
        # merge vào dec.meta (loại field nặng trước khi log)
        if isinstance(dec.meta, dict):
            dec.meta["profile"] = _profile
            dec.meta["scale_out_weights"] = dict(weights)
            if tp0_w is not None:
                dec.meta["tp0_weight"] = tp0_w
        else:
            dec.meta = {"profile": _profile, "scale_out_weights": dict(weights), **({"tp0_weight": tp0_w} if tp0_w is not None else {})}
    except Exception:
        pass

    # ---- ensure locals before logging ----
    decision = dec.decision or "WAIT"
    state = dec.state
    confidence = 0.0
    try:
        if isinstance(dec.meta, dict):
            confidence = float(dec.meta.get("confidence", 0.0) or 0.0)
    except Exception:
        confidence = 0.0

    # ---- logging with exchange-like decimals ----
    dp = _infer_dp(symbol, dec.setup.entry, features_by_tf, evidence_bundle)
    f_entry = _fmt(dec.setup.entry, dp)
    f_sl    = _fmt(dec.setup.sl, dp)
    f_tp1   = _fmt(tp1, dp) if tp1 is not None else None
    f_tp2   = _fmt(tp2, dp) if tp2 is not None else None
    f_tp3   = _fmt(tp3, dp) if tp3 is not None else None

    # legacy log line(s) — keep for backward-compat printing
    # Set plan STRATEGY for templates
    if (state or '').lower() == 'reversal':
        plan['STRATEGY'] = 'Reversal'
    legacy_lines = []
    legacy_lines.append(
        " ".join(
            [
                f"[{symbol}]",
                f"DECISION={decision}",
                f"| STATE={state or '-'}",
                f"| DIR={plan['direction'] or '-'}",
                f"| entry={f_entry}",
                f"sl={f_sl}",
                f"TP1={f_tp1}" if f_tp1 is not None else "TP1=None",
                f"TP2={f_tp2}" if f_tp2 is not None else "TP2=None",
                f"TP3={f_tp3}" if f_tp3 is not None else "TP3=None",
                f"RR1={f'{rr1:.1f}' if rr1 is not None else 'None'}",
                f"RR2={f'{rr2:.1f}' if rr2 is not None else 'None'}",
                f"RR3={f'{rr3:.1f}' if rr3 is not None else 'None'}",
                (
                    (lambda _v: f"LEV={__import__('math').floor(float(_v)):.1f}x")(size_hint)
                    if isinstance(size_hint,(int,float)) else ""
                ),
            ]
        )
    )

    # headline (one-liner) — show all three TPs
    _tp_parts_hl = [
        f"TP1={f_tp1}" if f_tp1 is not None else "TP1=None",
        f"TP2={f_tp2}" if f_tp2 is not None else "TP2=None",
        f"TP3={f_tp3}" if f_tp3 is not None else "TP3=None",
    ]
    _tp_text_hl = " ".join(_tp_parts_hl)
    headline = f"[{symbol}] {decision} | {state or '-'} {plan['direction'] or '-'} | E={f_entry} SL={f_sl} {_tp_text_hl}"

    # Telegram signal (nếu ENTER): format theo dp
    telegram_signal = None
    if decision == "ENTER" and plan["direction"] and dec.setup.sl is not None and (dec.setup.entry is not None or tp1 is not None):
        strategy = (state or "").replace("_", " ").title()
        entry_lines = []
        if dec.setup.entry is not None:
            entry_lines.append(f"Entry: {f_entry}")
        if f_tp1 is not None:
            entry_lines.append(f"TP1: {f_tp1}")
        if f_tp2 is not None:
            entry_lines.append(f"TP2: {f_tp2}")
        if f_tp3 is not None:
            entry_lines.append(f"TP3: {f_tp3}")
        telegram_signal = "\n".join(
            [
                f"#{symbol.replace('/', '')} {plan['direction']}",
                f"State: {state or '-'} | Strategy: {strategy}",
                *entry_lines,
                f"SL: {f_sl}",
                f"RR1: {rr1:.1f}" if rr1 is not None else "",
            ]
        ).strip()
    # Chuẩn hoá logs cho main.py:
    # - Giữ legacy text trong logs["TEXT"] (list)
    # - Cung cấp cấu trúc cho WAIT/ENTER để main.py lấy missing/reasons
    _meta_for_logs = _sanitize_meta_for_logs(dec.meta if isinstance(dec.meta, dict) else {})
    logs: Dict[str, Any] = {
        "TEXT": legacy_lines,
        "ENTER": {"state_meta": _meta_for_logs} if decision == "ENTER" else {},
        "WAIT": (
            {
                "missing": list(dec.reasons or []),
                "reasons": list(dec.reasons or []),
                "state_meta": _meta_for_logs,
            }
            if decision != "ENTER"
            else {}
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
    if "soft_proximity" in dec.reasons:
        notes.append(f"Soft proximity (BB/EMA): {prox.get('why','')}")
    if "intraday_reversal_shock" in dec.reasons:
        notes.append("1H volatility shock — chờ reclaim/ổn định rồi mới vào")
    if any("guard_1h_recent_opposite" in r for r in dec.reasons or []):
        notes.append("1H opposite impulse (engulf/marubozu) — đợi hấp thụ xong")

    # Build humanized strategy label for templates/teaser
    def _strategy_label(state: str, meta: Dict[str, Any] | None) -> str:
        label = (state or "").replace("_", " ").title() if state else "-"
        try:
            gate = (meta or {}).get("gate")
            if gate == "continuation" and "trend" in (state or ""):
                label = "Breakout (Continuation)"
            if (meta or {}).get("early_breakout"):
                # nếu early breakout, bổ sung tag
                if "Breakout" in label or "Trend Break" in label or "Trend" in label:
                    label = label.replace("Trend Break", "Breakout").replace("Trend", "Breakout")
                label = f"{label} — Early"
        except Exception:
           pass
        return label

    out = {
        "symbol": symbol,
        "timeframe": timeframe,
        "asof": evidence_bundle.get("asof"),
        "state": state,
        "confidence": round(confidence, 3),
        "decision": decision,
        "plan": plan,
        "logs": logs,
        "reasons": list(dec.reasons or []),  # tiện lợi, phòng khi caller cần
        "notes": notes,
        "headline": headline,
        "telegram_signal": telegram_signal,
        "strategy": _strategy_label(state, dec.meta),
        # meta gọn cho caller; tránh DataFrame
        "meta": _sanitize_meta_for_logs(dec.meta if isinstance(dec.meta, dict) else {}),
    }
    return out
