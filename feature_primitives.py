
"""
feature_primitives.py — First-layer “feature primitives” for OHLCV streams
--------------------------------------------------------------------------
Expected input columns (already enriched beforehand):
  open, high, low, close, volume, ema20, ema50, rsi14, atr14,
  bb_upper, bb_mid, bb_lower, vol_sma20, vol_ratio, vol_z20,
  (body_pct, upper_wick_pct, lower_wick_pct if candle patterns are used).

Feature groups:
  1) Trend (ZigZag/Swings, Trend EMA, Candle patterns)
  2) Momentum (Volume, RSI & Divergence, Volatility/BBW)
  3) Support/Resistance (ranked hard levels + soft levels)
  4) Multi-timeframe orchestration + Volume Profile per TF
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo

# Import from indicators (must exist in your project)
from indicators import enrich_indicators, calc_vp

VN_TZ = ZoneInfo("Asia/Ho_Chi_Minh")

# =========================
# Config: TF → Volume Profile params & SR weights
# =========================
TF_VP = {
    "4H": {"window_bars": 240, "bins": 30, "top_k": 6},
    "1D": {"window_bars": 180, "bins": 24, "top_k": 5},
    "1W": {"window_bars": 156, "bins": 18, "top_k": 5},
}
LEVEL_WEIGHTS = {"touch": 0.35, "dwell": 0.20, "psych": 0.25, "vp": 0.20}

# --- robust coerce helper ---
def _ensure_series(x) -> pd.Series:
    if isinstance(x, pd.Series):
        return x
    return pd.Series(x)

def _zscore(series: pd.Series, window: int = 20) -> pd.Series:
    s = _ensure_series(pd.to_numeric(series, errors="coerce"))
    mean = s.rolling(window).mean()
    std = s.rolling(window).std(ddof=0)
    return (s - mean) / std

# =========================
# Helpers (stream-safe)
# =========================
def _last_closed_bar(df: pd.DataFrame) -> pd.Series:
    """
    Return the most recently CLOSED bar: if there are ≥2 bars, use iloc[-2],
    otherwise use iloc[-1].
    """
    if df is None or len(df) == 0:
        raise ValueError("empty dataframe")
    return df.iloc[-2] if len(df) >= 2 else df.iloc[-1]

# =========================
# Trend block
# =========================
def _zigzag(series: pd.Series, pct: float) -> list[tuple[pd.Timestamp, float]]:
    """
    Simple percent-based ZigZag over a closing price series.
    Iterate sequentially; when move reaches ±pct%, mark a turning point.
    Returns a list of (timestamp, price).
    """
    s = pd.to_numeric(series, errors="coerce")
    idx = s.index
    if len(s) < 2 or s.dropna().empty:
        return []
    pct = abs(float(pct))
    pts = []
    last_p = s.iloc[0]
    last_i = idx[0]
    dirn = 0  # +1 up, -1 down, 0 unknown
    for i in range(1, len(s)):
        p = s.iloc[i]
        if pd.isna(p) or pd.isna(last_p):
            continue
        chg = (p / last_p - 1.0) * 100.0
        if dirn >= 0 and chg >= pct:
            # swing up
            pts.append((last_i, float(last_p)))
            dirn = 1
            last_p = p
            last_i = idx[i]
        elif dirn <= 0 and chg <= -pct:
            # swing down
            pts.append((last_i, float(last_p)))
            dirn = -1
            last_p = p
            last_i = idx[i]
        else:
            # extend move extreme
            if (dirn >= 0 and p > last_p) or (dirn <= 0 and p < last_p) or dirn == 0:
                last_p = p
                last_i = idx[i]
    pts.append((last_i, float(last_p)))
    # ensure unique/ordered
    uniq = []
    for ts, pr in pts:
        ts = pd.to_datetime(ts)
        if not uniq or ts != uniq[-1][0]:
            uniq.append((ts, pr))
    return uniq

def compute_swings(
    df: pd.DataFrame, pct: float = 2.0, lookback: int = 250, max_keep: int = 20, last_n_each: int = 3
) -> dict:
    sub = df.tail(int(lookback))
    zz = _zigzag(sub["close"], pct=pct)
    if len(zz) < 2:
        return {"zigzag": zz, "labels": [], "last_HH": [], "last_LL": []}
    labels = []
    last_HH, last_LL = [], []
    for i in range(1, len(zz)):
        t0, p0 = zz[i - 1]
        t1, p1 = zz[i]
        lab = "HH" if p1 > p0 else ("LL" if p1 < p0 else "EQ")
        labels.append((t1, lab, p1))
        if lab == "HH":
            last_HH.append((t1, p1))
        elif lab == "LL":
            last_LL.append((t1, p1))
    zz = zz[-max_keep:]
    labels = labels[-max_keep:]
    return {
        "zigzag": zz,
        "labels": labels,
        "last_HH": last_HH[-last_n_each:],
        "last_LL": last_LL[-last_n_each:],
    }

def compute_trend(df: pd.DataFrame) -> dict:
    e = df
    ema20 = e["ema20"]
    ema50 = e["ema50"]
    spread = (ema20 - ema50)
    # Smooth slope: average of last 3 diffs of ema50
    ema50_slope = (ema50.diff().tail(3).mean())
    px = e["close"].iloc[-1]
    state = "side"
    if px > ema20.iloc[-1] > ema50.iloc[-1]:
        state = "up"
    elif px < ema20.iloc[-1] < ema50.iloc[-1]:
        state = "down"
    return {
        "state": state,
        "ema_spread": float(spread.iloc[-1]),
        "ema50_slope": float(ema50_slope),
    }

def compute_candles(df: pd.DataFrame) -> dict:
    """
    Pin, Engulfing, Inside using body_pct/upper_wick_pct/lower_wick_pct & candle color.
    Uses the last CLOSED bar for safety when streaming.
    """
    if len(df) < 2:
        return {
            "bullish_pin": False,
            "bearish_pin": False,
            "bullish_engulf": False,
            "bearish_engulf": False,
            "inside_bar": False,
        }
    last = _last_closed_bar(df)
    prev = df.iloc[-2]
    o, c = float(last["open"]), float(last["close"])
    color_up = c > o
    body_pct = float(last.get("body_pct", np.nan))
    uw = float(last.get("upper_wick_pct", np.nan))
    lw = float(last.get("lower_wick_pct", np.nan))
    # Heuristic pins
    bullish_pin = (lw >= 0.5) and (body_pct <= 0.35) and color_up
    bearish_pin = (uw >= 0.5) and (body_pct <= 0.35) and (not color_up)
    # Engulfing: body of the current bar engulfs the previous bar's body
    o1, c1 = float(prev["open"]), float(prev["close"])
    bull_engulf = (color_up and (o <= c1) and (c >= o1) and (c1 < o1))
    bear_engulf = ((not color_up) and (o >= c1) and (c <= o1) and (c1 > o1))
    # Inside bar: high/low within previous bar's high/low
    inside = (last["high"] <= prev["high"]) and (last["low"] >= prev["low"])
    return {
        "bullish_pin": bool(bullish_pin),
        "bearish_pin": bool(bearish_pin),
        "bullish_engulf": bool(bull_engulf),
        "bearish_engulf": bool(bear_engulf),
        "inside_bar": bool(inside),
    }

# =========================
# Momentum block
# =========================
def compute_volume_features(df: pd.DataFrame) -> dict:
    v = pd.to_numeric(df["volume"], errors="coerce")
    v3 = v.rolling(3).mean()
    v5 = v.rolling(5).mean()
    v10 = v.rolling(10).mean()
    v20 = pd.to_numeric(df.get("vol_sma20", np.nan), errors="coerce")
    contraction = bool((v5.iloc[-1] < v10.iloc[-1]) and (v10.iloc[-1] < v20.iloc[-1])) if len(v) >= 20 else False
    vol_ratio = float(df.get("vol_ratio", np.nan).iloc[-1])
    vol_z = float(df.get("vol_z20", np.nan).iloc[-1]) if "vol_z20" in df.columns else float(_zscore(v.shift(1), 20).iloc[-1])
    now = float(v.iloc[-1])
    prev_closed = float(v.shift(1).iloc[-1])
    median = float(v.rolling(20).median().iloc[-1])
    break_vol_ok = (vol_ratio >= 1.5) or (vol_z >= 1.0)
    break_vol_strong = (vol_ratio >= 2.0) or (vol_z >= 2.0)
    return {
        "v3": float(v3.iloc[-1]) if len(v) >= 3 else np.nan,
        "v5": float(v5.iloc[-1]) if len(v) >= 5 else np.nan,
        "v10": float(v10.iloc[-1]) if len(v) >= 10 else np.nan,
        "v20": float(v20.iloc[-1]) if not np.isnan(v20.iloc[-1]) else np.nan,
        "contraction": bool(contraction),
        "now": now,
        "prev_closed": prev_closed,
        "median": median,
        "vol_ratio": vol_ratio,
        "vol_z": vol_z,
        "break_vol_ok": bool(break_vol_ok),
        "break_vol_strong": bool(break_vol_strong),
    }

def _recent_extrema(vals: pd.Series, n: int = 30):
    s = pd.to_numeric(vals.tail(n), errors="coerce")
    if s.dropna().empty:
        return np.nan, np.nan, np.nan, np.nan
    return s.max(), s.idxmax(), s.min(), s.idxmin()

def compute_momentum(df: pd.DataFrame) -> dict:
    r = pd.to_numeric(df["rsi14"].tail(30), errors="coerce")
    rsi_last = float(r.iloc[-1]) if len(r) else np.nan
    # Divergence: simple check via recent highs/lows
    cmax, cmax_i, cmin, cmin_i = _recent_extrema(df["close"], n=30)
    rmax, rmax_i, rmin, rmin_i = _recent_extrema(r, n=30)
    divergence = "none"
    try:
        if pd.notna(cmax) and pd.notna(rmax):
            # price makes new high, RSI fails to make new high -> bearish
            prev_cmax = pd.to_numeric(df["close"].iloc[:-1].tail(30), errors="coerce").max()
            prev_rmax = pd.to_numeric(r.iloc[:-1], errors="coerce").max()
            if cmax >= prev_cmax and rmax <= prev_rmax:
                divergence = "bearish"
        if pd.notna(cmin) and pd.notna(rmin) and divergence == "none":
            # price makes new low, RSI fails to make new low -> bullish
            prev_cmin = pd.to_numeric(df["close"].iloc[:-1].tail(30), errors="coerce").min()
            prev_rmin = pd.to_numeric(r.iloc[:-1], errors="coerce").min()
            if cmin <= prev_cmin and rmin >= prev_rmin:
                divergence = "bullish"
    except Exception:
        divergence = "none"
    return {"rsi_last": rsi_last, "divergence": divergence}

def compute_volatility(df: pd.DataFrame, bbw_lookback: int = 50) -> dict:
    """
    Robust volatility snapshot (stream-safe):
      - Luôn đọc tại NẾN ĐÃ ĐÓNG gần nhất.
      - Nếu thiếu/NaN atr14: khôi phục bằng Wilder ATR trên tail(64), rồi fallback simple ATR.
      - BBW: lấy tại nến đã đóng; median tính trên lịch sử tới nến đó (không dùng nến đang chạy).
    """
    # Lấy nến đã đóng
    try:
        last = _last_closed_bar(df)
        idx = last.name
    except Exception:
        return {"atr": np.nan, "natr": np.nan, "bbw_last": np.nan, "bbw_med": np.nan, "squeeze": False}

    # --- close_last (guard 0/NaN) ---
    try:
        close_last = float(pd.to_numeric(df.loc[idx, "close"], errors="coerce"))
    except Exception:
        close_last = np.nan

    # --- ATR14 robust ---
    def _wilder_atr(_df: pd.DataFrame, period: int = 14) -> pd.Series:
        h = pd.to_numeric(_df["high"], errors="coerce")
        l = pd.to_numeric(_df["low"], errors="coerce")
        c = pd.to_numeric(_df["close"], errors="coerce")
        tr = pd.concat([(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
        alpha = 1.0 / float(period)
        return tr.ewm(alpha=alpha, adjust=False).mean().rename("atr14")

    # 1) ưu tiên atr14 đã enrich tại nến đã đóng
    try:
        atr14_val = float(pd.to_numeric(df.loc[idx, "atr14"], errors="coerce")) if "atr14" in df.columns else np.nan
    except Exception:
        atr14_val = np.nan

    # 2) nếu NaN → tính lại Wilder ATR trên tail(64) với kỳ linh hoạt
    if not np.isfinite(atr14_val):
        tail = df[["high", "low", "close"]].dropna().loc[:idx].tail(64)
        if len(tail) >= 4:
            period = int(min(14, max(3, len(tail) - 1)))
            atr_series = _wilder_atr(tail, period=period)
            if len(atr_series):
                atr14_val = float(atr_series.iloc[-1])

    # 3) nếu vẫn thiếu → simple ATR (mean TR 5–10 nến cuối)
    if not np.isfinite(atr14_val):
        tail = df[["high", "low", "close"]].dropna().loc[:idx].tail(10)
        if len(tail) >= 5:
            tr = pd.concat([
                (pd.to_numeric(tail["high"]) - pd.to_numeric(tail["low"])),
                (pd.to_numeric(tail["high"]) - pd.to_numeric(tail["close"]).shift()).abs(),
                (pd.to_numeric(tail["low"])  - pd.to_numeric(tail["close"]).shift()).abs(),
            ], axis=1).max(axis=1)
            atr14_val = float(tr.tail(5).mean())

    # natr
    natr = (atr14_val / close_last) if (np.isfinite(atr14_val) and np.isfinite(close_last) and close_last > 0.0) else np.nan

    # --- BBW robust tại nến đã đóng ---
    try:
        if "bb_width_pct" in df.columns:
            bbw_series = pd.to_numeric(df["bb_width_pct"], errors="coerce")
        elif {"bb_upper", "bb_mid", "bb_lower"}.issubset(df.columns):
            bb_mid = pd.to_numeric(df["bb_mid"], errors="coerce")
            base = bb_mid.where((bb_mid != 0) & bb_mid.notna(), other=pd.to_numeric(df["close"], errors="coerce"))
            bbw_series = (pd.to_numeric(df["bb_upper"], errors="coerce")
                          - pd.to_numeric(df["bb_lower"], errors="coerce")) / base * 100.0
            bbw_series = bbw_series.replace([np.inf, -np.inf], np.nan)
        else:
            bbw_series = pd.Series([np.nan] * len(df), index=df.index)
    except Exception:
        bbw_series = pd.Series([np.nan] * len(df), index=df.index)

    try:
        bbw_last = float(pd.to_numeric(bbw_series.loc[idx], errors="coerce")) if idx in bbw_series.index else np.nan
    except Exception:
        bbw_last = np.nan
    try:
        hist = pd.to_numeric(bbw_series.loc[:idx], errors="coerce").tail(int(bbw_lookback))
        bbw_med = float(hist.median()) if len(hist) else np.nan
    except Exception:
        bbw_med = np.nan

    squeeze = bool(np.isfinite(bbw_last) and np.isfinite(bbw_med) and (bbw_last < bbw_med))

    return {
        "atr": float(atr14_val) if np.isfinite(atr14_val) else np.nan,
        "natr": float(natr) if np.isfinite(natr) else np.nan,
        "bbw_last": float(bbw_last) if np.isfinite(bbw_last) else np.nan,
        "bbw_med": float(bbw_med) if np.isfinite(bbw_med) else np.nan,
        "squeeze": squeeze,
    }

# =========================
# Support / Resistance block
# =========================
def _round_to_psych(p: float) -> float:
    if np.isnan(p):
        return p
    # round to a psychological level by the magnitude of price
    mag = 10 ** int(np.floor(np.log10(max(1e-9, abs(p)))))
    step = mag / 2  # 0.5 * 10^k
    return round(p / step) * step

def compute_levels(
    df: pd.DataFrame,
    atr: float | None = None,
    tol_coef: float = 0.5,
    extremes: int = 12,
    lookback: int = 300,
    vp_zones: Optional[pd.DataFrame] = None,
    weights: Optional[dict] = None,
) -> dict:
    """
    Hard SR levels: cluster price levels by tolerance and score via touch/dwell/psych/vp.
    Returns sr_up/sr_down & bands_up/bands_down with ranking fields.
    """
    e = df.tail(int(lookback)).copy()
    px = float(e["close"].iloc[-1])
    if atr is None:
        # robust atr: tránh KeyError/NaN
        try:
            atr = float(pd.to_numeric(e["atr14"], errors="coerce").iloc[-1])
        except Exception:
            atr = np.nan
    if pd.isna(atr):
        # fallback nhẹ: dùng biên độ gần nhất nếu ATR chưa sẵn
        try:
            atr = float((e["high"].iloc[-1] - e["low"].iloc[-1]))
        except Exception:
            atr = 0.0
    tol = max(1e-9, atr * float(tol_coef))
    highs = e["high"]
    lows = e["low"]
    close = e["close"]
    # local extrema candidates
    roll_hi = highs.rolling(3, center=True).max()
    roll_lo = lows.rolling(3, center=True).min()
    candidates = []
    for i in range(1, len(e) - 1):
        if highs.iloc[i] >= roll_hi.iloc[i]:
            candidates.append(float(highs.iloc[i]))
        if lows.iloc[i] <= roll_lo.iloc[i]:
            candidates.append(float(lows.iloc[i]))
    # extremes by close
    candidates += list(np.linspace(close.min(), close.max(), num=max(2, extremes)))
    candidates = sorted([c for c in candidates if pd.notna(c)])
    # cluster by tolerance
    bands = []
    cur = []
    for c in candidates:
        if not cur:
            cur = [c]
        elif abs(c - np.mean(cur)) <= tol:
            cur.append(c)
        else:
            bands.append(cur)
            cur = [c]
    if cur:
        bands.append(cur)
    # enrich bands
    recs = []
    for b in bands:
        low = float(min(b))
        high = float(max(b))
        mid = (low + high) / 2.0
        # touches: number of closes within tol around mid
        in_band = (close.between(mid - tol, mid + tol)).sum()
        # dwell: longest sideways streak around band
        dwell = int(pd.Series(close.between(mid - tol, mid + tol)).rolling(5).sum().fillna(0).max())
        # psych: proximity to a rounded psychological level
        psych_mid = _round_to_psych(mid)
        psych_conf = float(1.0 - min(1.0, abs(mid - psych_mid) / max(1e-9, tol)))
        # vp overlap
        vp_weight = 0.0
        if vp_zones is not None and len(vp_zones):
            for _, z in vp_zones.iterrows():
                if (z["low"] <= mid <= z["high"]) or (low <= z["mid"] <= high):
                    vp_weight = max(vp_weight, float(z["volume_sum"]))
        recs.append({
            "low": low,
            "high": high,
            "mid": mid,
            "touches": int(in_band),
            "dwell": int(dwell),
            "psych_conf": float(psych_conf),
            "vp_raw": vp_weight
        })
    dfb = pd.DataFrame(recs)
    if dfb.empty:
        return {"sr_up": [], "sr_down": [], "bands_up": [], "bands_down": [], "tol": tol}
    if dfb["vp_raw"].max() > 0:
        dfb["vp_weight"] = dfb["vp_raw"] / dfb["vp_raw"].max()
    else:
        dfb["vp_weight"] = 0.0
    dfb["touch_n"] = dfb["touches"] / max(1, dfb["touches"].max())
    dfb["dwell_n"] = dfb["dwell"] / max(1, dfb["dwell"].max())
    w = (weights or LEVEL_WEIGHTS)
    dfb["score"] = (
        w["touch"] * dfb["touch_n"]
        + w["dwell"] * dfb["dwell_n"]
        + w["psych"] * dfb["psych_conf"]
        + w["vp"] * dfb["vp_weight"]
    )
    # classify
    q1, q2 = dfb["score"].quantile([0.33, 0.66])
    dfb["strength"] = np.where(
        dfb["score"] >= q2,
        "strong",
        np.where(dfb["score"] >= q1, "medium", "weak"),
    )
    # split up/down
    sr_up = dfb[dfb["mid"] > px].sort_values("mid")["mid"].tolist()
    sr_down = dfb[dfb["mid"] < px].sort_values("mid", ascending=False)["mid"].tolist()
    bands_up = dfb[dfb["mid"] > px].sort_values(["score", "mid"], ascending=[False, True]).to_dict("records")
    bands_down = dfb[dfb["mid"] < px].sort_values(["score", "mid"], ascending=[False, False]).to_dict("records")
    return {
        "sr_up": sr_up,
        "sr_down": sr_down,
        "bands_up": bands_up,
        "bands_down": bands_down,
        "tol": tol,
    }

def compute_soft_levels(df: pd.DataFrame) -> dict:
    px = float(df["close"].iloc[-1])
    levels = {
        "BB.upper": float(df["bb_upper"].iloc[-1]),
        "BB.mid": float(df["bb_mid"].iloc[-1]),
        "BB.lower": float(df["bb_lower"].iloc[-1]),
        "EMA20": float(df["ema20"].iloc[-1]),
        "EMA50": float(df["ema50"].iloc[-1]),
        "SMA20": float(df.get("sma20", df["close"].rolling(20).mean()).iloc[-1]),
        "SMA50": float(df.get("sma50", df["close"].rolling(50).mean()).iloc[-1]),
    }
    soft_up = sorted([v for v in levels.values() if v > px], key=lambda x: x)
    soft_down = sorted([v for v in levels.values() if v < px], key=lambda x: -x)
    return {"soft_up": soft_up, "soft_down": soft_down, "all": levels}

# =========================
# Public APIs
# =========================
def enrich_and_features(df: pd.DataFrame, timeframe: str) -> dict:
    """
    Convenience API: enrich then compute a compact snapshot for the latest bar.
    """
    if df is None or df.empty:
        return {"timeframe": timeframe, "df": df, "features": {}, "error": "empty"}
    e = enrich_indicators(df)
    # primitives
    swings = compute_swings(e, pct=2.0)
    trend = compute_trend(e)
    candles = compute_candles(e)
    vol = compute_volume_features(e)
    momentum = compute_momentum(e)
    vola = compute_volatility(e)
    # bundle snapshot
    features = {
        "swings": swings,
        "trend": trend,
        "candles": candles,
        "volume": vol,
        "momentum": momentum,
        "volatility": vola,
    }
    return {"timeframe": timeframe, "df": e, "features": features}

def compute_features_by_tf(dfs_by_tf: Dict[str, pd.DataFrame]) -> Dict[str, dict]:
    """
    Multi-timeframe orchestrator:
      - sort index → enrich → compute: swings, trend, candles, volume, momentum, volatility
      - vp_zones per TF via TF_VP (error-safe)
      - levels (ranked hard SR) + soft_levels
      - returns dict per TF: includes all blocks + enriched df
    """
    results: Dict[str, dict] = {}
    for tf, raw in dfs_by_tf.items():
        try:
            if raw is None or raw.empty:
                results[tf] = {"timeframe": tf, "df": raw, "error": "empty"}
                continue
            d = raw.sort_index()
            enriched = enrich_indicators(d)
            # Core primitives
            swings = compute_swings(enriched, pct=2.0)
            trend = compute_trend(enriched)
            candles = compute_candles(enriched)
            vol = compute_volume_features(enriched)
            momentum = compute_momentum(enriched)
            vola = compute_volatility(enriched)
            # VP zones per TF
            vp_cfg = TF_VP.get(tf, TF_VP["1D"])
            try:
                vp_zones = calc_vp(enriched, **vp_cfg)
            except Exception:
                vp_zones = None
            # Levels (hard + soft)
            levels = compute_levels(
                enriched, atr=None, tol_coef=0.5, lookback=300, vp_zones=vp_zones, weights=LEVEL_WEIGHTS
            )
            soft = compute_soft_levels(enriched)
            results[tf] = {
                "timeframe": tf,
                "df": enriched,
                "primitives": {
                    "swings": swings,
                    "trend": trend,
                    "candles": candles,
                    "volume": vol,
                    "momentum": momentum,
                    "volatility": vola,
                },
                "vp_zones": None if vp_zones is None else vp_zones.to_dict("records"),
                "levels": levels,
                "soft_levels": soft,
                "weights": LEVEL_WEIGHTS,
            }
        except Exception as e:
            results[tf] = {"timeframe": tf, "df": raw, "features": {}, "error": str(e)}
    return results
