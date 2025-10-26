# decision_monitor.py
import os, time, csv
from datetime import datetime

def log_decision_stats(plan: dict, features_by_tf: dict, perfdb=None, data_dir: str = "./data"):
    """
    Ghi thống kê mỗi khi engine ra quyết định (ENTER/WAIT/AVOID).
    Bao gồm: regime, NATR%, graded_params, guards, signal type, v.v.
    """
    try:
        if not plan or not isinstance(plan, dict):
            return
        meta = plan.get("meta", {}) or {}
        graded = meta.get("graded_setup", {})
        guards = meta.get("guards_triggered", [])
        regime = meta.get("regime") or meta.get("market_regime") or "-"
        decision = plan.get("decision") or "-"
        symbol = plan.get("symbol") or plan.get("SYMBOL") or "-"
        side = plan.get("side") or plan.get("DIRECTION") or "-"

        # Dữ liệu volatility từ features
        natr_pct = None
        try:
            df4 = features_by_tf.get("4H", {}).get("df")
            if df4 is not None and "natr_pct" in df4.columns:
                natr_pct = float(df4["natr_pct"].iloc[-2])
        except Exception:
            natr_pct = None

        row = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "symbol": symbol,
            "side": side,
            "decision": decision,
            "regime": regime,
            "natr_pct": natr_pct,
            "factor": graded.get("factor"),
            "lev_scale": graded.get("lev_scale"),
            "sl_mult": graded.get("sl_atr_mult"),
            "rr_scale": graded.get("rr_scale"),
            "guards": "|".join(guards),
        }

        path = os.path.join(data_dir, "decision_log.csv")
        os.makedirs(data_dir, exist_ok=True)
        write_header = not os.path.exists(path)
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=row.keys())
            if write_header:
                w.writeheader()
            w.writerow(row)

    except Exception as e:
        print("[decision_monitor] failed:", e)
