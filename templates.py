
from datetime import datetime, timezone
from typing import Dict, Any
import math, os



# Module exports (giúp static import/IDE & đảm bảo namespace đầy đủ)
__all__ = [
    "_side_of", "_entry_of", "_pct_for_hit",
    "_report_leverage", "_item_leverage", "fmt_price",
    "render_teaser", "render_full", "render_update",
    "render_summary", "render_kpi_teaser_two_parts", "render_kpi_week",
]

# ---------- helpers for KPI % calculation ----------
def _side_of(t: Dict[str, Any]) -> str:
    """
    Lấy side của lệnh theo các khóa phổ biến.
    """
    return str(t.get("side") or t.get("DIRECTION") or "").upper()

def _entry_of(t: Dict[str, Any]) -> float:
    """
    Lấy giá entry theo các khóa phổ biến. Thiếu thì trả 0 để tránh ZeroDivisionError.
    """
    for k in ("entry", "ENTRY", "entry_price", "price_entry"):
        try:
            v = float(t.get(k))
            if v and v > 0:
                return v
        except Exception:
            continue
    return 0.0

def _pct_for_hit(t: Dict[str, Any], price_hit: float) -> float:
    """
    % thay đổi so với entry theo side (LONG dương khi giá tăng; SHORT dương khi giá giảm).
    Trả về đơn vị % (ví dụ 1.23 nghĩa là +1.23%).
    An toàn với dữ liệu thiếu: nếu không đủ entry/price thì trả 0.0.
    """
    try:
        entry = _entry_of(t)
        if not entry or not price_hit:
            return 0.0
        pct = (float(price_hit) - float(entry)) / float(entry) * 100.0
        if _side_of(t) == "SHORT":
            pct = -pct
        return float(pct)
    except Exception:
        return 0.0


# --------- leverage helper for reports ----------
# (Compat aliases — để nơi khác gọi theo tên không dấu gạch dưới)
pct_for_hit = _pct_for_hit
entry_of = _entry_of
side_of = _side_of
# Defensive: đảm bảo helpers luôn có trong globals khi module được import
globals().setdefault("_pct_for_hit", _pct_for_hit)
globals().setdefault("_entry_of", _entry_of)
globals().setdefault("_side_of", _side_of)

def _report_leverage() -> float:
    """
    Hệ số đòn bẩy cho mục KPI 24H/tuần.
    Lấy từ ENV REPORT_LEVERAGE (vd: 3 cho x3). Mặc định 1.0 nếu không set/không hợp lệ.
    """
    try:
        lv = float(os.getenv("REPORT_LEVERAGE", "1"))
        return lv if lv > 0 else 1.0
    except Exception:
        return 1.0

# Lấy leverage tư vấn từ 1 item (signal/trade)
def _item_leverage(it: dict) -> float:
    for k in ("risk_size_hint", "leverage", "lev", "advice_leverage"):
        try:
            v = float(it.get(k)) if it and (k in it) else 0.0
            if v and v > 0:
                return v
        except Exception:
            continue
    return 0.0  # 0 nghĩa là không có dữ liệu

def fmt_price(v):
    """
    Auto-format cho CRYPTO:
    - Tự chọn số lẻ theo biên độ giá
    - Không dùng dấu phân tách nghìn (tránh nhầm dấu thập phân)
    """
    try:
        x = float(v)
    except Exception:
        return "-"
    ax = abs(x)
    if   ax >= 1000:  fmt = "{:,.2f}"
    elif ax >= 100:   fmt = "{:,.2f}"
    elif ax >= 10:    fmt = "{:,.2f}"
    elif ax >= 1:     fmt = "{:,.3f}"
    elif ax >= 0.1:   fmt = "{:,.4f}"
    elif ax >= 0.01:  fmt = "{:,.5f}"
    else:             fmt = "{:,.6f}"
    s = fmt.format(x).replace(",", "")     # bỏ dấu nghìn
    if "." in s:
        s = s.rstrip("0").rstrip(".")      # bỏ số 0 thừa cuối
    return s

def _humanize_state(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return "-"
    return s.replace("_", " ").title()

def render_teaser(plan: Dict[str, Any]) -> str:
    sym = plan.get("symbol", "")
    direction = plan.get("DIRECTION", "LONG")
    state = plan.get("STATE", "")
    strategy = plan.get("STRATEGY") or _humanize_state(state)
    # scale-out teaser (ưu tiên weights trong plan/meta)
    def _weights_line(p: Dict[str, Any]) -> str:
        w = (p.get("scale_out_weights") or {}) if isinstance(p, dict) else {}
        tp0w = p.get("tp0_weight")
        prof = (p.get("profile") or "").replace("-", " ")
        if w:
            def pct(x): 
                try: return f"{float(x)*100:.0f}%"
                except Exception: return "-"
            parts = [pct(w.get("tp1",0)), pct(w.get("tp2",0)), pct(w.get("tp3",0)), pct(w.get("tp4",0)), pct(w.get("tp5",0))]
            if isinstance(tp0w,(int,float)) and tp0w>0:
                return f"TP0: {pct(tp0w)} • " + " / ".join(parts) + (f"  ({prof})" if prof else "")
            return " / ".join(parts) + (f"  ({prof})" if prof else "")
        return "20% mỗi mốc TP"
    lines = [
        f"🧭 <b>{sym} | {direction}</b>",
        "",  # dòng trống sau tiêu đề
        f"<b>Entry:</b> —    <b>SL:</b> —",
        f"<b>TP:</b> — • — • — • — • —",
        f"<b>Scale-out:</b> {_weights_line(plan)}",
    ]
    return "\n".join(lines)

def render_full(plan: Dict[str, Any], username: str | None = None, watermark: bool = True) -> str:
    sym = plan.get("symbol", "")
    direction = plan.get("DIRECTION", "LONG")
    entry = fmt_price(plan.get("entry"))
    sl = fmt_price(plan.get("sl"))
    tp1 = fmt_price(plan.get("tp1")); tp2 = fmt_price(plan.get("tp2")); tp3 = fmt_price(plan.get("tp3")); tp4 = fmt_price(plan.get("tp4")); tp5 = fmt_price(plan.get("tp5"))
    # leverage (gợi ý)
    risk = plan.get("risk_size_hint")
    if isinstance(risk, (int, float)):
        risk_disp = math.floor(float(risk))
        don_bay_line = f"<b>Đòn bẩy:</b> x{int(risk_disp)}"
    else:
        don_bay_line = None
    strategy = plan.get("STRATEGY") or _humanize_state(plan.get("STATE", ""))
    # scale-out block
    def _scaleout_block(p: Dict[str, Any]) -> str | None:
        w = (p.get("scale_out_weights") or {}) if isinstance(p, dict) else {}
        prof = (p.get("profile") or "").replace("-", " ")
        tp0w = p.get("tp0_weight")
        if not w and not tp0w:
            return None
        def pct(x):
            try: return f"{float(x)*100:.0f}%"
            except Exception: return "-"
        parts = [
            f"TP1: {pct(w.get('tp1',0))}",
            f"TP2: {pct(w.get('tp2',0))}",
            f"TP3: {pct(w.get('tp3',0))}",
            f"TP4: {pct(w.get('tp4',0))}",
            f"TP5: {pct(w.get('tp5',0))}",
        ]
        head = f"<b>Scale-out:</b> " + (f"TP0: {pct(tp0w)} • " if isinstance(tp0w,(int,float)) and tp0w>0 else "")
        tail = " | ".join(parts) + (f"  ({prof})" if prof else "")
        return head + tail
    lines = [
        f"🧭 <b>{sym} | {direction}</b>",
        "",  # dòng trống sau tiêu đề
        
        f"<b>Entry:</b> {entry}",
        f"<b>SL:</b> {sl}",
        "",  # dòng trống sau Entry/SL
        
        f"<b>TP1:</b> {tp1}",
        f"<b>TP2:</b> {tp2}",
        f"<b>TP3:</b> {tp3}",
        f"<b>TP4:</b> {tp4}",
        f"<b>TP5:</b> {tp5}",
        "",  # dòng trống sau Entry/SL
    ]
    _so = _scaleout_block(plan)
    if _so:
        lines.append(_so)
    if don_bay_line:
        lines.append(don_bay_line)

    if watermark and username:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        lines.append(f"— sent to @{username} • {ts}")
    return "\n".join(lines)

def render_update(plan_or_trade: dict, event: str, extra: dict|None=None) -> str:
    sym = plan_or_trade.get("symbol",""); d = plan_or_trade.get("DIRECTION","")
    m = extra.get("margin_pct") if extra else None
    tail = f"\n<b>Lợi nhuận:</b> {m:.2f}%" if isinstance(m,(int,float)) else ""
    return f"<b>{sym} | {d}</b>\n<b>Update:</b> {event}{tail}"

def render_summary(kpi: dict, scope: str="Daily") -> str:
    return (
      f"<b>PNL {scope}</b>\n"
      f"• Trades: {kpi['n']}, Win-rate: {kpi['wr']:.0%}\n"
      f"• Avg R: {kpi['avgR']:.2f}\n"
      f"• Total R: {kpi['sumR']:.2f}"
    )
# ---------- helpers: danh sách lệnh đã đóng (1 cột, giữ icon) ----------
def _fmt_closed_cell(sym: str, pct: float, icon: str = "⚪",
                     width_sym: int = 12, width_pct: int = 7) -> str:
    """
    Trả về một “ô” dạng: ICON + SYMBOL (căn trái, độ rộng cố định) + PCT (căn phải).
    Dùng cho block <pre> để giữ lề trên Telegram/HTML.
    """
    sym = (sym or "?").upper()
    pct_str = f"{pct:+.2f}%"
    left  = f"{sym[:width_sym]:<{width_sym}}"
    right = f"{pct_str:>{width_pct}}"
    return f"{icon} {left} {right}"

def _format_closed_single_col(items: list) -> str:
    """
    Hiển thị danh sách đóng theo 1 cột (icon + mã + %), mỗi dòng 1 lệnh.
    Phù hợp giao diện mobile, tránh gãy hàng.
    """
    lines = []
    for it in (items or []):
        sym = (it.get("symbol") or "?").upper()
        try:
            pct = float(it.get("pct_weighted") or it.get("pct") or 0.0)
        except Exception:
            pct = 0.0

        # Icon theo lợi nhuận
        if pct > 0:
            icon = "🟢"
        elif pct < 0:
            icon = "⛔"
        else:
            icon = "⚪"

        lines.append(_fmt_closed_cell(sym, pct, icon))
    return "<pre>" + ("\n".join(lines) if lines else "(trống)") + "</pre>"

# NEW: Teaser 2 phần — Header + danh sách 24H, rồi khối hiệu suất NGÀY (today)
def render_kpi_teaser_two_parts(detail_24h: dict,
                                kpi_day: dict,
                                detail_day: dict,
                                report_date_str: str,
                                upgrade_url: str | None = None) -> str:
    lines = [f"🧭 <b>Kết quả giao dịch 24H qua — {report_date_str}</b>", ""]
    items = detail_24h.get("items", []) or []
    if not items:
        lines += ["Không có tín hiệu nào phù hợp.", ""]
    else:
        # Danh sách lệnh đã đóng (24H) — hiển thị 2 cột, giữ icon
        lines.append("<b>Danh sách lệnh đã đóng:</b>")
        lines.append(_format_closed_single_col(items))
        lines.append("")
        
    # Khối hiệu suất ngày (Today) — hiển thị cả R và % thực nhận
    lines.append("<b>Hiệu suất ngày:</b>")
    try:
        wr = float(kpi_day.get("wr", 0.0) or 0.0)
        avgR = float(kpi_day.get("avgR", 0.0) or 0.0)
        sumR = float(kpi_day.get("sumR", 0.0) or 0.0)
        avgPctW = float(kpi_day.get("avgPctW", 0.0) or 0.0)
        sumPctW = float(kpi_day.get("sumPctW", 0.0) or 0.0)
        # Tổng lệnh đã đóng trong ngày:
        # 1) ưu tiên số liệu từ detail_day (n hoặc items)
        # 2) nếu rỗng (hoặc =0) thì fallback sang thống kê 24H (detail_24h)
        def _count_items(d: dict) -> int:
            try:
                n = int(d.get("n") or 0)
            except Exception:
                n = 0
            if n <= 0:
                n = len(d.get("items") or [])
            return int(n)
        n_day = _count_items(detail_day)
        if n_day <= 0:
            n_day = _count_items(detail_24h)
        lines.append(f"• Tổng lệnh đã đóng: {n_day}")
        lines.append(f"• Tổng lợi nhuận: {sumPctW:.2f}%")
        lines.append(f"• Lợi nhuận trung bình: {avgPctW:.2f}%")
        lines.append(f"• Tỉ lệ thắng: {wr:.0%}")
        lines.append(f"• Tổng R: {sumR:.2f}")
        lines.append(f"• R trung bình: {avgR:.2f}")
        
    except Exception:
        lines.append("• (thiếu dữ liệu)")

    # Lời mời nâng cấp
    if upgrade_url:
        lines.append("🔒 <b>Nâng cấp Plus</b> để xem full tín hiệu & nhận thông báo sớm hơn.")
        lines.append(f'<a href="{upgrade_url}">👉 Nâng cấp ngay</a>')
    return "\n".join(lines)

# KPI WEEK
def render_kpi_week(detail: dict,
                    week_label: str,
                    *_,
                    upgrade_url: str | None = None) -> str:
    """
    KPI tuần — format giống block 'Hiệu suất ngày' của KPI 24H:
    - Không liệt kê danh sách lệnh đóng
    - Chỉ hiển thị các chỉ số tổng hợp: n, sumPctW, avgPctW, win-rate, sumR, avgR
    """
    totals   = detail.get("totals") or {}
    n        = int(totals.get("n") or 0)
    wr       = float(totals.get("win_rate") or 0.0)          # 0..1
    sumR     = float(totals.get("sum_R_weighted") or totals.get("sum_R") or 0.0)
    avgR     = float(totals.get("avg_R") or (sumR / n if n else 0.0))
    sumPctW  = float(totals.get("sum_pct_weighted") or totals.get("sum_pct_w") or totals.get("sum_pct") or 0.0)
    avgPctW  = float(totals.get("avg_pct_weighted") or (sumPctW / n if n else 0.0))

    lines = [f"🧭 <b>Kết quả giao dịch tuần qua — {week_label}</b>", ""]
    lines.append(f"• Tổng lệnh đã đóng: {n}")
    lines.append(f"• Tổng lợi nhuận: {sumPctW:.2f}%")
    lines.append(f"• Lợi nhuận trung bình: {avgPctW:.2f}%")
    lines.append(f"• Tỉ lệ thắng: {wr:.0%}")
    lines.append(f"• Tổng R: {sumR:.2f}")
    lines.append(f"• R trung bình: {avgR:.2f}")

    if upgrade_url:
        lines.append("🔒 <b>Nâng cấp Plus</b> để xem full tín hiệu & nhận thông báo sớm hơn.")
        lines.append(f'<a href="{upgrade_url}">👉 Nâng cấp ngay</a>')
    return "\n".join(lines)


