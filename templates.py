
from datetime import datetime, timezone
from typing import Dict, Any

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

def render_teaser(plan: Dict[str, Any]) -> str:
    sym = plan.get("symbol", "")
    direction = plan.get("DIRECTION", "LONG")
    state = plan.get("STATE", "")
    strategy = " • ".join([n for n in plan.get("notes", [])[:1]])  # one-liner
    return (
        f"🧭 <b>{sym} | {direction}</b>\n"
        f"<b>Entry:</b> —    <b>SL:</b> —\n"
        f"<b>TP:</b> — • — • —\n"
        f"<b>Chiến lược:</b> {strategy or state}"
    )

def render_full(plan: Dict[str, Any], username: str | None = None, watermark: bool = True) -> str:
    sym = plan.get("symbol", "")
    direction = plan.get("DIRECTION", "LONG")
    entry = fmt_price(plan.get("entry"))
    sl = fmt_price(plan.get("sl"))
    tp1 = fmt_price(plan.get("tp1")); tp2 = fmt_price(plan.get("tp2")); tp3 = fmt_price(plan.get("tp3"))
    rr = plan.get("rr"); risk = plan.get("risk_size_hint")
    rr_txt = f"{rr:.2f}" if isinstance(rr, (int,float)) else "-"
    hint = f" — <b>Size:</b> {risk:.1f}x" if isinstance(risk, (int,float)) else ""
    lines = [
        f"🧭 <b>{sym} | {direction}</b>",
        "",  # dòng trống sau tiêu đề
        
        f"<b>Entry:</b> {entry}",
        f"<b>SL:</b> {sl}",
        "",  # dòng trống sau Entry/SL
        
        f"<b>TP1:</b> {tp1}",
        f"<b>TP2:</b> {tp2}",
        f"<b>TP3:</b> {tp3}",
        "",  # dòng trống sau block TP
        
        f"<b>R:R:</b> {rr_txt}{hint}",
    ]

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
# NEW: KPI 24H chi tiết
def render_kpi_24h(detail: dict, report_date_str: str, upgrade_url: str | None = None) -> str:
    items = detail["items"]
    totals = detail["totals"]
    # 0) Header
    lines = [f"<b>Kết quả giao dịch 24H qua — {report_date_str}</b>", ""]
    # 1) Danh sách tín hiệu
    if not items:
        lines += ["Không có tín hiệu nào trong 24H qua.", ""]
    else:
        icons = {
        "TP1": "🟢",
        "TP2": "🟢",
        "TP3": "🟢",
        "SL": "⛔",
    }
    for it in detail["items"]:
        status = it["status"]
        icon = icons.get(status, "⚪")
        line = f"{icon} {it['symbol']}: {it['pct']:.2f}%"
        lines.append(line)
    # 2) Đánh giá
    lines += [
        "<b>Đánh giá</b>:",
        f"• Tổng lệnh đã đóng: {totals['n']}",
        f"• Tổng lợi nhuận: {totals['sum_pct']:.2f}%",
        f"• Lợi nhuận trung bình/lệnh: {totals['avg_pct']:.2f}%",
        f"• Tỉ lệ thắng: {totals['win_rate']*100:.2f}%",
        f"• Số lệnh thắng: {totals['wins']}",
        f"• Số lệnh thua: {totals['losses']}",
        ""
    ]
    # 3) Lời mời nâng cấp
    if upgrade_url:
        lines.append("🔒 <b>Nâng cấp Plus</b> để xem full tín hiệu & nhận thông báo sớm hơn.")
        lines.append(f'<a href="{upgrade_url}">👉 Nâng cấp ngay</a>')
    return "\n".join(lines)
