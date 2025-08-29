
from datetime import datetime, timezone
from typing import Dict, Any

def fmt_price(v):
    try:
        return "{:,}".format(int(round(float(v)))).replace(",", ".")
    except Exception:
        return v

def render_teaser(plan: Dict[str, Any]) -> str:
    sym = plan.get("symbol", "")
    direction = plan.get("DIRECTION", "LONG")
    state = plan.get("STATE", "")
    strategy = " • ".join([n for n in plan.get("notes", [])[:1]])  # one-liner
    return (
        f"<b>{sym} | {direction}</b>\n"
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
        f"<b>{sym} | {direction}</b>",
        f"<b>Entry:</b> {entry} — <b>SL:</b> {sl}",
        f"<b>TP1:</b> {tp1} • <b>TP2:</b> {tp2} • <b>TP3:</b> {tp3}",
        f"<b>R:R:</b> {rr_txt}{hint}",
    ]
    if watermark and username:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        lines.append(f"— sent to @{username} • {ts}")
    return "\n".join(lines)
