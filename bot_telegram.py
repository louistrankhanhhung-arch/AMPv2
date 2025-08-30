
import re
from typing import Dict, Any, Tuple
from templates import render_full
from storage import JsonStore, UserDB, SignalCache, PaymentDB
from config import BOT_TOKEN, OWNER_IDS, DATA_DIR, BANK_INFO, PLAN_DEFAULT_MONTHS, PROTECT_CONTENT, WATERMARK
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, InputFile
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, ContextTypes, filters

store = JsonStore(DATA_DIR)
users = UserDB(store)
signals = SignalCache(store)
payments = PaymentDB(store)

def is_owner(uid: int) -> bool:
    return uid in OWNER_IDS

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Lấy payload từ cả context.args và fallback từ text (khi client chỉ gửi /start)
    payload = " ".join(context.args) if context.args else ""
    if not payload and update.message and update.message.text:
        parts = update.message.text.split(" ", 1)
        if len(parts) == 2:
            payload = parts[1].strip()

    if payload.startswith("show_"):
        signal_id = payload.split("_", 1)[1]
        uid = update.effective_user.id
        uname = update.effective_user.username or ""
        if users.is_plus_active(uid):
            # Ưu tiên render từ PLAN (để watermark theo user). Fallback text nếu thiếu.
            plan = signals.get_plan(signal_id)
            if plan:
                txt = render_full(plan, uname, watermark=WATERMARK)
            else:
                full = signals.get_full(signal_id)
                if not full:
                    await update.message.reply_text("Xin lỗi, tín hiệu đã hết hạn cache.", quote=True)
                    return
                txt = full
            await update.message.reply_text(txt, parse_mode="HTML", protect_content=PROTECT_CONTENT)
        else:
            await upsell(update, context)
    elif payload.startswith("upgrade"):
        await upsell(update, context)
    else:
        await update.message.reply_text("Chào mừng! Dùng /latest để xem tín hiệu mới nhất (nếu có Plus), hoặc /help.")

# New: /latest – xem tín hiệu mới nhất theo tier
async def latest(update: Update, context: ContextTypes.DEFAULT_TYPE):
    sid = signals.get_latest_id()
    if not sid:
        await update.message.reply_text("Chưa có tín hiệu nào được đăng.")
        return
    uid = update.effective_user.id
    if users.is_plus_active(uid):
        uname = update.effective_user.username or ""
        plan = signals.get_plan(sid)
        if plan:
            txt = render_full(plan, uname, watermark=WATERMARK)
            await update.message.reply_text(txt, parse_mode="HTML", protect_content=PROTECT_CONTENT)
        else:
            await update.message.reply_text("Tín hiệu mới nhất đã hết hạn cache.")
    else:
        await upsell(update, context)

# New: /show <id> – xem theo id thủ công
async def show_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Dùng: /show <signal_id>")
        return
    sid = context.args[0]
    uid = update.effective_user.id
    if users.is_plus_active(uid):
        uname = update.effective_user.username or ""
        plan = signals.get_plan(sid)
        if plan:
            txt = render_full(plan, uname, watermark=WATERMARK)
            await update.message.reply_text(txt, parse_mode="HTML", protect_content=PROTECT_CONTENT)
        else:
            await update.message.reply_text("ID này đã hết hạn cache hoặc không tồn tại.")
    else:
        await upsell(update, context)

async def upsell(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uname = update.effective_user.username or "user"
    months = PLAN_DEFAULT_MONTHS
    text = (
        "Bạn chưa có Plus hoặc đã hết hạn.\n"
        f"• Quyền lợi: full Entry/SL/TP realtime, digest, cảnh báo TP/SL.\n"
        f"• Phí: vui lòng chuyển khoản thủ công.\n\n"
        f"<b>{BANK_INFO.get('name')}</b>\n"
        f"<b>Chủ TK:</b> {BANK_INFO.get('account_name')}\n"
        f"<b>Số TK:</b> {BANK_INFO.get('account_number')}\n"
        f"<b>Nội dung CK:</b> {BANK_INFO.get('note_format').format(username=uname, months=months)}\n\n"
        "Sau khi chuyển, bấm nút dưới để gửi xác nhận."
    )
    kb = InlineKeyboardMarkup([[InlineKeyboardButton("Đã chuyển xong", callback_data="paid")]])
    if update.message:
        try:
            qr_path = BANK_INFO.get("qr_image_path")
            if qr_path:
                await update.message.reply_photo(photo=InputFile(qr_path), caption=text, parse_mode="HTML", reply_markup=kb)
                return
        except Exception:
            pass
        await update.message.reply_text(text, parse_mode="HTML", reply_markup=kb)
    else:
        await update.callback_query.message.reply_text(text, parse_mode="HTML", reply_markup=kb)

async def on_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    data = q.data or ""
    if data == "paid":
        # Record a pending payment (admin will approve)
        payments.add(update.effective_user.id, amount=None, bank_ref=None, months=PLAN_DEFAULT_MONTHS, approved=False, admin_id=None)
        await q.answer("Đã ghi nhận. Admin sẽ duyệt trong ít phút.")
        await q.edit_message_reply_markup(None)

async def approve(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_owner(update.effective_user.id):
        return
    # /approve @username 30d  OR  /approve 123456789 2M
    try:
        args = update.message.text.split()[1:]
        target = args[0]
        months = 1
        if len(args) > 1:
            m = re.match(r"(\\d+)([mMdD])?", args[1])
            if m:
                val = int(m.group(1))
                unit = (m.group(2) or "M").upper()
                months = val if unit == "M" else max(1, val // 30)
        if target.startswith("@"):
            # find by username
            # In practice, you should map usernames to IDs. For simplicity, require numeric ID for now.
            await update.message.reply_text("Hãy dùng ID số (ví dụ: /approve 123456789 1M).")
            return
        uid = int(target)
        users.upsert(uid, months=months)
        await update.message.reply_text(f"Đã kích hoạt Plus cho {uid} trong {months} tháng.")
        try:
            await context.bot.send_message(chat_id=uid, text=f"Plus đã kích hoạt đến hạn sau {months} tháng. Bạn có thể bấm 'Xem full' ở teaser mới nhất.")
        except Exception:
            pass
    except Exception as e:
        await update.message.reply_text(f"Sai cú pháp. Dùng: /approve <telegram_id> <1M|30D>. Lỗi: {e}")

async def status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    u = users.get(uid)
    if not u or not u.get("expires_at"):
        await update.message.reply_text("Trạng thái: Free")
        return
    left = max(0, u["expires_at"] - int(__import__("time").time()))
    days = left // (24*3600)
    await update.message.reply_text(f"Trạng thái: Plus • còn {days} ngày.")

def run_bot():
    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("approve", approve))
    app.add_handler(CommandHandler("status", status_cmd))
    app.add_handler(CommandHandler("latest", latest))
    app.add_handler(CommandHandler("show", show_cmd))
    app.add_handler(CallbackQueryHandler(on_callback))
    app.run_polling()
    
if __name__ == "__main__":
    run_bot()
