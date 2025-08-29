
# Copy this to config.py and fill in values.
BOT_TOKEN = "123456789:ABC-your-telegram-bot-token"
CHANNEL_ID = -1001234567890        # Telegram Channel ID where teasers are posted (int)
OWNER_IDS = [123456789]            # Telegram user IDs allowed to use admin commands
BANK_INFO = {
    "name": "Vietcombank",
    "account_name": "TRAN KHANH HUNG",
    "account_number": "0123456789",
    "qr_image_path": "qr.png",     # optional, send photo if exists
    "note_format": "PLUS {username} {months}M"  # transfer message template
}
DATA_DIR = "./data"                # directory to store users/payments/cache
TEASER_SHOW_BUTTON = "Xem full"
TEASER_UPGRADE_BUTTON = "Nâng cấp Plus"
PLAN_DEFAULT_MONTHS = 1            # default duration for /approve when not specified
PROTECT_CONTENT = True             # protect full messages from forwarding
WATERMARK = True                   # append watermark to full messages
