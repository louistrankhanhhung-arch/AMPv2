
import os, asyncio, uuid
from typing import Dict, Any
from templates import render_teaser, render_full
from storage import JsonStore, SignalCache
from config import BOT_TOKEN, CHANNEL_ID, TEASER_SHOW_BUTTON, TEASER_UPGRADE_BUTTON, DATA_DIR
from telegram import Bot, InlineKeyboardMarkup, InlineKeyboardButton

class TelegramNotifier:
    """
    Usage:
        tn = TelegramNotifier()
        signal_id = await tn.post_teaser(plan)  # returns signal_id cached for bot DM
    """
    def __init__(self):
        self.bot = Bot(BOT_TOKEN)
        self.cache = SignalCache(JsonStore(DATA_DIR))

    async def post_teaser(self, plan: Dict[str, Any]) -> str:
        signal_id = str(uuid.uuid4())[:8]
        teaser = render_teaser(plan)
        # cache full text for bot DM
        full = render_full(plan, username=None, watermark=False)
        self.cache.put_full(signal_id, full)

        kb = InlineKeyboardMarkup([
            [InlineKeyboardButton(TEASER_SHOW_BUTTON, url=f"https://t.me/{(await self.bot.get_me()).username}?start=show_{signal_id}")],
            [InlineKeyboardButton(TEASER_UPGRADE_BUTTON, url=f"https://t.me/{(await self.bot.get_me()).username}?start=upgrade")]
        ])
        await self.bot.send_message(chat_id=CHANNEL_ID, text=teaser, parse_mode="HTML", reply_markup=kb)
        return signal_id
