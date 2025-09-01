import os, uuid, logging, requests
from typing import Dict, Any
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from templates import render_teaser
from storage import JsonStore, SignalCache
from config import BOT_TOKEN, CHANNEL_ID, TEASER_SHOW_BUTTON, TEASER_UPGRADE_BUTTON, DATA_DIR

log = logging.getLogger("notifier")
logging.basicConfig(level=os.getenv("LOG_LEVEL","INFO"),
                    format="%(asctime)s %(levelname)s %(message)s")

API_BASE = f"https://api.telegram.org/bot{BOT_TOKEN}"

def _build_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=3, backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset(["GET", "POST"]),
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(pool_connections=64, pool_maxsize=64, max_retries=retry)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    s.headers.update({"Content-Type": "application/json"})
    return s

class TelegramNotifier:
    """Sync notifier using requests.Session (pool lớn + retry)."""

    def __init__(self):
        self.session = _build_session()
        self.cache = SignalCache(JsonStore(DATA_DIR))
        # Lấy username một lần
        try:
            r = self.session.get(f"{API_BASE}/getMe", timeout=10)
            r.raise_for_status()
            self.username = r.json().get("result", {}).get("username")
            if not self.username:
                raise RuntimeError("Bot username not found in getMe")
            log.info(f"TelegramNotifier ready as @{self.username} -> channel {CHANNEL_ID}")
        except Exception as e:
            raise RuntimeError(f"getMe failed: {e}") from e

    def post_teaser(self, plan: Dict[str, Any]) -> tuple[str, int]:
        if not self.username:
            raise RuntimeError("Bot username is not available")
        signal_id = str(uuid.uuid4())[:8]

        # Teaser cho channel + cache PLAN để DM bot render full có watermark
        teaser = render_teaser(plan)
        # Cache PLAN (bot sẽ render full + watermark từ plan)
        self.cache.put_plan(signal_id, plan)
        # Dùng https deep-link để client luôn gửi /start <payload>
        url_show = f"https://t.me/{self.username}?start=show_{signal_id}"
        url_upgr = f"https://t.me/{self.username}?start=upgrade"
        kb = {"inline_keyboard": [[{"text": TEASER_SHOW_BUTTON, "url": url_show}],
                                  [{"text": TEASER_UPGRADE_BUTTON, "url": url_upgr}]]}
        payload = {
            "chat_id": int(CHANNEL_ID),
            "text": teaser,
            "parse_mode": "HTML",
            "reply_markup": kb
        }
        try:
            r = self.session.post(f"{API_BASE}/sendMessage", json=payload, timeout=15)
            r.raise_for_status()
            msg_id = int(r.json()["result"]["message_id"])
        except requests.RequestException as e:
            log.warning("teaser post failed: %s", e)
            raise
        # Sau khi gửi thành công (ra khỏi try/except) mới log & return
        log.info(
            "Posted teaser signal_id=%s symbol=%s dir=%s",
            signal_id, plan.get("symbol"), plan.get("DIRECTION")
        )
        return signal_id, msg_id

    def send_channel(self, html: str):
        payload = {"chat_id": int(CHANNEL_ID), "text": html, "parse_mode": "HTML"}
        r = self.session.post(f"{API_BASE}/sendMessage", json=payload, timeout=15)
        r.raise_for_status()

    def send_channel_reply(self, reply_to_message_id: int, html: str):
        payload = {
            "chat_id": int(CHANNEL_ID),
            "text": html,
            "parse_mode": "HTML",
            "reply_to_message_id": int(reply_to_message_id),
            "allow_sending_without_reply": True
        }
        r = self.session.post(f"{API_BASE}/sendMessage", json=payload, timeout=15)
        r.raise_for_status()
      
    def send_dm(self, user_id: int, html: str):
        payload = {"chat_id": int(user_id), "text": html, "parse_mode": "HTML"}
        r = self.session.post(f"{API_BASE}/sendMessage", json=payload, timeout=15)
        r.raise_for_status()
