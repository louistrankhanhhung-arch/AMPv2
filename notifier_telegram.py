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
    retry = Retry(total=3, backoff_factor=0.5,
                  status_forcelist=[429,500,502,503,504],
                  allowed_methods=frozenset(["GET","POST"]),
                  respect_retry_after_header=True)
    adapter = HTTPAdapter(pool_connections=64, pool_maxsize=64, max_retries=retry)
    s.mount("https://", adapter); s.mount("http://", adapter)
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
            log.warning(f"getMe failed: {e}")
            self.username = None

    def post_teaser(self, plan: Dict[str, Any]) -> str:
        signal_id = str(uuid.uuid4())[:8]
        teaser = render_teaser(plan)
        # cache PLAN để khi user xem full, bot render kèm watermark cá nhân
        self.cache.put_plan(signal_id, plan)
        kb = {
            "inline_keyboard": [
                [{"text": TEASER_SHOW_BUTTON, "url": f"https://t.me/{self.username}?start=show_{signal_id}"}],
                [{"text": TEASER_UPGRADE_BUTTON, "url": f"https://t.me/{self.username}?start=upgrade"}],
            ]
        }
        payload = {"chat_id": CHANNEL_ID, "text": teaser, "parse_mode": "HTML", "reply_markup": kb}
        r = self.session.post(f"{API_BASE}/sendMessage", json=payload, timeout=15)
        r.raise_for_status()
        log.info(f"Posted teaser signal_id={signal_id} symbol={plan.get('symbol')} dir={plan.get('DIRECTION')}")
        return signal_id
        except Exception as e:
            log.warning(f"teaser post failed: {e}")
            raise
