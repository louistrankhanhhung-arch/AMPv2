# fb_notifier.py
import os, logging, re, html
import requests

log = logging.getLogger("fb")

def _strip_html(text: str) -> str:
    """Chuyển nội dung HTML (Telegram style) về plain text cho Facebook."""
    if not isinstance(text, str):
        return ""
    t = text.replace("<br>", "\n").replace("<br/>", "\n").replace("<br />", "\n")
    t = re.sub(r"</?(b|strong)>", "", t, flags=re.I)
    t = re.sub(r"</?(i|em)>", "", t, flags=re.I)
    t = re.sub(r"</?(u|s|strike|code|pre)>", "", t, flags=re.I)
    t = re.sub(r"<a[^>]*>(.*?)</a>", r"\1", t, flags=re.I | re.S)
    t = re.sub(r"<[^>]+>", "", t)
    t = html.unescape(t)
    # Gọn gàng xuống dòng kép
    t = re.sub(r"\n{3,}", "\n\n", t).strip()
    return t

class FBNotifier:
    def __init__(self):
        self.page_id = os.getenv("FB_PAGE_ID", "").strip()
        self.token   = os.getenv("FB_PAGE_TOKEN", "").strip()
        self.enabled = bool(self.page_id and self.token and os.getenv("FB_ENABLED", "1") != "0")
        if not self.enabled:
            log.warning("FBNotifier disabled (missing FB_PAGE_ID/FB_PAGE_TOKEN or FB_ENABLED=0)")

    def _post(self, endpoint: str, data: dict) -> bool:
        if not self.enabled:
            return False
        url = f"https://graph.facebook.com/v23.0/{self.page_id}/{endpoint.lstrip('/')}"
        try:
            data = dict(data or {})
            data["access_token"] = self.token
            r = requests.post(url, data=data, timeout=20)
            if r.status_code != 200:
                log.warning("FB post failed (%s): %s", r.status_code, r.text[:400])
                return False
            log.info("FB post ok: %s", r.text[:200])
            return True
        except Exception as e:
            log.warning("FB post exception: %s", e)
            return False

    def post_text(self, text: str) -> bool:
        """Đăng bài text (tự strip HTML)."""
        if not text:
            return False
        msg = _strip_html(text)
        if not msg:
            return False
        return self._post("feed", {"message": msg})

    def post_photo(self, image_url: str, caption: str = "") -> bool:
        """Đăng ảnh có caption (tự strip HTML)."""
        if not (image_url and isinstance(image_url, str)):
            return False
        cap = _strip_html(caption or "")
        return self._post("photos", {"url": image_url, "caption": cap})

    # Helpers cho app
    def post_teaser(self, teaser_html: str) -> bool:
        return self.post_text(teaser_html)

    def post_kpi_24h(self, kpi_html: str) -> bool:
        return self.post_text(kpi_html)

    def post_kpi_week(self, kpi_html: str) -> bool:
        return self.post_text(kpi_html)
