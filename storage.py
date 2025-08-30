
import os, json, time, datetime, threading
from typing import Optional, Dict, Any, List

class JsonStore:
    def __init__(self, data_dir: str = "./data"):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        self._locks = {}

    def _path(self, name: str) -> str:
        return os.path.join(self.data_dir, name + ".json")

    def _lock(self, name: str):
        if name not in self._locks:
            self._locks[name] = threading.Lock()
        return self._locks[name]

    def read(self, name: str) -> dict:
        path = self._path(name)
        if not os.path.exists(path):
            return {}
        with self._lock(name):
            with open(path, "r", encoding="utf-8") as f:
                try:
                    return json.load(f)
                except Exception:
                    return {}

    def write(self, name: str, data: dict) -> None:
        path = self._path(name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with self._lock(name):
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            os.replace(tmp, path)

class UserDB:
    def __init__(self, store: JsonStore):
        self.store = store

    def _now(self):
        return int(time.time())

    def get(self, telegram_id: int) -> dict:
        users = self.store.read("users")
        return users.get(str(telegram_id), {})

    def is_plus_active(self, telegram_id: int) -> bool:
        u = self.get(telegram_id)
        exp = int(u.get("expires_at", 0))
        return exp > self._now()

    def upsert(self, telegram_id: int, username: str | None = None, months: int = 1) -> dict:
        users = self.store.read("users")
        key = str(telegram_id)
        now = self._now()
        delta = months * 30 * 24 * 3600
        if key in users and int(users[key].get("expires_at", 0)) > now:
            users[key]["expires_at"] += delta
        else:
            users[key] = users.get(key, {})
            users[key]["expires_at"] = now + delta
        if username:
            users[key]["username"] = username
        users[key]["plan"] = "plus"
        self.store.write("users", users)
        return users[key]

    def set_expiry(self, telegram_id: int, ts: int) -> None:
        users = self.store.read("users")
        key = str(telegram_id)
        u = users.get(key, {})
        u["expires_at"] = ts
        users[key] = u
        self.store.write("users", users)

class PaymentDB:
    def __init__(self, store: JsonStore):
        self.store = store

    def add(self, telegram_id: int, amount: int | None, bank_ref: str | None, months: int = 1, approved: bool = False, admin_id: int | None = None) -> str:
        payments = self.store.read("payments")
        pid = str(int(time.time())) + "-" + str(telegram_id)
        payments[pid] = {
            "telegram_id": telegram_id,
            "amount": amount,
            "bank_ref": bank_ref,
            "months": months,
            "approved": approved,
            "admin_id": admin_id,
            "created_at": int(time.time())
        }
        self.store.write("payments", payments)
        return pid

    def approve(self, payment_id: str, admin_id: int):
        payments = self.store.read("payments")
        if payment_id in payments:
            payments[payment_id]["approved"] = True
            payments[payment_id]["admin_id"] = admin_id
            self.store.write("payments", payments)

class SignalCache:
    def __init__(self, store: JsonStore):
        self.store = store

    def put_full(self, signal_id: str, text: str):
        data = self.store.read("signals")
        data[signal_id] = {"text": text, "ts": int(time.time())}
        self.store.write("signals", data)

    def get_full(self, signal_id: str) -> str | None:
        data = self.store.read("signals")
        s = data.get(signal_id)
        return s.get("text") if s else None

    # New: store plan dict for watermark rendering per user
    def put_plan(self, signal_id: str, plan: dict):
        data = self.store.read("signals")
        data[signal_id] = {**data.get(signal_id, {}), "plan": plan, "ts": int(time.time())}
        self.store.write("signals", data)

    def get_plan(self, signal_id: str) -> dict | None:
        data = self.store.read("signals")
        s = data.get(signal_id)
        return s.get("plan") if s else None
