
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

class SignalPerfDB:
    def __init__(self, store: JsonStore):
        self.store = store
    def _all(self) -> dict:
        return self.store.read("trades")
    def _write(self, data: dict) -> None:
        self.store.write("trades", data)
    def open(self, sid: str, plan: dict) -> None:
        data = self._all()
        data[sid] = {
            "sid": sid, "symbol": plan.get("symbol"), "dir": plan.get("DIRECTION"),
            "entry": plan.get("entry"), "sl": plan.get("sl"),
            "tp1": plan.get("tp1") or plan.get("tp"), "tp2": plan.get("tp2"), "tp3": plan.get("tp3"),
            "posted_at": int(__import__("time").time()),
            "status": "OPEN", "hits": {}, "r_ladder": {
                "tp1": plan.get("rr1") or plan.get("rr"), "tp2": plan.get("rr2"), "tp3": plan.get("rr3")
            },
            "realized_R": 0.0, "close_reason": None
        }
        self._write(data)
    def by_symbol(self, symbol: str) -> list:
        return [t for t in self._all().values() if t.get("symbol")==symbol and t.get("status") in ("OPEN","TP1","TP2")]
    def set_hit(self, sid: str, level: str, R: float) -> dict:
        data = self._all(); t = data.get(sid, {})
        if not t: return {}
        t["hits"][level] = int(__import__("time").time())
        t["status"] = level.upper()
        t["realized_R"] = float(t.get("realized_R",0.0) + (R or 0.0))
        data[sid] = t; self._write(data); return t
    def close(self, sid: str, reason: str) -> dict:
        data = self._all(); t = data.get(sid, {})
        if not t: return {}
        t["status"] = "TP3" if reason=="TP3" else "SL"
        t["close_reason"] = reason
        data[sid] = t; self._write(data); return t
    def kpis(self, period: str="day") -> dict:
        # tính tổng hợp theo ngày/tuần từ trades.json (đơn giản: dựa theo posted_at)

class UserDB:
    def __init__(self, store: JsonStore):
        self.store = store

    def _now(self) -> int:
        return int(time.time())

    def list_all(self) -> dict:
        """Trả về dict {telegram_id: {...}}"""
        return self.store.read("users")

    def list_active(self) -> dict:
        now = self._now()
        users = self.store.read("users")
        return {uid: u for uid, u in users.items() if int(u.get("expires_at", 0)) > now}

    def get(self, telegram_id: int) -> dict:
        users = self.store.read("users")
        return users.get(str(telegram_id), {})

    def is_plus_active(self, telegram_id: int) -> bool:
        u = self.get(telegram_id)
        exp = int(u.get("expires_at", 0))
        return exp > self._now()

    def upsert(self, telegram_id: int, username: str | None = None, months: int = 1) -> dict:
        # gia hạn theo tháng (giữ nguyên logic cũ)
        users = self.store.read("users")
        key = str(telegram_id)
        now = self._now()
        delta = months * 30 * 24 * 3600
        if key in users and int(users[key].get("expires_at", 0)) > now:
            users[key]["expires_at"] = int(users[key]["expires_at"]) + delta
        else:
            users[key] = users.get(key, {})
            users[key]["expires_at"] = now + delta
        if username:
            users[key]["username"] = username
        users[key]["plan"] = "plus"
        self.store.write("users", users)
        return users[key]

    # ===== Cộng số ngày trực tiếp =====
    def extend_days(self, telegram_id: int, days: int) -> dict:
        users = self.store.read("users")
        key = str(telegram_id)
        now = self._now()
        delta = int(days) * 24 * 3600
        if key in users and int(users[key].get("expires_at", 0)) > now:
            users[key]["expires_at"] = int(users[key]["expires_at"]) + delta
        else:
            username = users.get(key, {}).get("username")
            users[key] = {"username": username, "created_at": now}
            users[key]["expires_at"] = now + delta
        users[key]["plan"] = "plus"
        self.store.write("users", users)
        return users[key]


    def upsert(self, telegram_id: int, username: str | None = None, months: int = 1) -> dict:
        # (giữ nguyên logic cũ của bạn – không sửa ở đây nếu đã ổn)
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

    # ===== New: cộng số ngày trực tiếp =====
    def extend_days(self, telegram_id: int, days: int):
        users = self.store.read("users")
        key = str(telegram_id)
        now = self._now()
        delta = int(days) * 24 * 3600
        if key in users and int(users[key].get("expires_at", 0)) > now:
            users[key]["expires_at"] = int(users[key]["expires_at"]) + delta
        else:
            users[key] = {
                "username": users.get(key, {}).get("username"),
                "created_at": users.get(key, {}).get("created_at", now),
                "expires_at": now + delta
            }
        self.store.write("users", users)

    # ===== New: thu hồi ngay =====
    def revoke(self, telegram_id: int):
        users = self.store.read("users")
        key = str(telegram_id)
        if key in users:
            users[key]["expires_at"] = 0
            self.store.write("users", users)
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

    # thêm tham số order_id để truy vết
    def add(
        self,
        telegram_id: int,
        amount: int | None,
        bank_ref: str | None,
        months: int = 1,
        approved: bool = False,
        admin_id: int | None = None,
        order_id: str | None = None,
    ) -> str:
        payments = self.store.read("payments")
        pid = str(int(time.time())) + "-" + str(telegram_id)
        payments[pid] = {
            "telegram_id": telegram_id,
            "amount": amount,
            "bank_ref": bank_ref,
            "months": months,
            "approved": approved,
            "admin_id": admin_id,
            "order_id": order_id,
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

    # New: store/retrieve the raw plan to render watermark theo user
    def put_plan(self, signal_id: str, plan: dict):
        data = self.store.read("signals")
        data[signal_id] = {**data.get(signal_id, {}), "plan": plan, "ts": int(time.time())}
        data["_latest_id"] = signal_id
        self.store.write("signals", data)

    def get_plan(self, signal_id: str) -> dict | None:
        data = self.store.read("signals")
        s = data.get(signal_id)
        return s.get("plan") if s else None

    def get_latest_id(self) -> str | None:
       return self.store.read("signals").get("_latest_id")

    def get_plan(self, signal_id: str) -> dict | None:
       data = self.store.read("signals")
       s = data.get(signal_id)
       return s.get("plan") if s else None
