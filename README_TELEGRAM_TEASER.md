
# Telegram Teaser Integration

This package adds the "teaser in channel → full in bot DM" flow.

## Files
- `config_example.py` → copy to `config.py` and fill in.
- `storage.py` → JSON-based store (users, payments, signals cache).
- `templates.py` → HTML renderers for teaser/full.
- `notifier_telegram.py` → async sender to channel with inline buttons; caches full text.
- `bot_telegram.py` → DM bot: shows full if Plus active; admin `/approve` for manual payments.

## Hook into your scanning pipeline
In your code after you compute a plan with `DECISION=ENTER`, call `TelegramNotifier.post_teaser(plan)`.

Example patch for your main loop:
```python
from notifier_telegram import TelegramNotifier
tn = TelegramNotifier()

# inside your per-signal loop:
if plan.get("DECISION") == "ENTER":
    await tn.post_teaser({**plan, "symbol": sym})
```

## Run
1. `cp config_example.py config.py` and edit values.
2. Run bot: `python bot_telegram.py`
3. From your scanner process (async or via `asyncio.run`), call `TelegramNotifier.post_teaser` on ENTER events.

## Notes
- Full messages are sent in DM with `protect_content=True` and optional watermark.
- Manual bank transfers: users click "Đã chuyển xong"; admin runs `/approve <telegram_id> 1M`.
