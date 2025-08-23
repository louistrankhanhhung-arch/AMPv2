# universe.py
import os

DEFAULT_UNIVERSE = [
    "AAVE/USDT","ADA/USDT","APT/USDT","ARB/USDT","ATOM/USDT","AVAX/USDT","BNB/USDT","BTC/USDT",
    "DOT/USDT","DYDX/USDT","ETH/USDT","FET/USDT","FIL/USDT","GRT/USDT","ICP/USDT","IMX/USDT",
    "INJ/USDT","LDO/USDT","LINK/USDT","NEAR/USDT","OP/USDT","PENDLE/USDT","SOL/USDT","SUI/USDT",
    "TON/USDT","TRX/USDT","UNI/USDT","XRP/USDT","SEI/USDT",
]

def _parse_csv(s: str):
    return [x.strip().upper() for x in (s or "").split(",") if x.strip()]

def get_universe_from_env():
    env = os.getenv("SYMBOLS", "")
    lst = _parse_csv(env)
    return lst if lst else DEFAULT_UNIVERSE[:]

def resolve_symbols(symbols_param: str):
    """
    Ưu tiên query param (nếu có), sau đó ENV SYMBOLS, cuối cùng là DEFAULT_UNIVERSE.
    """
    q = _parse_csv(symbols_param)
    return q if q else get_universe_from_env()
