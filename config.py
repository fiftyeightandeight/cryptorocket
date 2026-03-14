import os
from pathlib import Path

from hyperliquid.utils.constants import MAINNET_API_URL, TESTNET_API_URL

# Wallet credentials (from environment variables)
WALLET_ADDRESS = os.environ.get("HYPERLIQUID_WALLET_ADDRESS")
WALLET_PRIVATE_KEY = os.environ.get("HYPERLIQUID_PRIVATE_KEY")

# Optional: API wallet (recommended for bots — can trade but not withdraw)
API_WALLET_KEY = os.environ.get("HYPERLIQUID_API_WALLET_KEY")

# Network
USE_TESTNET = os.environ.get("HYPERLIQUID_TESTNET", "false").lower() == "true"
BASE_URL = TESTNET_API_URL if USE_TESTNET else MAINNET_API_URL

# Database
DB_PATH = Path(os.environ.get("CRYPTO_DB_PATH", Path(__file__).parent / "data" / "crypto.duckdb"))

# Data collection defaults
DEFAULT_INTERVALS = ["1h", "1d"]
DEFAULT_LOOKBACK_DAYS = 365 * 2  # 2 years of history
CANDLE_LIMIT = 5000  # Max candles per API request
