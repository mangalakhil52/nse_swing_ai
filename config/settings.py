"""
System Configuration Settings Module for nse_swing_ai.
Loads settings from environment variables and provides typed, validated parameters.
"""

from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Base Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = Field(default=Path("./data"))
    CACHE_DIR: Path = Field(default=Path("./data/cache"))
    LOG_DIR: Path = Field(default=Path("./logs"))

    # Database
    DATABASE_URL: str = Field(
        default="sqlite:///./data/nse_swing_ai.db",
        description="SQLAlchemy database connection string (default SQLite)"
    )

    # Trading & Capital Parameters
    ACCOUNT_CAPITAL: float = Field(default=1000000.0, description="Base trading capital in INR")
    MAX_RISK_PER_TRADE_PCT: float = Field(default=1.0, description="Max risk percentage per trade (e.g. 1.0 = 1%)")
    MAX_PORTFOLIO_HEAT_PCT: float = Field(default=4.0, description="Max cumulative open risk across all positions")
    MAX_POSITIONS: int = Field(default=3, description="Maximum concurrent swing positions")
    MAX_PICKS_PER_SCAN: int = Field(default=3, description="Maximum actionable recommendations per daily run")
    MIN_ADTV_CRORES: float = Field(default=5.0, description="Minimum 20-day Average Daily Traded Value in Crores")
    MIN_STOCK_PRICE: float = Field(default=20.0, description="Minimum stock price in INR to avoid penny stocks")
    MAX_STOP_LOSS_PCT: float = Field(default=8.0, description="Maximum allowable structural stop loss percentage")
    MIN_RR_TARGET_1: float = Field(default=1.8, description="Minimum Risk-to-Reward ratio for Target 1")
    MIN_RR_TARGET_2: float = Field(default=2.5, description="Minimum Risk-to-Reward ratio for Target 2")

    # Data Ingestion & Network
    NSE_BASE_URL: str = Field(default="https://www.nseindia.com")
    CHARTINK_BASE_URL: str = Field(default="https://chartink.com")
    REQUEST_TIMEOUT_SECONDS: int = Field(default=20)
    USER_AGENT: str = Field(
        default="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
    )

    # Indian Tax & Transaction Costs (Net Friction Model)
    STT_PCT: float = Field(default=0.1, description="Securities Transaction Tax on delivery Buy & Sell (%)")
    BROKERAGE_MAX_RUPEES: float = Field(default=20.0, description="Max flat brokerage per trade in INR")
    EXCHANGE_TURNOVER_PCT: float = Field(default=0.00345, description="NSE exchange turnover fee (%)")
    GST_PCT: float = Field(default=18.0, description="GST on brokerage and exchange turnover fees (%)")
    STAMP_DUTY_PCT: float = Field(default=0.015, description="Stamp duty on buy orders (%)")
    SLIPPAGE_BUFFER_PCT: float = Field(default=0.10, description="Slippage buffer on entry and exit (%)")

    # Notification
    TELEGRAM_BOT_TOKEN: str = Field(default="")
    TELEGRAM_CHAT_ID: str = Field(default="")

    # Versioning & Telemetry
    MODEL_VERSION: str = Field(default="v1.0.0")
    PROMPT_VERSION: str = Field(default="v1.0.0")
    LOG_LEVEL: str = Field(default="INFO")

    def ensure_directories(self) -> None:
        """Ensure necessary runtime directories exist."""
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.LOG_DIR.mkdir(parents=True, exist_ok=True)


# Global settings singleton
settings = Settings()
settings.ensure_directories()
