"""
System Configuration Settings Module for nse_swing_ai.
Loads settings from environment variables and provides typed, validated parameters.
"""

from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = Field(default=Path("./data"))
    CACHE_DIR: Path = Field(default=Path("./data/cache"))
    LOG_DIR: Path = Field(default=Path("./logs"))
    DATABASE_URL: str = Field(default="sqlite:///./data/nse_swing_ai.db")

    ACCOUNT_CAPITAL: float = Field(default=1000000.0)
    MAX_RISK_PER_TRADE_PCT: float = Field(default=1.0)
    MAX_PORTFOLIO_HEAT_PCT: float = Field(default=4.0)
    MAX_POSITIONS: int = Field(default=3)
    MAX_PICKS_PER_SCAN: int = Field(default=3)
    MIN_ADTV_CRORES: float = Field(default=5.0)
    MIN_STOCK_PRICE: float = Field(default=20.0)
    MAX_STOP_LOSS_PCT: float = Field(default=8.0)
    MIN_RR_TARGET_1: float = Field(default=1.8)
    MIN_RR_TARGET_2: float = Field(default=2.5)

    NSE_BASE_URL: str = Field(default="https://www.nseindia.com")
    CHARTINK_BASE_URL: str = Field(default="https://chartink.com")
    REQUEST_TIMEOUT_SECONDS: int = Field(default=20)
    USER_AGENT: str = Field(default="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/125.0.0.0 Safari/537.36")

    # Optional secondary evidence provider. Core scanning works with this disabled.
    INDIANAPI_API_KEY: str = Field(default="")
    INDIANAPI_BASE_URL: str = Field(default="https://analyst.indianapi.in")
    INDIANAPI_TIMEOUT_SECONDS: int = Field(default=15)
    INDIANAPI_ENABLED: bool = Field(default=False)

    STT_PCT: float = Field(default=0.1)
    BROKERAGE_MAX_RUPEES: float = Field(default=20.0)
    EXCHANGE_TURNOVER_PCT: float = Field(default=0.00345)
    GST_PCT: float = Field(default=18.0)
    STAMP_DUTY_PCT: float = Field(default=0.015)
    SLIPPAGE_BUFFER_PCT: float = Field(default=0.10)

    TELEGRAM_BOT_TOKEN: str = Field(default="")
    TELEGRAM_CHAT_ID: str = Field(default="")
    MODEL_VERSION: str = Field(default="v1.0.0")
    PROMPT_VERSION: str = Field(default="v1.0.0")
    LOG_LEVEL: str = Field(default="INFO")

    def ensure_directories(self) -> None:
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.LOG_DIR.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_directories()
