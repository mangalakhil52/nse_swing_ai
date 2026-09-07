"""Dependency-free Telegram Bot API delivery for paper/shadow alerts."""
from __future__ import annotations

import json
import logging
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)


class TelegramDeliveryError(RuntimeError):
    """Raised when required Telegram delivery fails."""


class TelegramSender:
    def __init__(self, token: str | None = None, chat_id: str | None = None, timeout: int = 20):
        self.token = token or os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID")
        self.timeout = timeout

    @property
    def configured(self) -> bool:
        return bool(self.token and self.chat_id)

    def send(self, text: str, *, required: bool = False) -> bool:
        if not self.configured:
            msg = "Telegram is not configured; set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID."
            if required:
                raise TelegramDeliveryError(msg)
            logger.info(msg)
            return False
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        payload = json.dumps({
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": "Markdown",
            "disable_web_page_preview": True,
        }).encode("utf-8")
        request = Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
            if not body.get("ok"):
                raise TelegramDeliveryError(f"Telegram API rejected message: {body}")
            logger.info("Telegram alert delivered successfully")
            return True
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            if required:
                raise TelegramDeliveryError(f"Telegram delivery failed: {exc}") from exc
            logger.exception("Telegram delivery failed")
            return False
