"""
Telegram Bot Notifier Module.
Handles HTTP API dispatch of markdown-formatted trade alerts and daily ledger updates to Telegram channels/chats.
"""

import logging
import os
from typing import Any
import httpx

from config.settings import settings
from src.shadow.alerts import TelegramFormatter

logger = logging.getLogger(__name__)


class TelegramBotNotifier:
    """Dispatches trade alerts and daily performance summaries via Telegram Bot API."""

    def __init__(self, bot_token: str | None = None, chat_id: str | None = None):
        self.bot_token = bot_token or os.environ.get("TELEGRAM_BOT_TOKEN") or settings.TELEGRAM_BOT_TOKEN
        self.chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID") or settings.TELEGRAM_CHAT_ID

        if self.is_configured:
            logger.info(f"Telegram Bot initialized for Chat ID: {str(self.chat_id)[:4]}***")
        else:
            logger.warning("Telegram Bot NOT configured (missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID).")

    @property
    def is_configured(self) -> bool:
        """Returns True if bot token and chat ID are provided."""
        token_valid = bool(self.bot_token and str(self.bot_token).strip() != "")
        chat_valid = bool(self.chat_id and str(self.chat_id).strip() != "")
        return token_valid and chat_valid

    async def send_message(self, text: str, parse_mode: str = "Markdown") -> bool:
        """Sends a text message to the configured Telegram chat."""
        if not self.is_configured:
            logger.warning("Telegram secrets not detected. Skipping Telegram alert.")
            logger.info("Message preview:\n" + text)
            return False

        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": parse_mode,
            "disable_web_page_preview": True,
        }

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.post(url, json=payload)
                if resp.status_code == 200:
                    logger.info("✅ Successfully dispatched Telegram alert!")
                    return True
                else:
                    logger.error(f"❌ Telegram API returned HTTP {resp.status_code}: {resp.text}")
                    return False
        except Exception as e:
            logger.error(f"❌ Failed to send Telegram message due to exception: {e}")
            return False

    async def dispatch_recommendations(self, recs: list[Any], market_regime: str) -> None:
        """Dispatches summary alert followed by detailed dossiers for each trade."""
        summary = TelegramFormatter.format_scan_summary(recs, market_regime)
        await self.send_message(summary)

        for rec in recs:
            dossier = TelegramFormatter.format_recommendation(rec)
            await self.send_message(dossier)

    async def dispatch_ledger_update(self, update_summary: str) -> None:
        """Dispatches daily shadow ledger status update."""
        msg = f"📊 *NSE SWING AI — DAILY TRADE LEDGER UPDATE*\n━━━━━━━━━━━━━━━━━━━━━━━━━\n{update_summary}"
        await self.send_message(msg)
