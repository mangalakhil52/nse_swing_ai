from src.shadow.telegram import TelegramSender


def test_telegram_sender_is_optional_without_credentials(monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    sender = TelegramSender()
    assert sender.configured is False
    assert sender.send("test", required=False) is False


def test_telegram_sender_requires_credentials_when_required(monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    sender = TelegramSender()
    try:
        sender.send("test", required=True)
    except Exception as exc:
        assert "TELEGRAM_BOT_TOKEN" in str(exc)
    else:
        raise AssertionError("required Telegram delivery should fail without credentials")
