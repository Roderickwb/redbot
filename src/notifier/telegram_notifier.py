import requests

class TelegramNotifier:
    def __init__(self, bot_token: str, chat_id: str, timeout=6):
        self.api = f"https://api.telegram.org/bot{bot_token}"
        self.chat_id = chat_id
        self.timeout = timeout

    def safe_send(self, text: str):
        try:
            requests.post(
                f"{self.api}/sendMessage",
                json={"chat_id": self.chat_id, "text": text},
                timeout=self.timeout
            )
        except Exception:
            pass  # never crash the bot for a notification
