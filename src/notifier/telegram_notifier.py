import os

import requests
from dotenv import load_dotenv


class TelegramNotifier:
    def __init__(self, bot_token: str, chat_id: str, timeout=6):
        self.api = f"https://api.telegram.org/bot{bot_token}"
        self.chat_id = chat_id
        self.timeout = timeout

    def safe_send(self, text: str):
        try:
            resp = requests.post(
                f"{self.api}/sendMessage",
                json={"chat_id": self.chat_id, "text": text},
                timeout=self.timeout,
            )

            if resp.status_code != 200:
                print("[TelegramNotifier] Error:", resp.status_code, resp.text)
            else:
                print("[TelegramNotifier] Message sent OK")
        except Exception as e:
            print("[TelegramNotifier] Exception while sending message:", e)


if __name__ == "__main__":
    load_dotenv()
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")

    if not bot_token or not chat_id:
        raise RuntimeError("Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env before testing Telegram.")

    notifier = TelegramNotifier(bot_token, chat_id)
    notifier.safe_send("Telegram testbericht vanaf deze runtime.")
