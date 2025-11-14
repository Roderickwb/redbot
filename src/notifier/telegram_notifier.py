import requests

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
                timeout=self.timeout
            )

            if resp.status_code != 200:
                print("[TelegramNotifier] Error:",
                      resp.status_code,
                      resp.text)
            else:
                print("[TelegramNotifier] Message sent OK")
        except Exception as e:
            # niet crashen, maar WEL loggen
            print("[TelegramNotifier] Exception while sending message:", e)


# ========= Alleen voor test vanaf je laptop =========
if __name__ == "__main__":
    BOT_TOKEN = "8428736345:AAEwagLlw2cpNnGymbLESPCcm1vXu4u8Ios"
    CHAT_ID = "8256312700"

    notifier = TelegramNotifier(BOT_TOKEN, CHAT_ID)
    notifier.safe_send("Testbericht vanaf mijn laptop ✅")

