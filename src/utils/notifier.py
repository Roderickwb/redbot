import os, requests, logging

class Notifier:
    def __init__(self, enabled: bool, chat_id: str, token_env: str = "TELEGRAM_BOT_TOKEN", timeout=5):
        self.enabled = enabled
        self.chat_id = str(chat_id)
        self.token = os.getenv(token_env, "")
        self.timeout = timeout
        self.base = f"https://api.telegram.org/bot{self.token}" if self.token else None
        self.log = logging.getLogger("notifier")

    def send(self, text: str):
        if not (self.enabled and self.base):
            return False
        try:
            r = requests.post(f"{self.base}/sendMessage",
                              json={"chat_id": self.chat_id, "text": text, "disable_web_page_preview": True},
                              timeout=self.timeout)
            ok = r.ok and r.json().get("ok", False)
            if not ok:
                self.log.warning("Telegram send failed: %s", r.text)
            return ok
        except Exception as e:
            self.log.warning("Telegram send error: %s", e)
            return False

    # convenience helpers
    def open(self, symbol, side, price, atr):
        return self.send(f"🟢 OPEN {symbol} {side.upper()} @ {float(price):.4f} | ATR {float(atr):.4f}")

    def partial(self, symbol, portion, price, reason):
        p = float(portion)*100
        return self.send(f"🟡 PARTIAL {symbol} {p:.0f}% @ {float(price):.4f} ({reason})")

    def closed(self, symbol, reason, price):
        return self.send(f"🔴 CLOSED {symbol} @ {float(price):.4f} [{reason}]")

    def meltdown(self, active: bool):
        return self.send("🚨 Meltdown ACTIVE — new entries paused") if active else self.send("✅ Meltdown CLEARED")
