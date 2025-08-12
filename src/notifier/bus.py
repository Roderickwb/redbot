_notifier = None

def set_notifier(n):
    global _notifier
    _notifier = n

def send(msg: str):
    if _notifier:
        try:
            _notifier.safe_send(msg)
        except Exception:
            pass
