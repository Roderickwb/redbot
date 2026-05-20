from __future__ import annotations

import os

from fastapi import Header, HTTPException


def require_operator_token(x_operator_token: str = Header(default="")) -> None:
    """Require a token for write actions when OPERATOR_APP_TOKEN is configured."""
    expected = os.getenv("OPERATOR_APP_TOKEN", "").strip()
    if not expected:
        return
    if x_operator_token != expected:
        raise HTTPException(status_code=401, detail="Invalid operator token")
