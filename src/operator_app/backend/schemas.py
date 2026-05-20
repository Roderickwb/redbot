from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


OperatorAction = Literal["approve", "reject", "wait", "freeze", "snooze", "note"]


class DecisionRequest(BaseModel):
    source_id: str = Field(..., min_length=1)
    source_type: str = "recommendation"
    action: OperatorAction
    operator: str = "mobile_app"
    reason: str = ""
    scope: str = "recommendation"
    source_path: str = ""
    expires_utc: str = ""


class ApiError(BaseModel):
    error: str
