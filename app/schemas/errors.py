from pydantic import BaseModel


class ErrorResponse(BaseModel):
    """Standard error response body."""

    detail: str
    code: str | None = None
