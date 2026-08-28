"""JWT authentication utilities.

Uses HS256-based JWT tokens for authentication.  The bearer token is
extracted from the ``Authorization`` header via the ``HTTPBearer`` security
scheme, which is automatically registered in the OpenAPI documentation.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import cast

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.config import get_settings

settings = get_settings()
security = HTTPBearer(
    auto_error=False,
    description="JWT bearer token obtained from the authentication service.",
)


def create_access_token(subject: str, expires_delta: timedelta | None = None) -> str:
    """Create a signed JWT access token."""
    expire = datetime.now(UTC) + (
        expires_delta or timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    payload = {"sub": subject, "exp": expire}
    return jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


def decode_access_token(token: str) -> dict[str, object]:
    """Decode and validate a JWT access token."""
    try:
        return cast(
            dict[str, object],
            jwt.decode(
                token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM]
            ),
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
        )


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
) -> dict[str, object]:
    """FastAPI dependency that extracts and validates the current user from the JWT."""
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
        )
    return decode_access_token(credentials.credentials)


async def get_current_user_optional(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
) -> dict[str, object] | None:
    """FastAPI dependency that optionally extracts the current user.

    Returns None when no credentials are provided, instead of raising 401.
    Useful for endpoints that work for both authenticated and anonymous users.
    """
    if credentials is None:
        return None
    return decode_access_token(credentials.credentials)
