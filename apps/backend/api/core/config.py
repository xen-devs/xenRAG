import os
from pathlib import Path
from dotenv import load_dotenv

_backend_root = Path(__file__).resolve().parents[2]
load_dotenv(_backend_root / ".env")


def get_required_env(key: str) -> str:
    value = os.getenv(key)
    if not value:
        raise RuntimeError(f"{key} environment variable must be set")
    return value


# Database
DATABASE_URL = get_required_env("DATABASE_URL")

# JWT
JWT_SECRET_KEY = get_required_env("JWT_SECRET_KEY")

_ALLOWED_JWT_ALGORITHMS = {"HS256", "HS384", "HS512"}
_jwt_alg = os.getenv("JWT_ALGORITHM", "HS256")
if _jwt_alg not in _ALLOWED_JWT_ALGORITHMS:
    raise RuntimeError(
        f"JWT_ALGORITHM '{_jwt_alg}' is not allowed. "
        f"Must be one of: {', '.join(sorted(_ALLOWED_JWT_ALGORITHMS))}"
    )
JWT_ALGORITHM = _jwt_alg

JWT_EXPIRY_DAYS = int(os.getenv("JWT_EXPIRY", "7"))

# CORS
CORS_ORIGINS_STR = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:5173,http://localhost:3000",
)
CORS_ORIGINS = [origin.strip() for origin in CORS_ORIGINS_STR.split(",") if origin.strip()]
