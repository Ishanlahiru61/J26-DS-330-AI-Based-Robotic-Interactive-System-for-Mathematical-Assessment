"""Application configuration boundary."""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    app_env: str = os.getenv("APP_ENV", "development")
    log_level: str = os.getenv("APP_LOG_LEVEL", "INFO")
    database_url: str = os.getenv(
        "DATABASE_URL", "sqlite:///./dyscalculia_screening.db"
    )
    screening_thresholds_validated: bool = (
        os.getenv("SCREENING_THRESHOLDS_VALIDATED", "false").lower() == "true"
    )


settings = Settings()
