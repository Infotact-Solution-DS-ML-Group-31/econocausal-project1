"""Application settings and logging configuration for EconoCausal API."""

import logging
import sys
from functools import lru_cache
from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables or .env file."""

    app_name: str = "EconoCausal API"
    app_version: str = "0.1.0"
    environment: Literal["development", "staging", "production"] = "development"
    debug: bool = True

    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000

    # Optimization solver defaults
    default_campaign_budget: float = 10000.0
    max_campaign_budget: float = 1_000_000.0
    solver_timeout_seconds: float = 30.0

    # Logging settings
    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


@lru_cache()
def get_settings() -> Settings:
    """Cached singleton instance of application settings."""
    return Settings()


def setup_logging(settings: Settings) -> None:
    """Configure structured logging using Python standard library logging."""
    log_format = "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s"
    level = getattr(logging, settings.log_level.upper(), logging.INFO)

    logging.basicConfig(
        level=level,
        format=log_format,
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )


# Initialize global configuration instance
settings = get_settings()
setup_logging(settings)
logger = logging.getLogger("econocausal")