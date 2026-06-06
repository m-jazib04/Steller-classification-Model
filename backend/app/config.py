"""Application configuration."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "Stellar Classification API"
    debug: bool = False
    cors_origins: str = "http://localhost:3000,https://*.vercel.app"
    model_dir: Path = Path(__file__).resolve().parents[2] / "model"
    log_level: str = "INFO"


settings = Settings()
