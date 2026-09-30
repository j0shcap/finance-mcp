"""Runtime configuration loaded from environment variables (prefix FINANCE_MCP_)."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Server configuration. All fields have safe defaults so zero-config works.

    Environment only, deliberately: no ``env_file``. MCP clients launch the server from
    an arbitrary working directory, so a relative ``.env`` could be missing or belong to
    another project; every MCP client config can set the server's environment instead.
    """

    model_config = SettingsConfigDict(env_prefix="FINANCE_MCP_")

    quote_cache_ttl_seconds: int = Field(default=30, ge=0)
    history_cache_ttl_seconds: int = Field(default=300, ge=0)
    fundamentals_cache_ttl_seconds: int = Field(default=3600, ge=0)
    #: Most bars returned by get_price_history before truncation (the summary still
    #: covers the whole window). Bounds the response size a single call can produce.
    max_history_bars: int = Field(default=260, gt=0, le=10_000)


def get_settings() -> Settings:
    """Return the loaded settings instance."""
    return Settings()
