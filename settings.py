import json
import os
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class OpenAPISettings:
    # Credentials - preferentially loaded from environment variables
    client_id: Optional[str] = None
    client_secret: Optional[str] = None

    # Connection type: "demo" or "live". This will be used with OpenApiPy's EndPoints.
    host_type: str = "demo"

    # Optional: cTrader Account ID (long integer, but often represented as string in configs)
    # This is the ID of the trading account you want to authorize for trading operations.
    # The library will likely require this for calls like GetTrader, SubscribeSpots etc.
    # This is NOT the same as client_id (which is for the application).
    default_ctid_trader_account_id: Optional[int] = None # Store as int if it's numeric

    # OAuth2 specific URLs
    spotware_auth_url: str = "https://connect.spotware.com/oauth/v2/auth" # Standard URL
    spotware_token_url: str = "https://connect.spotware.com/oauth/v2/token" # Standard URL
    redirect_uri: str = "http://localhost:5000/callback" # As specified


@dataclass
class GeneralSettings:
    default_symbol: str = "EUR/USD"
    chart_update_interval_ms: int = 500
    min_bars_for_trading: int = 50
    risk_percentage: float = 1.0
    batch_profit_target: float = 10.0
    # Add other general app settings here if any

@dataclass
class AISettings:
    """Settings for the AI Overseer integration."""
    use_ai_overseer: bool = False
    advisor_url: Optional[str] = None
    advisor_auth_token: Optional[str] = None
    advisor_timeout_ms: int = 7000
    advisor_min_confidence: float = 0.65

@dataclass
class StrategySettings:
    """Settings for the trading strategy."""
    ema_period: int = 50
    atr_period: int = 14
    adx_period: int = 14
    rsi_period: int = 14
    adx_threshold: int = 25
    rsi_overbought: int = 70
    rsi_oversold: int = 30
    stop_mult: float = 1.0
    target_mult: float = 0.5
    buffer_mult: float = 0.05

@dataclass
class Settings:
    openapi: OpenAPISettings
    general: GeneralSettings
    ai: AISettings
    strategy: StrategySettings

    @staticmethod
    def load(path: str = "config.json") -> "Settings":
        # Load secrets from environment variables first
        env_client_id = os.environ.get("CTRADER_CLIENT_ID")
        env_client_secret = os.environ.get("CTRADER_CLIENT_SECRET")

        try:
            with open(path, 'r') as f:
                cfg_data = json.load(f)
        except FileNotFoundError:
            print(f"Warning: Settings file '{path}' not found. Using default values and environment variables.")
            cfg_data = {}
        except json.JSONDecodeError:
            print(f"Warning: Error decoding JSON from '{path}'. Using default values and environment variables.")
            cfg_data = {}

        openapi_cfg = cfg_data.get("openapi", {})
        general_cfg = cfg_data.get("general", {})
        ai_cfg = cfg_data.get("ai", {})
        strategy_cfg = cfg_data.get("strategy", {})

        # Prioritize env vars for secrets, then config file, then None
        client_id = env_client_id if env_client_id else openapi_cfg.get("client_id")
        client_secret = env_client_secret if env_client_secret else openapi_cfg.get("client_secret")

        if not client_id:
            print("Warning: cTrader Client ID not found in environment variables (CTRADER_CLIENT_ID) or config.json.")
        if not client_secret:
            print("Warning: cTrader Client Secret not found in environment variables (CTRADER_CLIENT_SECRET) or config.json.")

        openapi_settings = OpenAPISettings(
            client_id=client_id,
            client_secret=client_secret,
            host_type=openapi_cfg.get("host_type", "demo").lower(), # Ensure lowercase "demo" or "live"
            default_ctid_trader_account_id=openapi_cfg.get("default_ctid_trader_account_id"),
            spotware_auth_url=openapi_cfg.get("spotware_auth_url", "https://connect.spotware.com/oauth/v2/auth"), # Standard default
            spotware_token_url=openapi_cfg.get("spotware_token_url", "https://connect.spotware.com/oauth/v2/token"),
            redirect_uri=openapi_cfg.get("redirect_uri", "http://localhost:5000/callback") # Should generally not be overridden from config
        )

        general_settings = GeneralSettings(
            default_symbol=general_cfg.get("default_symbol", "EUR/USD"),
            chart_update_interval_ms=general_cfg.get("chart_update_interval_ms", 500),
            min_bars_for_trading=general_cfg.get("min_bars_for_trading", 50),
            risk_percentage=general_cfg.get("risk_percentage", 1.0),
            batch_profit_target=general_cfg.get("batch_profit_target", 10.0)
        )

        ai_settings = AISettings(
            use_ai_overseer=ai_cfg.get("use_ai_overseer", False),
            advisor_url=ai_cfg.get("advisor_url"),
            advisor_auth_token=os.environ.get("ADVISOR_AUTH_TOKEN") or ai_cfg.get("advisor_auth_token"),
            advisor_timeout_ms=ai_cfg.get("advisor_timeout_ms", 7000),
            advisor_min_confidence=ai_cfg.get("advisor_min_confidence", 0.65)
        )

        strategy_settings = StrategySettings(
            ema_period=strategy_cfg.get("ema_period", 50),
            atr_period=strategy_cfg.get("atr_period", 14),
            adx_period=strategy_cfg.get("adx_period", 14),
            rsi_period=strategy_cfg.get("rsi_period", 14),
            adx_threshold=strategy_cfg.get("adx_threshold", 25),
            rsi_overbought=strategy_cfg.get("rsi_overbought", 70),
            rsi_oversold=strategy_cfg.get("rsi_oversold", 30),
            stop_mult=strategy_cfg.get("stop_mult", 1.0),
            target_mult=strategy_cfg.get("target_mult", 0.5),
            buffer_mult=strategy_cfg.get("buffer_mult", 0.05),
        )

        return Settings(openapi=openapi_settings, general=general_settings, ai=ai_settings, strategy=strategy_settings)

    def save(self, path: str = "config.json") -> None:
        # Create a representation of settings that is safe to save (e.g., without tokens)
        # Only save configurable parts, not runtime state like access tokens.
        openapi_data_to_save = {
            "client_id": self.openapi.client_id if not os.environ.get("CTRADER_CLIENT_ID") else None,
            "client_secret": self.openapi.client_secret if not os.environ.get("CTRADER_CLIENT_SECRET") else None,
            "host_type": self.openapi.host_type,
            "default_ctid_trader_account_id": self.openapi.default_ctid_trader_account_id,
            "spotware_auth_url": self.openapi.spotware_auth_url,
            "spotware_token_url": self.openapi.spotware_token_url,
            "redirect_uri": self.openapi.redirect_uri # Typically not changed by user, but saved for completeness
        }
        # Filter out None values to keep config clean, especially for secrets from env
        # For the new URLs, they have defaults, so they won't be None unless explicitly set to None (which is unlikely)
        openapi_data_to_save = {k: v for k, v in openapi_data_to_save.items() if v is not None}

        if openapi_data_to_save.get("client_id") or openapi_data_to_save.get("client_secret"):
            print(f"Warning: Saving Client ID or Client Secret to '{path}'. "
                  "It's generally recommended to use environment variables for these secrets.")

        data_to_save = {
            "openapi": openapi_data_to_save,
            "general": {
                "default_symbol": self.general.default_symbol,
                "chart_update_interval_ms": self.general.chart_update_interval_ms,
                "min_bars_for_trading": self.general.min_bars_for_trading,
                "risk_percentage": self.general.risk_percentage,
                "batch_profit_target": self.general.batch_profit_target,
            },
            "ai": {
                "use_ai_overseer": self.ai.use_ai_overseer,
                "advisor_url": self.ai.advisor_url,
                "advisor_auth_token": self.ai.advisor_auth_token if not os.environ.get("ADVISOR_AUTH_TOKEN") else None,
                "advisor_timeout_ms": self.ai.advisor_timeout_ms,
                "advisor_min_confidence": self.ai.advisor_min_confidence
            },
            "strategy": {
                "ema_period": self.strategy.ema_period,
                "atr_period": self.strategy.atr_period,
                "adx_period": self.strategy.adx_period,
                "rsi_period": self.strategy.rsi_period,
                "adx_threshold": self.strategy.adx_threshold,
                "rsi_overbought": self.strategy.rsi_overbought,
                "rsi_oversold": self.strategy.rsi_oversold,
                "stop_mult": self.strategy.stop_mult,
                "target_mult": self.strategy.target_mult,
                "buffer_mult": self.strategy.buffer_mult,
            }
        }
        with open(path, 'w') as f:
            json.dump(data_to_save, f, indent=4)
