from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    # Application settings
    APP_NAME: str = "Code Generation Assistant"
    DEBUG: bool = False
    
    # LLM settings
    OPENROUTER_API_KEY: Optional[str] = None
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    DEFAULT_MODEL: str = "deepseek/deepseek-chat-v3-0324:free"
    

    # Document processing settings
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    
    # Memory settings
    CONVERSATION_WINDOW_SIZE: int = 10
    
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

# Create settings instance
settings = Settings()
