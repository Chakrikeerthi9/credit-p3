from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    SUPABASE_URL: str
    SUPABASE_KEY: str
    MODEL_VERSION: str = "v1.0.0"
    MAX_BATCH_SIZE: int = 5000
    SUPABASE_SERVICE_KEY: str

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

settings = Settings()