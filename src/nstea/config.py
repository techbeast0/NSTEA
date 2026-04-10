"""Centralized configuration for NS-TEA using Pydantic Settings."""

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = {"env_prefix": "NSTEA_", "env_file": ".env", "extra": "ignore"}

    # Model provider: "ollama" or "huggingface"
    model_provider: str = "ollama"

    # Model ID — provider-specific model identifier
    # Ollama local: "gpt-oss:20b", "llama3.1:8b", "qwen3:8b"
    # Ollama cloud: "gpt-oss:120b-cloud"
    # HuggingFace: "meta-llama/Meta-Llama-3-8B-Instruct", etc.
    model_id: str = "gpt-oss:120b-cloud"

    # Ollama
    ollama_host: str = "http://localhost:11434"
    ollama_api_key: str = ""

    # HuggingFace
    hf_token: str = ""

    # Generation parameters
    temperature: float = 0.2
    max_tokens: int = 4096

    # Data paths (relative to project root)
    fhir_data_dir: str = "data/fhir"
    test_cases_dir: str = "data/test_cases"

    # Logging
    log_level: str = "INFO"

    @property
    def project_root(self) -> Path:
        return Path(__file__).resolve().parent.parent.parent

    @property
    def fhir_path(self) -> Path:
        return self.project_root / self.fhir_data_dir

    @property
    def test_cases_path(self) -> Path:
        return self.project_root / self.test_cases_dir


settings = Settings()
