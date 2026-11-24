import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from langchain_google_genai import ChatGoogleGenerativeAI

class Settings(BaseSettings):
    # Groq configuration
    LLM_PROVIDER: str = Field(default="groq", env="LLM_PROVIDER")
    GROQ_API_KEY: str = Field(default="", env="GROQ_API_KEY")
    GROQ_MODEL: str = Field(default="openai/gpt-oss-120b", env="GROQ_MODEL")

    GEMINI_API_KEY: str = Field(default="", env="GEMINI_API_KEY")
    GEMINI_MODEL: str = Field(default="gemini-3.8-flash", env="GEMINI_MODEL")
    # Backward compatible placeholder (unused)
    OLLAMA_MODEL: str = Field(default="llama3", env="OLLAMA_MODEL")
    # Chroma persist directory
    CHROMA_DIR: str = Field(default="./chroma_db", env="CHROMA_DIR")
    # FastAPI port
    API_PORT: int = Field(default=8000, env="API_PORT")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

settings = Settings()

def get_gemini_llm():
    """Return a configured LLM instance based on the provider.
    - If LLM_PROVIDER is "groq", returns a Groq Chat model.
    - Otherwise falls back to Gemini (or dummy if key missing).
    """
    import os
    # Groq provider
    if getattr(settings, "LLM_PROVIDER", "").lower() == "groq":
        from langchain_groq import ChatGroq
        if not settings.GROQ_API_KEY:
            raise RuntimeError("Groq API key not configured. Set GROQ_API_KEY in .env before using Groq features.")
        os.environ["GROQ_API_KEY"] = settings.GROQ_API_KEY
        model_name = settings.GROQ_MODEL or "groq/gpt-3.5-turbo"
        return ChatGroq(model_name=model_name, temperature=0)
    # Gemini fallback
    if not settings.GEMINI_API_KEY:
        class DummyLLM:
            def invoke(self, *_, **__):
                raise RuntimeError("Gemini API key not configured. Set GEMINI_API_KEY in .env before using Gemini features.")
        return DummyLLM()
    os.environ["GOOGLE_API_KEY"] = settings.GEMINI_API_KEY
    return ChatGoogleGenerativeAI(model=settings.GEMINI_MODEL)
