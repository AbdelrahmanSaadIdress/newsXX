from typing import Dict, Any
from stores.llm.providers.HugginFaceProvider import HuggingFaceLLMProvider
from stores.llm.providers.OpenAIProvider import OpenAIProvider


class LLMFactory:
    """
    Factory class for creating LLM providers.
    """

    PROVIDERS = {
        "huggingface": HuggingFaceLLMProvider,
        "openai":OpenAIProvider
    }

    @classmethod
    def create(
        cls,
        provider: str,
        config: Dict[str, Any]={}
    ):
        """
        Create an LLM provider.

        Parameters
        ----------
        provider : str
            Provider type (huggingface, openai, ollama, etc.)
        config : dict
            Provider configuration.

        Returns
        -------
        LLM Provider instance
        """

        provider = provider.lower()
        if provider not in cls.PROVIDERS:
            raise ValueError(f"Unsupported provider: {provider}")

        provider_class = cls.PROVIDERS[provider]

        return provider_class(**config)