from stores.llm.LLMEnums import OpenAIEnums
from openai import OpenAI
from typing import Generator
import logging


class OpenAIProvider:

    def __init__(
        self,
        api_key: str,
        api_url: str = None,
        default_input_max_characters: int = 1000,
        default_generation_max_output_tokens: int = 1000,
        default_generation_temperature: float = 0.1,
    ):
        self.api_key = api_key
        self.api_url = api_url

        self.default_input_max_characters        = default_input_max_characters
        self.default_generation_max_output_tokens = default_generation_max_output_tokens
        self.default_generation_temperature       = default_generation_temperature

        self.generation_model_id = None
        self.embedding_model_id  = None
        self.embedding_size      = None

        self.client = OpenAI(
            api_key  = self.api_key,
            base_url = self.api_url if self.api_url and len(self.api_url) else None,
        )

        self.enums  = OpenAIEnums
        self.logger = logging.getLogger(__name__)

    # ── model setters ─────────────────────────────────────────────────────────

    def set_generation_model(self, model_id: str) -> None:
        self.generation_model_id = model_id

    def set_embedding_model(self, model_id: str, embedding_size: int) -> None:
        self.embedding_model_id = model_id
        self.embedding_size     = embedding_size

    # ── text helpers ──────────────────────────────────────────────────────────

    def process_text(self, text: str) -> str:
        return text[:self.default_input_max_characters].strip()

    def construct_prompt(self, prompt: str, role: str) -> dict:
        return {"role": role, "content": prompt}

    # ── generation ────────────────────────────────────────────────────────────

    def generate_text(
        self,
        prompt: list[dict],
        chat_history: list = [],
        max_output_tokens: int = None,
        temperature: float = None,
    ) -> str | None:
        """Blocking call — returns the complete response as one string."""
        if not self._generation_ready():
            return None

        max_output_tokens = max_output_tokens or self.default_generation_max_output_tokens
        temperature       = temperature       or self.default_generation_temperature

        response = self.client.chat.completions.create(
            model       = self.generation_model_id,
            messages    = prompt,
            max_tokens  = max_output_tokens,
            temperature = temperature,
        )

        if not response or not response.choices or not response.choices[0].message:
            self.logger.error("Error while generating text with OpenAI")
            return None

        return response.choices[0].message.content

    def generate_text_stream(
        self,
        prompt: list[dict],
        max_output_tokens: int = None,
        temperature: float = None,
    ) -> Generator[str, None, None]:
        """
        Blocking sync generator — yields tokens one by one as OpenAI produces them.

        This is intentionally sync because the caller (_stream_in_executor)
        runs it inside a thread-pool executor and pipes each token back to
        the async event loop via an asyncio.Queue.  Do not call this directly
        from async code.
        """
        if not self._generation_ready():
            return

        max_output_tokens = max_output_tokens or self.default_generation_max_output_tokens
        temperature       = temperature       or self.default_generation_temperature

        with self.client.chat.completions.create(
            model       = self.generation_model_id,
            messages    = prompt,
            max_tokens  = max_output_tokens,
            temperature = temperature,
            stream      = True,
        ) as stream:
            for chunk in stream:
                token = chunk.choices[0].delta.content
                if token is not None:
                    yield token

    # ── embeddings ────────────────────────────────────────────────────────────

    def embed_text(self, text: str, document_type: str = None) -> list[float] | None:
        if not self.client:
            self.logger.error("OpenAI client was not set")
            return None

        if not self.embedding_model_id:
            self.logger.error("Embedding model for OpenAI was not set")
            return None

        response = self.client.embeddings.create(
            model = self.embedding_model_id,
            input = text,
        )

        if not response or not response.data or not response.data[0].embedding:
            self.logger.error("Error while embedding text with OpenAI")
            return None

        return response.data[0].embedding

    # ── private ───────────────────────────────────────────────────────────────

    def _generation_ready(self) -> bool:
        if not self.client:
            self.logger.error("OpenAI client was not set")
            return False
        if not self.generation_model_id:
            self.logger.error("Generation model for OpenAI was not set")
            return False
        return True