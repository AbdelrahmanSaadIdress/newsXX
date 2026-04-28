import logging
from typing import Optional, Tuple
from stores.llm.provider_interface import BaseLLMProvider
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

logger = logging.getLogger(__name__)


class HuggingFaceLLMProvider(BaseLLMProvider):
    """
    Professional LLM Provider for loading HuggingFace models.

    Responsibilities
    ----------------
    - Load tokenizer
    - Load model
    - Handle device placement
    - Optional quantization
    - Provide reusable interface
    """

    def __init__(
        self,
        model_name_or_path: str = "deepseek-ai/deepseek-coder-6.7b-instruct",
        device: Optional[str] = "cuda",
        torch_dtype: Optional[torch.dtype] = torch.float16,
        trust_remote_code: bool = True,
    ):
        """
        Parameters
        ----------
        model_name_or_path : str
            HuggingFace model path or repo id.
        device : str, optional
            Device to run model on ("cuda", "cpu", "auto").
        torch_dtype : torch.dtype
            Data type for model weights.
        trust_remote_code : bool
            Whether to allow custom model code.
        """

        self.model_name_or_path = model_name_or_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch_dtype
        self.trust_remote_code = trust_remote_code

        self.model, self.tokenizer =  self.load()

        logger.info(f"Initialized LLMProvider for {self.model_name_or_path}")

    def load(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Load the model and tokenizer.

        Returns
        -------
        Tuple[model, tokenizer]
        """

        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=self.trust_remote_code
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info("Loading model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            torch_dtype=self.torch_dtype,
            trust_remote_code=self.trust_remote_code,
            device_map="auto" if self.device == "cuda" else None,
        )

        if self.device == "cpu":
            self.model.to(self.device)

        logger.info("Model loaded successfully.")

        return self.model, self.tokenizer

    def get_model(self):
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        return self.model

    def get_tokenizer(self):
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded. Call load() first.")
        return self.tokenizer

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """
        Generate text from the model.
        """

        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True
            )

        return self.tokenizer.decode(output[0], skip_special_tokens=True)