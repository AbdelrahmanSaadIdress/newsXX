from __future__ import annotations
import chromadb
from helpers.Config import get_settings
from stores.llm.LLM_Factory import LLMFactory
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

import torch
import json_repair
import json
from pydantic import BaseModel


from schema import NewsDetails, TranslatedStory
from pydantic import ValidationError


class A_TDeps:
    """
    Initialised once at application startup and shared across all requests.
    Wraps the project's own Analyzer Model and ChromaDB.
    """

    def __init__(self, top_k: int = 6):
        settings   = get_settings()
        self.top_k = top_k

        # ── provider (your own OpenAIProvider via LLMFactory) ────────────────
        self.embed_provider = LLMFactory.create(
            provider=settings.PROVIDERS,
            config={
                "api_key": settings.OPENAI_API_KEY,
                "api_url": settings.OPENAI_API_URL,
            },
        )
        self.embed_provider.set_embedding_model(
            model_id       = settings.OPENAI_EMBEDDING_MODEL_ID,
            embedding_size = 1536 ,
        )

        base_model_name = settings.ANALYZER_MODEL_NAME
        adapter_name    = settings.ADAPTER_NAME

        self.tokenizer  = AutoTokenizer.from_pretrained(base_model_name)
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map="auto",
        )
        self.model = PeftModel.from_pretrained(self.base_model, adapter_name)

        self.system_message = "\n".join([
            "You are a professional NLP data parser.",
            "Follow the provided `Task` by the user and the `Output Scheme` to generate the `Output JSON`.",
            "Do not generate any introduction or conclusion.",
        ])
        self.analyzing_task         = "Extrat the story details into a JSON."
        self.translate_english_task = "You have to translate the story content into english associated with a title into a JSON."
        self.translate_french_task  = "You have to translate the story content into french associated with a title into a JSON."

        # ── ChromaDB ─────────────────────────────────────────────────────────
        self.chroma_client = chromadb.PersistentClient(path=settings.CHROMA_PATH)
        self.collection    = self.chroma_client.get_or_create_collection(
            name     = settings.CHROMA_COLL,
            metadata = {"hnsw:space": "cosine"},
        )

    # ── prompt builders ───────────────────────────────────────────────────────


    def build_instruction(self, story: str, task: str, schema: type[NewsDetails] | type[TranslatedStory]) -> str:
        """Build the user-side instruction block."""
        schema_str = json.dumps(schema.model_json_schema(), indent=2)
        return "\n".join([
            "# Story:",
            story,
            "",
            "# Task:",
            task,
            "",
            "# Output Schema:",
            "```json",
            schema_str,
            "```",
            "",
            "# Output JSON:",
            "```json",
        ])

    def build_prompt(self, instruction: str) -> str:
        """Wrap instruction in the chat template."""
        return "\n".join([
            "<|system|>",
            self.system_message,
            "<|user|>",
            instruction,
            "<|assistant|>",
        ])

    # ── JSON helpers ──────────────────────────────────────────────────────────

    def parse_json(self, text: str):
        try:
            print("*******************************************")
            print(text)
            print("*******************************************")

            return json_repair.loads(text)
        except Exception:
            return None

    # ── analysis ─────────────────────────────────────────────────────────────

    def extract_and_validate_analysis(self, full_response: str) -> NewsDetails | None:
        assistant_split = full_response.split("<|assistant|>")
        if len(assistant_split) < 2:
            print("Could not find assistant response")
            return None

        assistant_output = assistant_split[-1].strip()

        if assistant_output.startswith("```json"):
            assistant_output = assistant_output[len("```json"):].strip()
        if assistant_output.endswith("```"):
            assistant_output = assistant_output[:-3].strip()

        parsed = self.parse_json(assistant_output)
        if parsed is None:
            print("JSON parsing failed")
            return None

        if isinstance(parsed, list):
            if len(parsed) == 0:
                print("JSON parsing returned empty list")
                return None
            parsed = parsed[0]

        try:
            return NewsDetails(**parsed)
        except ValidationError as e:
            print(f"Validation error: {e}")
            return None

    def generate_analysis(self, story: str) -> NewsDetails | None:
        # FIX: build_instruction now receives the task; build_prompt receives only instruction
        instruction = self.build_instruction(story=story, task=self.analyzing_task, schema=NewsDetails)
        # instruction = self.build_instruction(story=story, task=self.analyzing_task)
        prompt      = self.build_prompt(instruction)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=4048,
                temperature=0.2,
                do_sample=False,
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return self.extract_and_validate_analysis(response)

    # ── translation ───────────────────────────────────────────────────────────

    def extract_and_validate_translation(self, full_response: str) -> TranslatedStory | None:
        assistant_split = full_response.split("<|assistant|>")
        if len(assistant_split) < 2:
            print("Could not find assistant response")
            return None

        assistant_output = assistant_split[-1].strip()

        if assistant_output.startswith("```json"):
            assistant_output = assistant_output[len("```json"):].strip()
        if assistant_output.endswith("```"):
            assistant_output = assistant_output[:-3].strip()

        parsed = self.parse_json(assistant_output)
        if parsed is None:
            print("JSON parsing failed")
            return None
        
        if isinstance(parsed, list):
            if len(parsed) == 0:
                print("JSON parsing returned empty list")
                return None
            parsed = parsed[0]


        try:
            return TranslatedStory(**parsed)
        except ValidationError as e:
            print(f"Validation error: {e}")
            return None

    def generate_english_translation(self, story: str) -> TranslatedStory | None:
        # FIX: same pattern — task passed to build_instruction, not build_prompt
        # instruction = self.build_instruction(story=story, task=self.translate_english_task)
        instruction = self.build_instruction(story=story, task=self.translate_english_task, schema=TranslatedStory)

        prompt      = self.build_prompt(instruction)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.2,
                do_sample=False,
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return self.extract_and_validate_translation(response)

    def generate_french_translation(self, story: str) -> TranslatedStory | None:
        # instruction = self.build_instruction(story=story, task=self.translate_french_task)
        instruction = self.build_instruction(story=story, task=self.translate_french_task, schema=TranslatedStory)

        prompt      = self.build_prompt(instruction)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=4048,
                temperature=0.2,
                do_sample=False
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return self.extract_and_validate_translation(response)