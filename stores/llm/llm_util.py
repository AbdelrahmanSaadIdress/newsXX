import torch, json
from json_repair import repair_json
from .LLM_Factory import LLMFactory
from helpers import get_settings, Settings

settings = get_settings()

def parse_json_safe(text: str):
    repaired_json = repair_json(text)
    data = json.loads(repaired_json)
    if isinstance(data, list):
        data = data[0]

    return data



def call_llm(config, messages):
    provider = LLMFactory.create(settings.PROVIDERS, config)
    provider.set_generation_model(model_id=settings.OPENAI_MODEL_ID)
    response = provider.generate_text(messages)
    data = parse_json_safe(response)
    return data




def call_llm_hf(config, messages):
    provider = LLMFactory.create(settings.PROVIDERS, config)
    model = provider.get_model()
    tokenizer = provider.get_tokenizer()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = tokenizer(
        text,
        return_tensors="pt"
    ).to(device)

    outputs = model.generate(
        **model_inputs,
        max_new_tokens=400,
        do_sample=False,
        temperature=0.1,
        pad_token_id=tokenizer.eos_token_id
    )
    # remove prompt tokens
    generated_ids = outputs[:, model_inputs.input_ids.shape[-1]:]
    response = tokenizer.decode(
        generated_ids[0],
        skip_special_tokens=True
    )
    data = parse_json_safe(response)