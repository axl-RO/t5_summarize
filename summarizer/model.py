from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

MODEL_NAME = "google/flan-t5-small"

tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)


STYLE_CONFIG = {
    "short": {
        "prompt": "Summarize this in 1-2 sentences: ",
        "max_length": 50,
        "min_length": 20,
    },
    "medium": {
        "prompt": "Summarize the following text: ",
        "max_length": 120,
        "min_length": 60,
    }
}

def summarize_batch(
    texts,
    style="medium",
    max_length=120,
    temperature=0.7,
    num_beams=4
):
    if style not in STYLE_CONFIG:
        raise ValueError("style must be 'short' or 'medium'")
    
    config = STYLE_CONFIG[style]

    prompts = [config["prompt"] + t for t in texts]

    
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )

    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=config["max_length"],
        min_length=config["min_length"],
        num_beams=num_beams,
        temperature=temperature,
        length_penalty=1.2,
        early_stopping=True,
    )


    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,
            num_beams=num_beams,
            temperature=temperature,
            early_stopping=True
        )

    return [
        tokenizer.decode(o, skip_special_tokens=True)
        for o in outputs
    ]
