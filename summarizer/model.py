from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

MODEL_NAME = "google/flan-t5-small"

tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)


STYLE_PROMPTS = {
    "short": "summarize briefly: ",
    "medium": "summarize the following text: "
}


def summarize_batch(
    texts,
    style="medium",
    max_length=120,
    temperature=0.7,
    num_beams=4
):
    if style not in STYLE_PROMPTS:
        raise ValueError("style must be 'short' or 'medium'")

    prompts = [STYLE_PROMPTS[style] + t for t in texts]

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
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
