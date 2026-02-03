from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

# -----------------------------
# Model & tokenizer
# -----------------------------
MODEL_NAME = "google/flan-t5-small"

tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

# -----------------------------
# Style configuration
# -----------------------------
STYLE_CONFIG = {
    "short": {
        "prompt": "Summarize this in 1-2 concise sentences: ",
        "max_length": 50,
        "min_length": 20,
        "length_penalty": 1.8
    },
    "medium": {
        "prompt": "Summarize the following text in detail: ",
        "max_length": 120,
        "min_length": 60,
        "length_penalty": 1.0
    }
}

# -----------------------------
# TOKEN-AWARE CHUNKING
# -----------------------------
def split_text(text, style="medium", max_tokens=480):

    prompt = STYLE_CONFIG[style]["prompt"]
    prompt_tokens = len(tokenizer.encode(prompt))

    safe_chunk_size = max_tokens - prompt_tokens

    token_ids = tokenizer.encode(text)

    chunks = []

    for i in range(0, len(token_ids), safe_chunk_size):
        chunk_ids = token_ids[i:i + safe_chunk_size]
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
        chunks.append(chunk_text)

    return chunks


# -----------------------------
# Core summarization
# -----------------------------
def summarize_batch(
    texts,
    style="medium",
    temperature=0.7,
    num_beams=4
):

    if style not in STYLE_CONFIG:
        raise ValueError("Invalid style")

    config = STYLE_CONFIG[style]

    prompts = [config["prompt"] + t for t in texts]

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
            max_length=config["max_length"],
            min_length=config["min_length"],
            num_beams=num_beams,
            length_penalty=config["length_penalty"],
            early_stopping=True
        )

    return [
        tokenizer.decode(o, skip_special_tokens=True)
        for o in outputs
    ]


# -----------------------------
# LONG TEXT SUMMARIZATION
# -----------------------------
def summarize_long_text(
    text,
    style="medium",
    temperature=0.7,
    num_beams=4
):

    # Step 1: Chunk input safely
    chunks = split_text(text, style=style)

    # Step 2: Summarize chunks
    chunk_summaries = summarize_batch(
        chunks,
        style=style,
        temperature=temperature,
        num_beams=num_beams
    )

    if len(chunk_summaries) == 1:
        return chunk_summaries[0]

    # Step 3: Label chunks for diversity
    labeled_summaries = [
        f"Section {i+1}: {s}"
        for i, s in enumerate(chunk_summaries)
    ]

    combined = "\n\n".join(labeled_summaries)

    # Step 4: Check token size again
    combined_tokens = len(tokenizer.encode(combined))

    if combined_tokens > 480:
        return summarize_long_text(
            combined,
            style=style,
            temperature=temperature,
            num_beams=num_beams
        )

    # Step 5: Final summarization
    return summarize_batch(
        [combined],
        style=style,
        temperature=temperature,
        num_beams=num_beams
    )[0]
