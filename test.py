# tested with Nvidia-driver 580 and Cuda-13
# GPU: NVIDIA GeForce RTX 3070 Ti Laptop GPU  8192MiB Memory.
# Answer Generation finished in 48.91 seconds
import time
import torch
from airllm import AutoModel

MODEL_NAME = "Qwen/Qwen3-0.6B"
MAX_LENGTH = 128

print("=" * 60)
print("Loading model...")

t0 = time.time()

model = AutoModel.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16,
)

print(f"Model loaded in {time.time() - t0:.2f} seconds")

# -----------------------------
# Chat messages
# -----------------------------
messages = [
    {
        "role": "user",
        "content": "/no_think\nWhat is the capital of the United States?\nOnly give me the name of the city."
    }
]

# -----------------------------
# Build chat prompt
# -----------------------------
prompt = model.tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

print("=" * 60)
print("Prompt sent to model:")
print(prompt)

# -----------------------------
# Tokenize
# -----------------------------
input_tokens = model.tokenizer(
    prompt,
    return_tensors="pt",
    truncation=True,
    max_length=MAX_LENGTH,
    padding=True,
    return_attention_mask=True,
)

print("=" * 60)
print("Tokenizer output keys:")
print(input_tokens.keys())

device = "cuda" if torch.cuda.is_available() else "cpu"

input_ids = input_tokens["input_ids"].to(device)
attention_mask = input_tokens["attention_mask"].to(device)

print("=" * 60)
print("Generating...")

t1 = time.time()

generation_output = model.generate(
    input_ids,
    attention_mask=attention_mask,
    max_new_tokens=32,
    do_sample=False,
    use_cache=True,
    return_dict_in_generate=True,
)

print(f"Generation finished in {time.time() - t1:.2f} seconds")

output = model.tokenizer.decode(
    generation_output.sequences[0],
    skip_special_tokens=True,
)

print("=" * 60)
print(output)
