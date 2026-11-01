import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Qwen/Qwen3-4B"

# 1. AutoTokenizer inspects the model repo and loads Qwen's exact 151,643 BPE vocab
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 2. Convert text into numerical token IDs
prompt = "Explain latency vs throughput."
inputs = tokenizer(prompt, return_tensors="pt")
# inputs["input_ids"] is a 2D tensor: tensor([[ 3432, 23912,  1890, 48291,    13]])

# 3. Load the Transformer model onto your GPU/MPS
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.bfloat16,
    device_map="auto"
)

# 4. Forward pass generates new token IDs
with torch.inference_mode():
    output_ids = model.generate(**inputs, max_new_tokens=50)

# 5. Tokenizer decodes the generated token IDs back to human text
response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(response)
