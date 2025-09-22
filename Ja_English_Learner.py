from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

# ローカルパスではなく、Hugging Face HubのリポジトリIDを直接指定します
model_id = "Qwen/Qwen2-0.5B"

# Tokenizer & Model
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

# プロンプト（課題に答える英作文）
prompt = """You are a Japanese English learner. 
Your native language is Japanese. 
Please write an English essay as if you are a Japanese student learning English. 
The essay should answer the following topic:

"It is important for college students to have a part-time job."

Write in simple English, and it is okay to include some small grammar mistakes that a Japanese learner might make."""

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# 生成
outputs = model.generate(
    **inputs,
    max_new_tokens=200,
    do_sample=True,
    temperature=0.7
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))