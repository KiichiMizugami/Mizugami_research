from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# デバイス設定
device = "cuda" if torch.cuda.is_available() else "cpu"

# Hugging Face HubのリポジトリID
model_id = "Qwen/Qwen2-0.5B"

# Tokenizer & Model
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,  # 高速化のためfloat16を使用
    device_map="auto"
)

# プロンプト
prompt = """You are a Japanese English learner,but your English is still simple..
Your native language is Japanese.
Please write an English essay as if you are a Japanese student learning English.
The essay should answer the following topic:

"It is important for college students to have a part-time job."

Write in simple English, and it is okay to include some small grammar mistakes that a Japanese learner might make."""

# トークナイズ
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# 入力のトークン数を取得
input_length = inputs.input_ids.shape[1]

# 生成する作文の数
num_essays = 20

print("--- LLMによる日本語母語話者の英作文シミュレーション（全20個） ---")
print("-" * 60)

for i in range(1, num_essays + 1):
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )

    # 入力部分を除外して生成部分だけを取り出す
    generated_tokens = outputs[0][input_length:]

    # デコード
    essay = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    print(f"\n<<< Essay {i} >>>")
    print(essay.strip())  # 余計な空白を削除
    print("-" * 60)

print("生成を完了しました。")
