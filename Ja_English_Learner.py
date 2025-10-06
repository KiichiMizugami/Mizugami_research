from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

# キャッシュ先を自分のホームに変更（権限エラー対策）
os.environ["HF_HOME"] = os.path.expanduser("~/huggingface")
os.environ["TRANSFORMERS_CACHE"] = os.path.expanduser("~/huggingface")

# 出力ファイル
output_file = "essays_output.txt"

# デバイス設定
device = "cuda" if torch.cuda.is_available() else "cpu"

# Hugging Face HubのリポジトリID
model_id = "Qwen/Qwen2-0.5B"

# Tokenizer & Model
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.float16,  # 修正済み
    device_map="auto"
)

# プロンプト
prompt = """You are a Japanese English learner.
Your native language is Japanese.
Please write an English essay as if you are a Japanese student learning English.
The essay should answer the following topic:

"It is important for college students to have a part-time job."

Write in simple English, and it is okay to include some small grammar mistakes that a Japanese learner might make."""

# トークナイズ
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
input_length = inputs.input_ids.shape[1]

# 生成する作文の数
num_essays = 20
outputs = model.generate(
    **inputs,
    max_new_tokens=350,
    do_sample=True,
    temperature=0.6,
    pad_token_id=tokenizer.eos_token_id,
    num_return_sequences=num_essays
)

print("--- LLMによる日本語母語話者の英作文シミュレーション（全20個） ---")
print("-" * 60)

# ファイルを一度開いてまとめて書き込む
with open(output_file, "w", encoding="utf-8") as f:
    for i, output in enumerate(outputs, 1):
        generated_tokens = output[input_length:]
        essay = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        # 画面に出力
        print(f"\n<<< Essay {i} >>>")
        print(essay)
        print("-" * 60)

        # ファイルに追記
        f.write(f"<<< Essay {i} >>>\n")
        f.write(essay + "\n")
        f.write("-" * 60 + "\n")

print(f"生成を完了しました。すべてのエッセイを '{output_file}' にまとめて保存しました。")

