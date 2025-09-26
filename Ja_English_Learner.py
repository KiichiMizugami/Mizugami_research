from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

# Hugging Face HubのリポジトリID
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

# 出力ファイルを開く（追記モード）
with open("ja_english_learner_essay.txt", "w", encoding="utf-8") as f:
    # 20個生成
    for i in range(1, 21):
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7
        )
        essay = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 翻訳プロンプトを作成
        translate_prompt = f"Translate the following English essay into Japanese:\n\n{essay}"
        translate_inputs = tokenizer(translate_prompt, return_tensors="pt").to(model.device)

        translate_outputs = model.generate(
            **translate_inputs,
            max_new_tokens=300,
            do_sample=False  # 翻訳はサンプリング不要で確定的に
        )
        ja_essay = tokenizer.decode(translate_outputs[0], skip_special_tokens=True)

        # ファイルに書き込み
        f.write(f"--- Essay {i} ---\n")
        f.write(essay + "\n\n")
        f.write("--- 日本語訳 ---\n")
        f.write(ja_essay + "\n\n")

        # 画面にも出力
        print(f"Essay {i} saved.")

print(" 20個の英作文と日本語訳を 'ja_english_learner_essay.txt' に保存しました！")

 