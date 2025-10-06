from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json

# ==============================
# モデル準備
# ==============================
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "Qwen/Qwen2-0.5B"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

# ==============================
# essays_output.txt から英作文を読み込む
# ==============================
with open("essays_output.txt", "r", encoding="utf-8") as f:
    essay_text = f.read()

# ==============================
# アノテーションプロンプト
# ==============================
annotation_prompt = f"""
You are a linguist expert specializing in English as a second language.
Annotate the following essay for grammar and linguistic issues.

Rules:
- Keep the essay text unchanged.
- Each annotation must be a JSON object with fields:
  - type
  - annotation_sentence
  - annotation_token
  - rationale
  - grammar_correctness (true/false)

Essay text:
{essay_text}

Return only a JSON array of annotations.
"""

# ==============================
# モデルで生成
# ==============================
inputs = tokenizer(annotation_prompt, return_tensors="pt").to(device)

outputs = model.generate(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_new_tokens=600,
    do_sample=True,
    temperature=0.7,
    pad_token_id=tokenizer.eos_token_id
)

annotated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# ==============================
# JSON部分だけ抽出してパース
# ==============================
try:
    annotations_start = annotated_text.find('[')
    annotations_end = annotated_text.rfind(']') + 1
    annotations_json = annotated_text[annotations_start:annotations_end]
    annotations = json.loads(annotations_json)
except Exception as e:
    print("JSONパースエラー:", e)
    annotations = []

# ==============================
# 結果を表示
# ==============================
print(json.dumps(annotations, indent=2, ensure_ascii=False))
