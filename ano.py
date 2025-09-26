from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import pandas as pd

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
# 1. 文法アノテーション生成
# ==============================
with open("ja_english_learner_essay.txt", "r", encoding="utf-8") as f:
    essays = f.read().split("--- Essay ")

annotated_results = {}

for essay in essays:
    if essay.strip() == "":
        continue
    lines = essay.strip().split("\n", 1)
    essay_num = lines[0].strip()
    essay_text = lines[1].strip() if len(lines) > 1 else ""

    annotation_prompt = f"""
You are a linguist expert specializing in doing text annotation in English as a second language.
You will annotate the following essay text based on grammar and linguistic aspects.
- Keep the passage unchanged.
- Each annotation should be an object with 5 fields:
  - type: type of annotation
  - annotation sentence: the annotated sentence
  - annotation token: the token(s) that are annotated
  - rationale: reason for this annotation
  - grammar correctness: true if grammatically correct, false otherwise
- Return the output as a JSON object with one or multiple annotations.

Essay text:
{essay_text}
"""

    inputs = tokenizer(annotation_prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=500,
        do_sample=True,
        temperature=0.7
    )

    annotated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    try:
        annotated_json = json.loads(annotated_text)
    except json.JSONDecodeError:
        annotated_json = {"error": "Failed to parse JSON", "output_text": annotated_text}

    annotated_results[essay_num] = annotated_json

with open("ja_english_learner_essay_annotated.json", "w", encoding="utf-8") as f:
    json.dump(annotated_results, f, ensure_ascii=False, indent=2)

print("JSON形式の文法アノテーションファイルを作成しました！")

# ==============================
# 2. JSONをCSVに変換
# ==============================
rows = []
for essay_num, annotations in annotated_results.items():
    if "error" in annotations:
        rows.append({
            "essay_num": essay_num,
            "type": None,
            "annotation_sentence": None,
            "annotation_token": None,
            "rationale": None,
            "grammar_correctness": None,
            "note": annotations.get("output_text")
        })
        continue

    for ann in annotations:
        rows.append({
            "essay_num": essay_num,
            "type": ann.get("type"),
            "annotation_sentence": ann.get("annotation sentence"),
            "annotation_token": ann.get("annotation token"),
            "rationale": ann.get("rationale"),
            "grammar_correctness": ann.get("grammar correctness"),
            "note": None
        })

df = pd.DataFrame(rows)
df.to_csv("ja_english_learner_essay_annotated.csv", index=False, encoding="utf-8-sig")

print("CSV形式で整理した文法アノテーションファイルを作成しました." \
"")
