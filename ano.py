from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import re
import os

# ==============================
# モデル準備
# ==============================
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "Qwen/Qwen2-7B-instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

# ==============================
# 英作文ファイル読み込み
# ==============================
with open("essays_output_qwen25_7b.txt", "r", encoding="utf-8") as f:
    essay_text = f.read()

# Essay単位で分割
essays = re.split(r"<<<\s*Essay\s*\d+\s*>>>", essay_text)
essays = [e.strip() for e in essays if e.strip()]

# ==============================
# JSON抽出関数
# ==============================
def extract_json_arrays(output_text):
    json_arrays = []
    matches = re.findall(r"\[.*?\]", output_text, re.DOTALL)
    for m in matches:
        try:
            arr = json.loads(m)
            if isinstance(arr, list):
                json_arrays.extend(arr)
        except json.JSONDecodeError:
            continue
    return json_arrays

# ==============================
# 英語部分抽出
# ==============================
def extract_english(text):
    lines = text.splitlines()
    english_lines = [l for l in lines if not re.search(r'[一-龯ぁ-んァ-ン]', l)]
    return "\n".join(english_lines).strip()

# ==============================
# パラグラフ単位に分割
# ==============================
def split_paragraphs(text):
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    return paragraphs

# ==============================
# 評価観点
# ==============================
ANNOTATION_ASPECTS = [
    "Grammatical Accuracy",
    "Fluency",
    "Discourse Coherence",
    "Lexical Choice",
    "Utterance Length/Complexity",
    "Sentence Structure/Complexity",
    "Focus/Clarity of Topic",
    "Communicative Intent"
]

# ==============================
# プロンプトテンプレート
# ==============================
PROMPT_TEMPLATE = """You are a linguist expert specializing in text annotation for English as a second language.
• Annotate the paragraph for the following aspects: {aspects}.
• Each annotation should include:
  - type (aspect name)
  - annotation sentence (the sentence being evaluated)
  - annotation token (the specific word or phrase, if applicable)
  - rationale (why it is rated that way)
  - grammar correctness (true or false)
• Return strictly one JSON array only, containing annotations for all aspects.
• Do not include any text outside the JSON array.

Essay paragraph:
{paragraph}

Example output:
[
  {{
    "type": "Grammatical Accuracy",
    "annotation sentence": "He bought a apple.",
    "annotation token": "a",
    "rationale": "Article 'a' should be 'an' before a vowel sound.",
    "grammar correctness": false
  }},
  {{
    "type": "Fluency",
    "annotation sentence": "He bought a apple.",
    "annotation token": "",
    "rationale": "Sentence is understandable but could be smoother.",
    "grammar correctness": true
  }}
]
"""

# ==============================
# 出力保存用リスト
# ==============================
all_annotations = []

# ==============================
# Essayごとにアノテーション生成
# ==============================
for idx, essay in enumerate(essays, start=1):
    print(f"=== Essay {idx} ===")
    
    english_text = extract_english(essay)
    paragraphs = split_paragraphs(english_text)
    essay_annotations = []

    for para in paragraphs:
        prompt = PROMPT_TEMPLATE.format(
            paragraph=para,
            aspects=", ".join(ANNOTATION_ASPECTS)
        )
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=1000,
            do_sample=False,  # 安定化のためサンプリングオフ
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id
        )
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        para_annotations = extract_json_arrays(output_text)
        essay_annotations.extend(para_annotations)

    all_annotations.append({
        "essay_id": idx,
        "essay_text": english_text,
        "annotations": essay_annotations
    })

# ==============================
# JSONファイルに保存
# ==============================
output_file = "l2_evaluation_qwen2_7b_8aspects.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(all_annotations, f, indent=2, ensure_ascii=False)

print(f"\n すべてのアノテーションを {output_file} に保存しました。")
