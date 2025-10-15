from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import re
import os
from glob import glob

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
# 英語部分抽出
# ==============================
def extract_english(text):
    lines = text.splitlines()
    english_lines = [l for l in lines if not re.search(r'[一-龯ぁ-んァ-ン]', l)]
    return "\n".join(english_lines).strip()

# ==============================
# ICNALE用タグ除去
# ==============================
def clean_icnale_text(raw_text):
    cleaned_lines = []
    for line in raw_text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("<") or line.startswith("</"):
            continue  # <s>, </s>, <g/>などをスキップ
        parts = line.split("\t")
        if len(parts) >= 1:
            cleaned_lines.append(parts[0])
    return " ".join(cleaned_lines)

# ==============================
# パラグラフ単位に分割
# ==============================
def split_paragraphs(text):
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    return paragraphs

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

Paragraph:
{paragraph}
"""

# ==============================
# アノテーション関数
# ==============================
def annotate_paragraphs(paragraphs):
    annotations = []
    for para in paragraphs:
        prompt = PROMPT_TEMPLATE.format(
            paragraph=para,
            aspects=", ".join(ANNOTATION_ASPECTS)
        )
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=1000,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id
        )
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        para_annotations = extract_json_arrays(output_text)
        annotations.extend(para_annotations)
    return annotations

# ==============================
# LLM生成文のアノテーション
# ==============================
with open("essays_output_qwen25_7b.txt", "r", encoding="utf-8") as f:
    essay_text = f.read()

essays = re.split(r"<<<\s*Essay\s*\d+\s*>>>", essay_text)
essays = [e.strip() for e in essays if e.strip()]

llm_annotations = []
for idx, essay in enumerate(essays, start=1):
    print(f"=== Annotating LLM Essay {idx} ===")
    english_text = extract_english(essay)
    paragraphs = split_paragraphs(english_text)
    essay_anns = annotate_paragraphs(paragraphs)
    llm_annotations.append({
        "essay_id": idx,
        "essay_text": english_text,
        "annotations": essay_anns
    })

with open("l2_annotations_llm.json", "w", encoding="utf-8") as f:
    json.dump(llm_annotations, f, indent=2, ensure_ascii=False)

# ==============================
# ICNALE文のアノテーション
# ==============================
icnale_dir = "./icnale/ICNALE_WE_2.6/WE_3_Classified_Mereged_Tagged/"
icnale_files = glob(os.path.join(icnale_dir, "*.txt"))

icnale_annotations = []
for idx, file_path in enumerate(icnale_files, start=1):
    print(f"=== Annotating ICNALE File {idx}/{len(icnale_files)} ===")
    with open(file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()
    cleaned_text = clean_icnale_text(raw_text)           # タグ除去
    english_text = extract_english(cleaned_text)         # 英語抽出（必要なら）
    paragraphs = split_paragraphs(english_text)
    file_anns = annotate_paragraphs(paragraphs)
    icnale_annotations.append({
        "file_name": os.path.basename(file_path),
        "text": english_text,
        "annotations": file_anns
    })

with open("l2_annotations_icnale.json", "w", encoding="utf-8") as f:
    json.dump(icnale_annotations, f, indent=2, ensure_ascii=False)

print("\n LLM文とICNALE文のアノテーションが完了しました。")
