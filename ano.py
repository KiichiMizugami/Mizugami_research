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
# 対象の英作文
# ==============================
essay_text = """
I will work more part-time job this year, so I can make more friends.
To learn English, I think that part-time job is good way.

It is not necessary to explain the whole topic, there are many parts to make part-time job. In my opinion, it can be a good experience. I want to understand society, and I want to learn English. I think part-time job can be a good way to achieve this goal.

I work in restaurant, and I have the same experience. I am still learning English. I have to learn how to deal with customers, and how to deal with people. I want to improve my English so that I can understand society better. I think it is good for college students to have a part-time job.

In the end, I want to say that it is important for college students to have a part-time job. It can give the college student experience to learn responsibility and how to talk with customer. It can also improve the college student
"""

# ==============================
# アノテーションプロンプト
# ==============================
annotation_prompt = f"""
You are a linguist expert specializing in English as a second language.
You will annotate the following essay based on grammar and linguistic aspects.

Rules for annotation:
- Keep the essay text unchanged.
- Each annotation should be a JSON object with the following fields:
  - type: type of annotation (e.g., tense error, article missing, singular/plural mistake, word choice)
  - annotation sentence: the full sentence that contains the issue
  - annotation token: the exact word(s) with the issue
  - rationale: reason for this annotation
  - grammar correctness: true if grammatically correct, false otherwise

Return the output as a JSON array (list of annotations).

Essay text:
{essay_text}

Output the annotations in JSON format:
"""

# ==============================
# モデルで生成
# ==============================
inputs = tokenizer(annotation_prompt, return_tensors="pt").to(device)
#print(annotation_prompt)
outputs = model.generate(
    **inputs,
    max_new_tokens=600,
    do_sample=True,
    temperature=0.7,
    pad_token_id=tokenizer.eos_token_id
)

annotated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(annotated_text)
# # ==============================
# # JSONの整形と出力
# # ==============================
# try:
#     annotated_json = json.loads(annotated_text)
# except json.JSONDecodeError:
#     annotated_json = {"error": "Failed to parse JSON", "output_text": annotated_text}

# print(json.dumps(annotated_json, ensure_ascii=False, indent=2))
