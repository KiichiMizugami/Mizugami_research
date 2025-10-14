import os
import re
import json
from collections import Counter
import math
import glob

# ==============================
# 1️⃣ ディレクトリ設定
# ==============================
ICNALE_DIR = "/home/mizugami/Ja_English_Learner/icnale/ICNALE_WE_2.6/WE_3_Classified_Mereged_Tagged"
LLM_FILE = "/home/mizugami/Ja_English_Learner/l2_evaluation_qwen2_7b_stable.json"

# ==============================
# 2️⃣ ICNALEテキスト読み込み
# ==============================
def load_icnale_texts(icnale_dir):
    texts = []
    for path in glob.glob(os.path.join(icnale_dir, "*.txt")):
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
            # 文単位に分割（簡易）
            sentences = [s.strip() for s in re.split(r'[.?!]\s+', content) if s.strip()]
            texts.extend(sentences)
    return texts

icnale_texts = load_icnale_texts(ICNALE_DIR)
print(f"ICNALE文数: {len(icnale_texts)}")

# ==============================
# 3️⃣ LLM生成文読み込み
# ==============================
def load_llm_texts(llm_file):
    with open(llm_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    sentences = []
    for essay in data:
        for ann in essay.get("annotations", []):
            s = ann.get("annotation sentence")
            if s:
                sentences.append(s.strip())
    return sentences

llm_texts = load_llm_texts(LLM_FILE)
print(f"LLM生成文数: {len(llm_texts)}")

# ==============================
# 4️⃣ 単語分布・情報理論指標
# ==============================
def word_distribution(sentences):
    counter = Counter()
    total = 0
    for s in sentences:
        tokens = s.lower().split()
        counter.update(tokens)
        total += len(tokens)
    probs = {w: c / total for w, c in counter.items()}
    return probs

def entropy(probs):
    return -sum(p * math.log2(p) for p in probs.values())

def kl_divergence(P, Q):
    eps = 1e-10
    all_tokens = set(P.keys()).union(Q.keys())
    return sum(Q.get(w, eps) * math.log2(Q.get(w, eps) / P.get(w, eps)) for w in all_tokens)

# 分布計算
P_llm = word_distribution(llm_texts)
Q_icnale = word_distribution(icnale_texts)

# 指標計算
H_icnale = entropy(Q_icnale)
H_llm = entropy(P_llm)
KL_icnale_llm = kl_divergence(P_llm, Q_icnale)

print("\n=== 情報理論的指標 ===")
print(f"ICNALE 文書の語彙エントロピー: {H_icnale:.4f}")
print(f"LLM生成文の語彙エントロピー: {H_llm:.4f}")
print(f"KLダイバージェンス (ICNALE || LLM): {KL_icnale_llm:.4f}")

# ==============================
# 5️⃣ 文法アノテーションと結合可能（例）
# ==============================
# 後で各文の8観点スコアと組み合わせれば、
# L1依存のバイアスや複雑度指標の解析が可能

