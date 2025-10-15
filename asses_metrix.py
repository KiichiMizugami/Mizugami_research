import json
import pandas as pd
from collections import Counter, defaultdict
import math
import matplotlib.pyplot as plt

# ===== 設定 =====
LLM_FILE = "l2_annotations_llm.json"
ICNALE_FILE = "l2_annotations_icnale.json"

# ===== データ読み込み関数 =====
def load_annotations(filepath, source_name):
    """各ファイルを読み込み、フラット化してDataFrame化"""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = []
    texts = []
    for essay in data:
        essay_id = essay.get("essay_id") or essay.get("file_name")
        text = essay.get("essay_text") or essay.get("text", "")
        texts.append(text)
        for ann in essay.get("annotations", []):
            records.append({
                "source": source_name,
                "essay_id": essay_id,
                "aspect": ann.get("type"),
                "sentence": ann.get("annotation sentence"),
                "grammar_correctness": ann.get("grammar correctness"),
                # X: L1 (ICNALEは元データから、LLMはプロンプトで指定)
                "L1": essay.get("L1", "ja") if source_name=="ICNALE" else "ja"
            })
    return pd.DataFrame(records), texts

# ===== データ読み込み =====
df_llm, llm_texts = load_annotations(LLM_FILE, "LLM")
df_icnale, icnale_texts = load_annotations(ICNALE_FILE, "ICNALE")
df_all = pd.concat([df_llm, df_icnale], ignore_index=True)

# ===== 基本統計 =====
print("=== L2 Annotation Comparative Analysis ===")
print(f"LLM Essays: {df_llm['essay_id'].nunique()}, Annotations: {len(df_llm)}")
print(f"ICNALE Essays: {df_icnale['essay_id'].nunique()}, Annotations: {len(df_icnale)}")

# ===== 観点ごとの出現頻度比較 =====
aspect_counts = (
    df_all.groupby(["source", "aspect"])
    .size()
    .reset_index(name="count")
    .pivot(index="aspect", columns="source", values="count")
    .fillna(0)
    .astype(int)
)
print("\n[観点別アノテーション数比較]")
print(aspect_counts.to_markdown())

# ===== True/False（grammar_correctness）の比率比較 =====
correctness_pivot = (
    df_all.groupby(["source", "aspect", "grammar_correctness"])
    .size()
    .reset_index(name="count")
    .pivot_table(
        index=["aspect"],
        columns=["source", "grammar_correctness"],
        values="count",
        fill_value=0
    )
)
print("\n[True/False（grammar correctness）分布比較]")
print(correctness_pivot.to_markdown())

# ===== 差分の可視化（割合） =====
aspect_share = aspect_counts.div(aspect_counts.sum(axis=0), axis=1) * 100
print("\n[観点ごとの割合（%）比較]")
print(aspect_share.round(2).to_markdown())

# ==============================
# 情報理論的解析
# ==============================
def get_words(texts):
    words = []
    for text in texts:
        words.extend(text.split())
    return words

def word_probs(words):
    total = len(words)
    counts = Counter(words)
    return {w: c/total for w, c in counts.items()}

def entropy(prob_dist):
    return -sum(p * math.log2(p) for p in prob_dist.values())

def kl_divergence(P, Q, epsilon=1e-12):
    all_words = set(P.keys()) | set(Q.keys())
    kl = 0.0
    for w in all_words:
        p = P.get(w, epsilon)
        q = Q.get(w, epsilon)
        kl += p * math.log2(p / q)
    return kl

# 単語分布とエントロピー
P = word_probs(get_words(icnale_texts))
Q = word_probs(get_words(llm_texts))
H_icnale = entropy(P)
H_llm = entropy(Q)
KL = kl_divergence(P, Q)

print("\n=== Lexical Information-Theoretic Analysis ===")
print(f"ICNALE Lexical Entropy: {H_icnale:.4f}")
print(f"LLM Lexical Entropy: {H_llm:.4f}")
print(f"KL Divergence (ICNALE || LLM): {KL:.4f}")

# ==============================
# 条件付き相互情報量 I(X;Y|D)
# X: L1, Y: aspect, D: essay_id
# ==============================
def conditional_mutual_info(df):
    total = len(df)
    # P(x,y,d)
    p_xyz = Counter(tuple(x) for x in df[["L1","aspect","essay_id"]].values)
    # P(x|d) and P(y|d)
    p_xd = defaultdict(lambda: 0)
    p_yd = defaultdict(lambda: 0)
    p_d = defaultdict(lambda: 0)
    for _, row in df.iterrows():
        x, y, d = row["L1"], row["aspect"], row["essay_id"]
        p_xyz[(x,y,d)] += 1
        p_xd[(x,d)] += 1
        p_yd[(y,d)] += 1
        p_d[d] += 1
    I = 0.0
    for (x,y,d), c_xyz in p_xyz.items():
        pxyd = c_xyz / total
        pxd = p_xd[(x,d)] / total
        pyd = p_yd[(y,d)] / total
        pd = p_d[d] / total
        I += pxyd * math.log2(pxyd / (pxd * pyd + 1e-12) + 1e-12)
    return I

I_icnale = conditional_mutual_info(df_icnale)
I_llm = conditional_mutual_info(df_llm)
delta_I = I_llm - I_icnale

print("\n=== L1-Dependent Bias Analysis (Conditional Mutual Information) ===")
print(f"I(ICNALE) = {I_icnale:.4f}")
print(f"I(LLM)    = {I_llm:.4f}")
print(f"ΔI        = {delta_I:.4f}  (0に近いほど人間L2のL1依存パターンを忠実に模倣)")

# ===== 文長分布 =====
def sentence_lengths(texts):
    lengths = []
    for t in texts:
        for s in t.split('.'):
            words = s.strip().split()
            if words:
                lengths.append(len(words))
    return lengths

icnale_lens = sentence_lengths(icnale_texts)
llm_lens = sentence_lengths(llm_texts)

plt.hist([icnale_lens, llm_lens], bins=20, label=["ICNALE", "LLM"], alpha=0.7)
plt.xlabel("Sentence length (words)")
plt.ylabel("Frequency")
plt.title("Sentence Length Distribution")
plt.legend()
plt.show()
