# Compares answers with the report (RAG) against the plain model, scored by Gemini.
import random, re, time
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from esg_core import *

N_EVAL = 12   # set to None to test every question (slower)

chunks, vectors, embed_model = load_index()
companies = sorted({c["company"] for c in chunks})

templates = [
    "What are {c}'s total Scope 1 and Scope 2 greenhouse gas emissions?",
    "What is {c}'s net-zero or carbon neutrality target year?",
    "What share of {c}'s electricity comes from renewable energy?",
    "How much water does {c} withdraw or consume in total?",
    "What percentage of {c}'s employees are women?",
    "How much waste does {c} generate and what share is recycled?",
    "What is {c}'s injury rate or workplace safety record?",
    "What are {c}'s Scope 3 emissions?",
    "What was {c}'s CEO's salary in 2010?",
]
questions = [(t.format(c=pretty(c)), c) for t in templates for c in companies]

random.seed(1)
subset = questions if N_EVAL is None else random.sample(questions, min(N_EVAL, len(questions)))
print(len(questions), "questions available,", len(subset), "will be tested")


def score(prompt):
    out = generate(prompt)
    m = re.search(r"\d*\.?\d+", out)
    return min(1.0, float(m.group())) if m else 0.0


def faithfulness(context, ans):
    return score(
        f"Context:\n{context}\n\nAnswer:\n{ans}\n\n"
        "Give a score from 0 to 1 for how much of the answer is supported by the context. "
        "1 means every claim is in the context, 0 means none is. "
        "If the answer says the information is not disclosed or not available, give 1. "
        "Reply with only the number.")


def relevancy(question, ans):
    return score(
        f"Question: {question}\n\nAnswer:\n{ans}\n\n"
        "Give a score from 0 to 1 for how directly the answer addresses the question. "
        "Reply with only the number.")


rows = []
for q, comp in subset:
    rag_ans, hits = answer(q, chunks, vectors, embed_model, company=comp)
    context = "\n\n".join(h["text"] for h in hits)
    plain_ans = generate(q + " Answer in 2 or 3 sentences.")

    for label, ans in [("RAG", rag_ans), ("Plain AI", plain_ans)]:
        rows.append({"question": q, "system": label, "answer": ans,
                     "faithfulness": faithfulness(context, ans),
                     "relevancy": relevancy(q, ans)})
    print("done:", q)
    time.sleep(1)

df = pd.DataFrame(rows)
df.to_csv("results.csv", index=False)
summary = df.groupby("system")[["faithfulness", "relevancy"]].mean().round(2)
print(summary)

summary.plot(kind="bar", ylim=(0, 1), rot=0, figsize=(6, 4), title="RAG vs plain AI")
plt.ylabel("average score (0 to 1)")
plt.tight_layout()
plt.savefig("rag_vs_plain.png", dpi=150)
print("saved results.csv and rag_vs_plain.png")
