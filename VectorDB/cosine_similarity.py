from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

query = "What's the best way to give an LLM long-term memory?"
true_stmt  = "Vector databases are the best way to give an LLM long-term memory."
false_stmt = "Vector databases are NOT the best way to give an LLM long-term memory."

query_emb, true_emb, false_emb = model.encode([query, true_stmt, false_stmt])
cos = lambda x, y: np.dot(x,y) / (np.linalg.norm(x) * np.linalg.norm(y))

print("query vs TRUE:   ", cos(query_emb, true_emb))
print("query vs NEGATED:", cos(query_emb, false_emb))


"""
The negated sentence still scores 0.63, and on this model that's very high. Unrelated sentences usually land near 0 to 0.2. So the model separated true from false by a thin margin, yet both sit far closer to your query than anything off-topic. If your corpus holds only one document on this topic, it gets retrieved whether it's right or wrong. Raw scores also aren't calibrated, so 0.76 doesn't mean "76% relevant". Only the ranking and the margins between candidates carry information.
"""

# Part A: rank a small corpus against the same query

from sentence_transformers import util

docs = [
    "Vector databases are the best way to give an LLM long-term memory.",
    "Vector databases are NOT the best way to give an LLM long-term memory.",
    "Giving an LLM persistent memory usually means storing embeddings and retrieving them.",
    "The recipe calls for two cups of flour and a pinch of salt.",
    "Long-term memory in humans depends on the hippocampus.",
]

scores = util.cos_sim(query_emb, model.encode(docs))[0] # Because query_emb has shape (1, dim), we take the first row to get similarity with each document.
for s,d in sorted(zip(scores.tolist(), docs), reverse=True):
    print(f"Score: {s:.4f}, Document: {d}")


# Part B: swapped detail

query2 = "When is the team meeting?"
d_tue = "The team meeting is on Tuesday at 3pm."
d_thu = "The team meeting is on Thursday at 3pm."

query2_emb, d_tue_emb, d_thu_emb = model.encode([query2, d_tue, d_thu])
print("query2 vs Tuesday: ", util.cos_sim(query2_emb, d_tue_emb).item())
print("query2 vs Thursday:", util.cos_sim(query2_emb, d_thu_emb).item())


"""
The ranking is the finding. Three things stand out:

Wording beat meaning. The hippocampus sentence (wrong domain, shares "long-term memory") scored 0.604, ahead of the paraphrase that actually answers your question (0.600). And the negated sentence beat both.
Only the coarse split was reliable. The model cleanly separated "about this topic" from "the recipe" (-0.01). Within the topic, gaps like 0.604 vs 0.600 or 0.847 vs 0.824 are noise.
Near-duplicates are indistinguishable. Two docs that differ only in the critical fact (Tuesday vs. Thursday) score almost the same. In real systems this is your old policy vs. your new policy.
"""

from sentence_transformers import CrossEncoder
ce = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')  # small download

# Part A: same query, same 5 docs
ce_scores = ce.predict([(query, d) for d in docs])
for s,d in sorted(zip(ce_scores, docs), reverse=True):
    print(f"Score: {s:.4f}, Document: {d}")

query3 = "Is the team meeting on Tuesday?"
print("Reranker: ", ce.predict([(query3, d_tue), (query3, d_thu )]))
print("Bi-encoder: ", util.cos_sim(model.encode(query3), model.encode([d_tue, d_thu])))


