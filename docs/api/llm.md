# LLM Evaluation

Lightweight LLM/GenAI evaluation assertions — no external model dependencies. Covers semantic similarity, toxicity detection, hallucination checking, and LLM-specific latency metrics (TTFT/ITL).

**Module:** `mltk.domains.llm`

---

## Similarity
- `assert_semantic_similarity(references, hypotheses, min_score, method)` — token-level F1 or embedding cosine

## Safety
- `assert_no_toxicity(texts, max_toxic_pct, patterns)` — regex/keyword toxicity detection
- `assert_no_hallucination(claims, sources, method)` — keyword overlap factuality check

## LLM Latency
- `assert_ttft(func, *args, max_ms)` — Time to First Token
- `assert_itl(func, *args, max_ms)` — Inter-Token Latency

---
