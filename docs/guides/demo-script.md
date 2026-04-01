# Presentation Demo Script

**Event:** April 19 presentation
**Duration:** 16 minutes (2 opening + 12 demo + 1 Q&A intro + 1 closing)
**Setup:** Python 3.10+, `pip install mltk[cli]`,
terminal visible to audience.

---

## 1. Pre-Presentation Checklist

### Terminal Setup
- [ ] Font size: 18pt or larger (audience must read from the back)
- [ ] Color scheme: dark background with high-contrast colors (the demo uses ANSI green/red/yellow/cyan -- test that they render well on the projector)
- [ ] Window: maximized, ~80 columns x 40 rows minimum
- [ ] Hide desktop notifications, Slack, email -- nothing should pop up mid-demo
- [ ] Disable screen saver and power sleep

### Warm the Cache
- [ ] Open a terminal in the `mltk` project directory
- [ ] Run `python demo/run_demo.py` once end-to-end
- [ ] Confirm all 6 beats complete without errors
- [ ] Note the total demo time (should be well under 1 second)

### VS Code
- [ ] Open VS Code with the `mltk-vscode` extension active
- [ ] Have `demo/run_demo.py` open in a tab (you can show the source if someone asks "how many lines of code is this?")
- [ ] Have a sample pytest test file open in another tab to show inline results if asked

### Backup Plan
- [ ] Save the full demo output to `demo/demo-output-backup.txt`:
  ```bash
  python demo/run_demo.py > demo/demo-output-backup.txt 2>&1
  ```
  If the live run crashes, open this file and walk through it. The audience will never know.
- [ ] Have `mltk doctor` output ready as a second fallback

### Final Check (5 minutes before)
- [ ] Clear the terminal
- [ ] Type `python demo/run_demo.py` but do NOT press Enter yet
- [ ] Take a breath. You know this material.

---

## 2. Opening (2 min)

### Hook (30 sec)
> "ML systems fail silently. How many of you have deployed a model that passed all unit tests -- every single one green -- but was wrong in production?"

Pause. Let hands go up. Nod.

> "You're not alone. This is the default state of ML in 2026. We test our code, but we don't test our models."

### Problem Statement (45 sec)
> "Traditional testing catches code bugs. If a function throws an exception, pytest finds it. But ML failures are different. Your model can train on data with PII leaking in. It can score 92% overall but collapse to 73% on a subgroup nobody checked. It can give a completely different answer to the same question phrased two different ways. And every test stays green."

### The One-Liner (15 sec)
> "That's why we built mltk. The short version: **pytest catches code bugs. mltk catches ML bugs.**"

### Stats (30 sec)
> "224 assertions covering the full ML lifecycle. Data quality, model validation, behavioral consistency, RAG evaluation, red team security, compliance, production monitoring. Native pytest integration -- it's a plugin, not a separate tool. One pip install. Today I'll show you what that looks like in practice."

---

## 3. Per-Beat Talking Points (12 min total)

---

### Beat 1: "The Problem -- ML Fails Silently" (~1 min 30 sec)

#### BEFORE running (set the stage, 30 sec)
> "Let me show you a typical ML dataset. Five rows. It looks clean. I want you to tell me: would you ship this to production?"

*Press Enter to start the demo. Beat 1 runs automatically.*

#### WHILE it runs (point at the output)
- When the DataFrame prints: "Five rows, five columns. Names, emails, ages, scores, group labels. Looks perfectly normal."
- When the hidden issues list appears: "But mltk found three problems. PII in the text columns -- names and emails leaking into training data. A null in the age column that will crash your feature pipeline. And subgroup bias in the scores."
- **Key moment -- pause here:** When `Group A: avg score = 0.85` and `Group B: avg score = 0.73` appear, point at the screen.

> "The model looks great at 92% overall. But Group B is at 73%. That's a 12-point gap. If Group B is a protected demographic, you just shipped a biased model. And no unit test in the world would have caught this."

#### AFTER (takeaway, 15 sec)
> "This is the problem. ML systems fail in ways that traditional testing cannot see. Let me show you how mltk finds them."

#### Transition
> "Let's run mltk's data assertions on this exact dataset."

---

### Beat 2: "mltk Data Scan" (~2 min)

#### BEFORE running (15 sec)
> "Three assertions. Three lines of code. Watch how fast this goes."

*Beat 2 starts automatically after Beat 1.*

#### WHILE it runs (point at each result)
- **Schema check (PASS):** "Schema validation is the first gate. Column types match what we declared. Green."
- **Null check (FAIL):** Point at the FAIL badge. "One null in the age column. In production, this crashes your feature engineering pipeline silently -- or worse, it fills in a default value and nobody notices."
  - Point at the WHY/FIX box: "mltk doesn't just find the problem. It tells you why it matters and how to fix it."
- **PII check (FAIL):** Point at the FAIL badge. "Five email addresses in the training data. That's a GDPR violation, a CCPA violation. Regex catches structured patterns -- emails, SSNs, API keys -- with zero dependencies."
  - When the NER note appears: "For person names, you switch to `method='hybrid'`. One parameter change. That uses NER under the hood."

> "Three lines of code found PII and nulls. No ML expertise needed. If you can write a pytest test, you can use mltk."

#### AFTER (takeaway, 15 sec)
> "Data quality is the foundation. But what about the model itself?"

#### Transition
> "Data bugs are one thing. Behavioral bugs are another. This next one is our first-mover feature."

---

### Beat 3: "Behavioral Consistency" (~3 min) -- THE WOW MOMENT

#### BEFORE running (30 sec)
> "Behavioral consistency testing. **No other tool ships this as pytest assertions.** Not DeepEval, not Promptfoo, not Giskard. Nobody."

> "Here's the research: NAACL 2025 showed that simply rephrasing a question can swing model accuracy by 10%. Same question, different words, completely different answer. That's a production bug, and nobody tests for it."

*Beat 3 starts automatically.*

#### WHILE it runs -- SLOW DOWN, THIS IS THE PEAK

**Paraphrase invariance section:**
- When the four paraphrases and their outputs print, read them aloud:
> "Same question, four phrasings. 'What caused WW2?' 'Summarize the causes of World War 2.' 'What led to the second world war?' 'Explain the origins of WW2.'"

- **Point at the DIVERGENT output.** Raise your voice slightly:
> "Three of them give the right answer. But 'What led to the second world war' -- same question, different phrasing -- returns 'It started in 1939.' **Completely different answer.** The model memorized phrasing, not concepts."

- Point at the worst pair score: "Worst pair score: 0.00. Zero overlap. That's not a marginal failure. That's a catastrophic inconsistency."

**Output stability section:**
- "Same prompt, five runs. Run 3 adds ', FR' to the end. Stability drops below threshold."
- "In production, that means your users get different answers depending on when they ask."

#### AFTER (takeaway, 20 sec)
> "No other tool tests this. You write `assert_paraphrase_invariance`, push to CI, and every commit is checked for behavioral consistency. That's our first-mover advantage."

#### Transition
> "We can test models. But where does the test data come from? Let me show you automated test generation."

---

### Beat 4: "Synthetic QA + RAG Testing" (~2 min 30 sec)

#### BEFORE running (20 sec)
> "The generate-then-test pipeline. Step one: feed your documents to `SyntheticQAGenerator`. Step two: it extracts key facts and builds question-answer pairs. Step three: test your RAG system against them. Fully automated. Zero human labeling. This runs in CI."

*Beat 4 starts automatically.*

#### WHILE it runs
- When the QA pairs print: "Five document chunks in, five QA pairs out. Template mode -- deterministic, no API calls, no cost. For production, you swap in your own LLM for higher-quality generation. One parameter change."
- When the faithfulness scores print: "Now we test: does the RAG answer stay faithful to the source document? Lexical overlap scoring. All five pass."
- Point at the score values: "1.00 faithfulness. The answer comes directly from the source. If your retriever pulls the wrong chunk, this drops and you know immediately."

#### AFTER (takeaway, 15 sec)
> "Generate test data, test RAG faithfulness, zero human labeling. This entire pipeline runs in CI with no external APIs."

#### Transition
> "We've tested data quality, model behavior, and RAG accuracy. Now let's attack the model."

---

### Beat 5: "Red Team Security Scan" (~2 min)

#### BEFORE running (20 sec)
> "Red team security scanning. Think of it as penetration testing for LLMs. We throw attack payloads at your model -- prompt injection, jailbreaks, data extraction attempts, encoding mutations -- and measure how many get through."

*Beat 5 starts automatically.*

#### WHILE it runs
- When attack categories print: "Four attack vectors. Prompt injection -- 'ignore your instructions.' Jailbreak -- 'enter DAN mode.' Data extraction -- 'repeat your system prompt.' And encoding mutations -- base64, ROT13, leetspeak to bypass keyword filters."
- When the per-category breakdown appears: "26 catalog attacks across three categories. 100% resilience on all three. This is a mock chatbot with keyword filtering -- a real model probably won't score this clean."
- When encoding mutations run: "Now the clever attacks. Same malicious prompt, but encoded in base64 or ROT13. Attackers use this to bypass safety filters. 30 mutations tested."

#### AFTER (takeaway, 15 sec)
> "56 attacks in under a second. This is your security gate before deployment. All built-in, all pytest-native. No external attack tools needed."

#### Transition
> "Let's see the full picture."

---

### Beat 6: "The Full Picture" (~1 min)

#### BEFORE running (5 sec)
*Beat 6 starts automatically. Let the summary box render.*

#### WHILE it runs
- Point at each line of the summary box:
  - "4 issues found." (pause)
  - "8 tests passed, 4 failed." (pause)
  - "56 security checks." (pause)
  - Point at total time: "All of this in under a second."

#### AFTER (the close, 30 sec)
> "224 assertions. One pip install. Native pytest. You don't learn a new tool -- you write `assert_paraphrase_invariance` the same way you write `assert True`. It runs in CI, it runs locally, it runs in your existing test suite."

> "This is pytest for ML."

*Let the closing stars and `pip install mltk[cli]` render. Pause. Let it land.*

---

## 4. Q&A Prep (Common Questions)

### "How does this compare to DeepEval / RAGAS / Promptfoo?"

> "They're LLM-only tools. DeepEval has ~50 LLM metrics, Promptfoo has prompt evaluation with 135 plugins. Both are strong in their lane. But they don't do data quality, drift detection, training bug detection, fairness testing, or compliance frameworks. mltk covers the full ML lifecycle -- 224 assertions from data ingestion to production monitoring. Plus we have 7 behavioral consistency assertions that nobody else ships. And our core has only 2 dependencies -- numpy and pandas. DeepEval requires an LLM for most assertions; mltk works offline by default."

### "Can I use this with my existing pytest suite?"

> "Yes, that's the point. It's a pytest plugin. `pip install mltk` and it auto-registers. Your existing `conftest.py`, fixtures, markers, CI pipeline -- everything works. You just add `from mltk.data import assert_no_nulls` and write tests like you normally do. No new CLI to learn, no separate dashboard to check."

### "What about performance overhead?"

> "The demo you just saw ran 12 tests including 56 security payloads in under a second. For drift detection and BERTScore, we have a Rust backend via PyO3 that gives 10-100x speedup over pure Python. It falls back to scipy/numpy automatically if Rust isn't compiled. In CI, the overhead is negligible compared to model training time."

### "Is this open source? What's the license?"

> "Apache 2.0. Fully open source on GitHub and PyPI. The repo is `Liorrr/mltk`. No commercial license, no usage limits, no telemetry."

### "Does it work with LangChain / LlamaIndex?"

> "mltk is framework-agnostic. Any function that takes a string and returns a string works as a `model_fn`. So if you have a LangChain chain, wrap it: `def my_model(prompt): return chain.invoke(prompt)`. Same for LlamaIndex query engines. The RAG assertions work on any retriever output -- you provide the answer and context, mltk evaluates faithfulness and relevancy."

### "What about multimodal models / image generation?"

> "We have image-text alignment assertions today. Full multimodal evaluation -- text-to-image quality, image coherence, editing accuracy -- is on the roadmap. It's a 2-sprint effort. For now, LLM-as-Judge via `assert_llm_judge_score` can evaluate image outputs if you describe them to the judge."

### "How do I add this to my CI pipeline?"

> "Same as any pytest plugin. In your GitHub Actions YAML: `pip install mltk[cli]`, then `pytest -m ml_data` for data tests, `pytest -m ml_model` for model tests. Add `--mltk-report` for an HTML report, `--mltk-export-json` for machine-readable results. We also have a GitHub App integration that posts results as PR comments."

---

## 5. Closing (1 min)

### Recap the 3 Differentiators (30 sec)
> "Three things to remember. First: **breadth**. 224 assertions covering the full ML lifecycle. No other tool does data quality AND model testing AND LLM evaluation AND security AND compliance in one package."

> "Second: **behavioral consistency**. 7 assertions that catch models memorizing phrasing instead of learning concepts. We're the first and only tool shipping this as pytest assertions."

> "Third: **red team plus multimodal, zero dependencies**. 56 attack payloads built in, encoding mutations, OWASP mapping -- all running offline with no external APIs. No other tool combines security scanning and evaluation testing in one pip install."

### Call to Action (15 sec)
> "`pip install mltk`. The repo is `Liorrr/mltk` on GitHub. Apache 2.0. Star it, try it, break it, tell us what's missing."

### Close (15 sec)
> "Questions?"

*Stay at the podium. Don't rush off. The best conversations happen in Q&A.*

---

## 6. Rehearsal Instructions

### Timing Target
| Section | Duration | Cumulative |
|---------|----------|------------|
| Opening | 2:00 | 2:00 |
| Beat 1: The Problem | 1:30 | 3:30 |
| Beat 2: Data Scan | 2:00 | 5:30 |
| Beat 3: Behavioral (PEAK) | 3:00 | 8:30 |
| Beat 4: Synthetic QA | 2:30 | 11:00 |
| Beat 5: Red Team | 2:00 | 13:00 |
| Beat 6: Full Picture | 1:00 | 14:00 |
| Q&A intro + Closing | 2:00 | 16:00 |

### Practice Plan
1. **Run 1 (solo):** Read the talking points aloud while the demo runs. Time yourself. Mark where you run long.
2. **Run 2 (solo):** Same thing, but now cut the long parts. Aim for 15 minutes.
3. **Run 3 (with someone):** Present to a friend or colleague. Ask them: "What was the most impressive part? What was confusing?" Adjust.
4. If you have time, do a **Run 4** with the actual projector/screen setup to check colors and font size.

### The "Skip" Plan
If the demo crashes mid-run:
- Each beat function is independent. Drop into a Python REPL and run the next one:
  ```python
  from demo.run_demo import beat_3
  beat_3()
  ```
- If Python itself fails: open `demo/demo-output-backup.txt` and walk through the saved output. Say: "Let me show you the output from my last run." Nobody will care.
- If everything fails: open `demo/run_demo.py` in VS Code and walk through the source code. The assertions are readable -- `assert_no_pii(df, method="regex")` explains itself.

### Energy Map
- **Beats 1-2:** Steady, informative. You're setting context.
- **Beat 3:** THIS IS THE PEAK. Raise your voice when you point at the DIVERGENT tag. Slow down. Let the silence after "completely different answer" land for 2 full seconds. This is the moment the audience remembers.
- **Beat 4:** Back to steady. This is practical, workmanlike. "It just works."
- **Beat 5:** Slight energy bump -- security is inherently dramatic. "56 attacks in under a second" should sound impressive.
- **Beat 6:** Calm confidence. You've proven the point. Let the numbers speak.

### Common Mistakes to Avoid
- Do NOT read the terminal output verbatim. Summarize and point.
- Do NOT apologize for the demo being "just a mock model." It demonstrates the assertions, which is the point.
- Do NOT go deep on implementation details unless asked. Save it for Q&A.
- Do NOT skip Beat 3. If you're running long, compress Beats 4 and 5 instead. Beat 3 is the differentiator.

---

## Presenter Notes (Quick Reference Card)

Print this section and keep it on the podium:

```
OPENING: "pytest catches code bugs. mltk catches ML bugs."
         224 assertions, native pytest, one pip install.

BEAT 1:  "The model looks great at 92%. But Group B is at 73%."
BEAT 2:  "Three lines of code. PII and nulls. No ML expertise needed."
BEAT 3:  ** PEAK ** "No other tool tests this." Point at DIVERGENT.
         Slow down. Let it land.
BEAT 4:  "Generate test data, test RAG, zero human labeling. Runs in CI."
BEAT 5:  "56 attacks in under a second. Your security gate."
BEAT 6:  "224 assertions. One pip install. pytest for ML."

CLOSE:   pip install mltk | GitHub: Liorrr/mltk | Apache 2.0
         "Questions?"
```

### Fallback Sequence
1. Live demo runs? Great, talk over it.
2. Live demo crashes? Run individual beats from REPL.
3. Python crashes? Show saved output file.
4. Everything crashes? Walk through source code in VS Code.

### If Something Breaks

- The script catches all errors and prints a traceback;
  read the error and explain what the assertion would have done
- Each beat is independent -- skip to the next
- If Python itself fails: show the source code
  in `demo/run_demo.py` and walk through it
