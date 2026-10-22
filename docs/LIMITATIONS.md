# Limitations

Retrieval numbers are from the 22 Oct 2026 bge-small run. Add answer-level failure examples after the LLM eval.

- **No answer-level metrics yet.** Faithfulness, citation precision, abstention and cost are empty until
  `ANTHROPIC_API_KEY` is used for `llmeval run`.
- **Corpus version.** The index holds the original 2016 and 2024 Official Journal texts. The AI Act has been amended
  since (application dates postponed, articles such as 4a, 60a and 75a-d added). An answer can therefore be out of
  date, for example on when obligations apply. Questions about Omnibus articles are in the gold set as unanswerable.
  A production system would version the corpus and show the version in each answer.
- **Gold set quality.** Drafted cases were checked by matching each `reference` regex against the Official Journal
  article text (`verified-oj`). Further cases (`mine`) were written from the same text. The regex still accepts a
  wrong answer that happens to contain the key fact, and rejects a right answer worded differently. Spot-check
  before treating a case as interview-proof.
- **Ids are article-level.** Recall@k counts a hit when the right article is retrieved, even if the relevant
  paragraph is not in the passage given to the model.
- **Embedder window.** bge-small reads at most 512 tokens (about 380 words). The 600-word chunk variant is
  truncated when embedded, so that ablation row measures truncation as much as chunk size.
- **Cross-article questions.** Questions that need two or more articles (for example fines for a specific
  obligation) are the likeliest to fail; the gold set has few of them.
- **Annexes and recitals.** Annexes are indexed as documents. Recitals are not indexed, so interpretive context is
  missing.
- **Legal reasoning.** The system quotes and cites text. It does not apply the law to a situation and gives no
  advice.
- **Prompt injection is mitigated, not solved.** Passages are quoted as data and the system prompt forbids following
  them, and the canary test checks one injection style. Other styles are untested.
- **Citation check is syntactic.** A cited id must have been provided to the model, but the faithfulness judge, not
  the citation scorer, decides whether the cited passage supports the sentence.
- **Single-user scale.** No authentication on `/ask`, no rate limiting, in-memory BM25, one process. `/ingest` is
  disabled unless a token is configured.
- **Cost and latency** are measured on retrieval (p50/p95) and as token totals for answers; no price table is
  included because prices change.
- **Sampling.** Answers and judge replies use temperature 0 only if the installed Anthropic SDK accepts the argument;
  newer SDK versions do not, and then the model's default sampling applies. Repeat runs can differ slightly, which is
  why the evaluation uses 3 trials and the judge reliability study measures run-to-run variance.
