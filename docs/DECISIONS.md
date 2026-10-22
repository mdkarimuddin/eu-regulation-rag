# Design decisions

**One document per article, ids like `gdpr-art-17`.** Legal questions are answered at article level, citations are
human-checkable, and the gold set can name sources without pointing at chunk numbers that change with chunk size.

**Parse plain text, not EUR-Lex CSS classes.** The splitter works on lines such as `Article 17` followed by a title.
It survives markup changes and is tested on a fixture. It is validated by counting articles (99 and 113) and fails
loudly when any are missing, so a bad download cannot silently produce a smaller corpus.

**Heading inside every chunk** (`GDPR Article 17 - Right to erasure`). The embedding and the prompt both carry the
provenance, which helps queries that name an article or a topic.

**Hybrid retrieval with reciprocal rank fusion.** Legal text mixes exact terms ("72 hours", "Article 83") that BM25
handles well with paraphrased questions that vectors handle well. RRF needs no score calibration between the two.
On the 22 Oct 2026 run (95 questions, bge-small) hybrid recall@5 was 0.947 vs 0.926 vector-only and 0.911 BM25.
The cross-encoder reranker dropped recall@5 to 0.937 and raised p95 from 17 ms to 642 ms, so it is off by default.

**Recall measured on a 30-chunk candidate list collapsed to articles**, while only 6 chunks go to the model. This
separates "was the article found" from "did it fit in the prompt".

**Citations are ids the model was given.** `parse_answer` drops ids that were not in the prompt and reports them as
invalid, so a made-up citation cannot count as correct.

**`NOT_FOUND` sentinel for abstention.** Unambiguous to parse, and converted to a fixed refusal sentence that the
harness's safety scorer recognises.

**Injection handling in layers:** passages wrapped in tags with closing tags neutralised, a system rule to treat them
as data, a canary document in the index, and a failing test condition if the canary string reaches an answer.

**Index as files by default, pgvector optional.** The corpus is a few thousand chunks, so a numpy matrix is enough
and keeps tests and the Docker image simple. The pgvector backend exists because production deployments use a
database; it is exercised in CI against a real Postgres.

**Hashing embedder in tests.** Deterministic and instant, but lexical. It tests the plumbing; it says nothing about
retrieval quality, which only the real embedder run can show.

**Gold set in the harness's YAML format**, so `llmeval` runs it directly and the same file drives retrieval-only and
full-answer evaluations.
