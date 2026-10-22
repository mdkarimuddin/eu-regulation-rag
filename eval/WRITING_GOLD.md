# Writing and verifying the gold set

The assignment asks for 100+ questions: about 60 written by you and about 40 drafted then checked by you.
`gold.yaml` now has 119 cases (95 answerable). The original drafts are tagged `verified-oj` after their `reference`
regex was checked against the Official Journal article text on 22 Oct 2026. Cases tagged `mine` were written from
the same article text. Read a sample of both tags yourself before an interview; a regex match is not a legal review.

## Verify the drafts (do this first)

For each case tagged `draft`, open the official article (EUR-Lex) and check three things:
1. The question has a clear answer in the article text.
2. `gold_sources` names the article that contains the answer (add a second id if two articles are needed).
3. `reference` (a case-insensitive regex) matches a correct answer and would not match a wrong one.
Fix the case, then delete the `draft` tag. For unanswerable cases, check that the corpus really has no answer.

## Write your own 60

Rules that make the set useful:
- Write the question first, then find the article. Ask what a developer or compliance officer would ask, in their
  words, not the article's words. Paraphrased questions are the ones that separate vector from BM25 retrieval.
- Include some that need two articles (give both ids in `gold_sources`) and some that name no article number.
- Include 5-10 near-misses: questions that sound answerable but are not in the text (add `expect_refusal: true`,
  tag `unanswerable`, no `gold_sources`).
- Avoid the AI Act's application dates and anything the Digital Omnibus changed; the corpus is the original text.
- Keep `reference` to one key fact. If the answer is a number, accept the formats a model might use.
- Do not tune the questions after seeing results. Add new ones instead and note the date.

## Template

```yaml
- id: mine-001
  input: "Can I keep customer emails forever if I only use them for newsletters?"
  reference: 'storage limitation|no longer than is necessary'
  match: regex
  gold_sources: [gdpr-art-5]
  tags: [gdpr, mine]
```

Run `eurag check-gold` after editing: it fails on any gold id that is not in the corpus.
