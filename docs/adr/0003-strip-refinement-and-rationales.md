# 3. Knowledge strips are SciFact sentences, and the kept strips are the rationale

Status: accepted

## Context

CRAG's refinement decomposes each retrieved passage into small "knowledge strips", scores
each strip's relevance, drops the irrelevant ones and recomposes the rest. SciFact asks for
something very close: a label per abstract plus the rationale sentences (at most three
count) that justify it, scored by sentence id.

## Decision

* A strip is one SciFact sentence. Fragments shorter than four words (section headers such
  as `METHODS`) are merged into the following sentence, so the relevance model never scores
  a header alone. Each strip keeps the sentence ids it covers; its *anchor* is the sentence
  that closed it.
* Strip relevance is the ms-marco cross-encoder score of (claim, strip). The cut-off is
  fitted on train by maximising sentence-selection F1 over the top-3 abstracts, and at most
  three strips are kept per abstract.
* The kept strips are recomposed in document order into one passage, which the NLI model
  reads once (premise = passage, hypothesis = claim) to label the abstract.
* The anchors of the kept strips are reported as the rationale: decomposition and
  rationale selection are the same operation.

The comparison systems without refinement read every sentence separately and take the
maximum entailment and contradiction per abstract; their rationale is the best sentence
plus up to two more with probability at least 0.5 for the chosen label.

## Consequences

* An abstract whose strips all fall below the cut-off is filtered out (NOINFO for that
  abstract) before the NLI model reads it, so refinement doubles as a relevance filter.
* Strip ids always map back to SciFact sentence ids (property-tested), so predictions feed
  the official evaluator directly.
* Relevance is judged by a passage-ranking model, not by the NLI model; the two can
  disagree, which is exactly what the "CRAG without strip refinement" row measures.

## Alternatives rejected

* **Fixed windows of N tokens, as in the CRAG paper.** Windows straddle sentence
  boundaries, so their ids cannot be reported as SciFact rationales.
* **Feeding the whole abstract to the NLI model.** MNLI-style models were trained on short
  premises, and truncation at 256 tokens would cut off late rationales.
