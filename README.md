# Knowledge Gap Detection — Dataset Pipeline

This repository builds the training dataset for a classifier that
distinguishes **true epistemic gaps** ("science doesn't know X yet, and X
is a target for future research") from statements that merely *look* like
gaps in scientific writing about pediatric epilepsy genetics.

## 1. What counts as a knowledge gap

Full rules are in `Annotation Guidelines.html` (not committed here); this
is the working summary the pipeline and its labels are built against.

**Epistemic Gap (label = 1)** — a missing/incomplete piece of knowledge
that requires scientific investigation to resolve. Two forms:
- *Gap-oriented*: states something is unknown/unclear/understudied.
- *Goal-oriented*: forward-looking, names a concrete scientific target
  ("further studies are needed to determine X").

Everything else is **label = 0**, including three categories that are
easy to mistake for gaps:

1. **Forward-looking ≠ research goal.** "Future work" that's a practical,
   clinical, or methodological fix (bigger sample, better imaging,
   improved cell models) rather than new scientific knowledge is a
   *Practical Recommendation*, not a gap.
2. **Speculation alone isn't a gap.** Hedged language ("may", "suggests",
   "possibly") is normal scientific writing. It only counts as a gap when
   paired with an *explicit call for research* ("further investigation is
   warranted", "should be tested by..."). A real example from the data
   (article `10034091`) shows this rule in action: the same first
   sentence is annotated negative alone (bare speculation) and becomes
   positive once the next sentence's explicit call for research is
   appended.
3. **"Unknown"/"uncertain" as a classification label ≠ a gap.** ("epilepsy
   of unknown etiology" used to bucket cases is not a research target.)

There's also a **Non-Epistemic Gap** family — methodological, diagnostic,
therapeutic, data-scarcity, and study-limitation statements — that
defaults to negative and is only promoted to positive when explicitly
framed as a future research target (vs. describing a present limitation).
The `label` field on negative annotations in the source JSON
(`Speculative statement`, `FP`, `Addressed GFK`, `NEG: Therapeutic gap`,
`NEG: Diagnostic Gap`, `NEG: Methodological Gap`, `NEG: Data Scarcity`,
`NEG: Study Limitation`, `Practical recommendation`) records which of
these categories each negative example falls under; `Addressed GFK` marks
a gap that *was* real but this study already resolved it.

## 2. Source data audit

`data/raw/ongoing_ignorannotations_balanced.json` — 100 articles (each
abstract + introduction + discussion only, not the full text), manually
annotated with positive and negative knowledge-gap statements.

Findings from a full-file audit (not a sample):

- 420 positive / 443 negative annotations, genuinely balanced (one
  annotation was reclassified after model error analysis — see below).
- All 863 annotations are exact, unique substring matches inside their
  `source_section` text — 0 mismatches, 0 anchors appearing more than
  once in a section. This is the foundation the whole offset-based
  pipeline depends on.
- Section keys are **not normalized**: Introduction/Discussion appear as
  15 distinct strings across the corpus (`Introduction`, `1. Introduction`,
  `INTRODUCTION`, `RESULTS AND DISCUSSION`, `4 Discussions`, etc.).
  `source_section` on each annotation always matches the exact key used
  for that article, so lookup is always by literal per-article key, never
  a fixed assumed name.
- Abstract present in all 100 articles; Introduction missing in 11,
  Discussion missing in 9 (4 abstracts present-but-empty). These sections
  simply contribute no anchors/windows for that article.
- Only 1 case of a positive/negative annotation span overlap in the same
  article+section — and it's the deliberate "speculation vs. speculation +
  explicit call for research" example above, not a data error.

**One annotation correction** (found via model error analysis, not the
initial audit): article `10195938`'s `Discussion`-section positive
annotation — *"The main limitation of this study is its retrospective
character... [pure study-limitation description, no forward-looking
research language anywhere in the span]"* — was flagged because
`biobert` confidently misclassified it as negative, and reading the
actual `Annotation Guidelines.html` confirmed this exact phrasing
(word-for-word) is used there as a **negative** example of a Study
Limitation statement (no epistemic implication, no unfinished-work
framing). Reclassified from `positives` to `negatives` with
`"label": "NEG: Study Limitation"`. A parallel audit of five other
`positives` containing "limitation" found all five correctly labeled —
each has genuine forward-looking/goal-oriented language justifying the
positive label (e.g. `10357150`, which was initially also flagged as
borderline, matches the guideline's own worked example of a limitation
statement that *does* qualify as a Goal for Knowledge Gap). A parallel
audit of the model's false positives (case-report "not yet
documented/reported" language) found these are correctly labeled too —
5 of 6 are explicitly tagged `FP` in the source data, meaning the
annotators deliberately included them as hard negatives to test exactly
this boundary, not a data error. Net effect: 1 annotation moved out of
863; dataset regenerated (`data/processed/eval.csv`/`.jsonl` changed,
`train.csv`/`.jsonl` unchanged since the article falls entirely in the
eval split), all 15 QA checks in `check_dataset.ipynb` still pass.

## 3. Sentence segmentation

Extracted section text has recurring artifacts that break plain sentence
segmentation:

1. **Missing boundaries**: `"...enhanced NaV1.1 activity.Here, we aim..."`
   — no space after the period.
2. **Structural headings glued to text**: `"AbstractObjectiveMonogenic
   epilepsies..."`, and a CARE-case-report-style colon variant:
   `"Rationale:CUL3 (OMIM: 603136) encodes..."`.
3. **Biomedical notation that looks like a missing boundary but isn't**:
   HGVS variant notation (`p.Asp920Glu`, `c.400G>A`) — same
   `[letter].[Upper]` shape as (1), must not be split.
4. **Embedded literal newlines** from citation-superscript extraction,
   e.g. `"...PGM3,4\nCHD4,5\nUNC13B,6\n..."` — a plain sentence segmenter
   (pysbd) treats these as paragraph breaks and shreds the sentence into
   one-token fragments.
5. **Stray multi-number citation markers**, in two positions:
   - Between two real sentences, e.g. `"...NaV1.1 activity.20\n, \n21\n It
     is hypothesized..."` (originally inline superscript `"activity.20,21"`
     citing references 20 and 21). Neither pysbd nor the boundary-repair
     rule creates a cut point here (pysbd doesn't split before a digit, to
     avoid breaking decimals/numbered lists; the digit isn't uppercase so
     the repair rule doesn't fire either), so the marker gets glued onto
     the front of the following sentence instead of being removed.
   - **Mid-sentence**, e.g. `"...milder GEFS+ phenotypes,\n17\n, \n18\n
     albeit sometimes intensified..."` — same extraction artifact, but
     sitting inside a sentence rather than between two. Found via manual
     spot-checking of `train.csv`.

Implementation (`src/pipeline/segmentation.py`), all offset-preserving —
nothing is ever inserted or deleted, so sentence spans are always valid
character offsets into the *original, untouched* section text:

- **Heading stripping**: a curated list of structural/admin heading words
  (`Background`, `Objective`, `Methods`, ..., `Rationale`, `Patient
  concern`, `Diagnoses`, ..., `Ethical approval`, `Supplementary
  Information`, ...) is matched only when *glued* — no separating
  whitespace, signature of the extraction artifact — via a fixed-point
  regex loop (needed because chained headings like `AbstractObjective`
  can only reveal the second heading once the first is masked out for
  re-scanning). Matched spans are excluded from the sentence list
  entirely, never counted as content.
- **HGVS/abbreviation protection**: reference-sequence prefixes `c. g. m.
  n. p. r.` and common citation abbreviations (`et al.`, `Fig.`, `vs.`,
  ...) are excluded from boundary repair.
- **Boundary repair**: any other `[letter-or-digit].{Upper}` glue
  (broadened from letters-only after finding real misses like
  `"...of GABRA1.The two missense..."`, gene symbols ending in digits) is
  treated as a mandatory cut point.
- **Whitespace normalization for the underlying segmenter** (pysbd): every
  whitespace character, including stray embedded newlines, is replaced
  1-for-1 with a plain space in a length-preserving copy before running
  pysbd, so those newlines stop being read as paragraph breaks. Only this
  copy is used to get pysbd's boundary offsets; nothing downstream ever
  reads from it.
- **Citation-marker stripping**: treated the same way as a heading span
  (cut points at both ends, excluded from sentence content) whenever a
  span, immediately after sentence-ending punctuation and immediately
  before the next sentence's capital letter, matches one of two shapes:
  (a) **two or more** short comma/newline-separated number groups
  (`".20,21 It..."`), or (b) a **single** number group but only when
  bookended by a literal newline on both sides (`".\n19\n Similar..."`,
  a further extraction-artifact signature found via manual spot-checking
  of `train.csv`). A lone number *without* the newline bookend
  (`".9 However..."`) is deliberately left alone: gene/variant
  identifiers routinely end in one digit followed directly by a comma
  (`"PCDH19,"`, `"Cys182,"`, `"RBFOX1,"`), and without the double-newline
  signal there's no safe way to tell those apart from a citation marker —
  checked against the full corpus, the newline-bookended shape has 0
  collisions with gene/variant-identifier text.
- **Mid-sentence citation-marker stripping**: a different mechanism from
  the above, since a mid-sentence marker doesn't sit at a sentence
  boundary — instead of a cut point, the matched span is removed directly
  from the assembled window text (in `build_dataset.py`, after sentences
  are joined), replaced with a single space. Fires only when a span,
  immediately after a comma and immediately before a lowercase letter,
  has **2 or more** digit groups **and** contains an actual embedded
  newline. Both conditions matter: "after a comma" alone collides with
  ordinary comma-separated number lists (`"Patients 2, 5, 10 and 11"`,
  reaction times like `"3 h"`); requiring 2+ groups and a real newline
  together, checked against the full corpus, leaves 0 collisions across
  19 genuine matches. A single mid-sentence number (`",\n3\n of
  which..."`) is deliberately left alone, same reasoning as the
  single-number case above.
- Final sentence spans = the union of pysbd's boundaries, the repair cut
  points, the heading-span boundaries, and the citation-marker-span
  boundaries, split into segments, trimmed, with pure-heading and
  pure-punctuation fragments dropped.

**Verified on the full corpus**: 0 residual heading leftovers, 0 residual
unprotected glued boundaries (the only `[letter].{Upper}` pattern left
in any output sentence is the intentionally-protected `p.` HGVS prefix,
291 occurrences, all genuine variant notation), 0 residual bare-digit
citation-marker fragments surviving as their own sentence (down from 303
multi-number + 35 newline-bookended single-number occurrences in the raw
corpus).

**Known remaining gap** (documented, not fixed): a single citation number
with no newline bookend — whether between sentences (`"...and SYNGAP1.9
However..."`), mid-sentence after a comma (`"...factors,\n3\n of
which..."`), or glued directly to a preceding word with no punctuation
anchor at all (`"...database\n15\n showed..."`) — is left alone. In every
one of these positions, a lone digit is indistinguishable from a decimal
number or a real gene/variant identifier without risking real content,
and in the last position there isn't even a reliable punctuation anchor
to check. A rough full-corpus scan after the fixes above still finds
~116 windows with some residual single-number artifact of this kind, so
it's a real, present limitation, not a hypothetical one — but extending
further would mean guessing at increasingly ambiguous shapes, including
ones that would actively destroy real content if matched wrong (e.g. the
statistical symbol "I²" and genuine citation years like "(Nelson et al.,
2022 Preprint)" both surfaced as near-miss shapes while checking this).
This under-splits/under-cleans rather than corrupts — worth revisiting
with a more targeted approach if it turns out to affect anchor snapping,
but it did not in this corpus (0/863 anchors affected).

## 4. Anchor snapping

`src/pipeline/anchors.py`: for each annotation, the exact statement
(already known to be a unique substring — see the audit) is located via
`str.find`, then snapped to the smallest contiguous run of segmented
sentences that fully overlaps it. Verified on all 863 annotations: 0
not-found, 0 ambiguous, 0 with no overlapping sentence.

## 5. Window generation

`src/pipeline/windows.py`. Two different sizing rules:

- **Anchor-centered windows** (used for both positive anchors and
  standalone negative anchors): size = `len(anchor_sentences) + 2`, with
  the extra 2 sentences distributed 3 ways — `2-before`, `1-before +
  1-after`, `2-after` — clipped (not skipped) at section boundaries. A
  2-sentence anchor yields 4-sentence windows; a 5-sentence anchor yields
  7-sentence windows.
- **Nearby-negative blocks** (flanking a positive anchor): a *fixed* 3
  sentences immediately before, and a fixed 3 immediately after,
  independent of the anchor's own length. A side is generated with fewer
  than 3 sentences if that's all that's available, and skipped entirely
  only if zero sentences are available on that side.

So each positive anchor yields 3 positive windows + up to 2 nearby-negative
candidate windows; each negative anchor yields 3 candidate negative
windows.

**Resolution pass** (applied to every negative-labeled candidate,
regardless of which mechanism produced it, checked against every positive
anchor in the same section):
- Fully contains a positive anchor → promoted to label 1 (positive wins,
  even if the same window also partially brushes a *different* anchor).
- Partially overlaps a positive anchor without fully containing any →
  discarded (ambiguous, not confidently either label).
- No overlap → stays a negative example.

**Window text** = the original sentence substrings (verbatim, untouched)
joined with a single space — not a raw contiguous slice of the original
text, so that any structural heading spans skipped over between two
selected sentences don't leak back into the training text.

**Deduplication**: identical window text collapses to one example; if the
same text appears with both labels, the positive copy wins.

## 6. Train/eval split

`src/pipeline/split.py`: articles are split 80/20 (seed 42) *before*
augmentation, so no paper's sentences appear in both sets.

## 7. Running it

```
pip install -r requirements.txt
python build_dataset.py
```

Outputs `data/processed/train.jsonl` and `data/processed/eval.jsonl`, one
JSON object per line:

```json
{
  "article_id": "10018541",
  "section": "Discussion",
  "text": "...",
  "label": 1,
  "source": "positive_anchor",
  "variant": "1before_1after",
  "promoted": false,
  "anchor_text": "...",
  "n_sentences": 3
}
```

`source` is one of `positive_anchor` / `negative_anchor` /
`nearby_negative`; `promoted` marks windows that flipped from candidate
negative to positive via the positive-wins rule.

## 8. Current dataset (seed 42, 80/20 split)

- 100 articles → 80 train / 20 eval.
- 3379 candidate windows generated → 26 discarded (partial anchor
  overlap) → 426 promoted to positive (positive-wins) → 793 collapsed as
  exact-text duplicates → **2560 final windows**.
- Train: 1877 windows (876 positive / 1001 negative) — unaffected by the
  annotation correction in section 2, since that article falls entirely
  in the eval split.
- Eval: 683 windows (306 positive / 377 negative).
- Window length: min 8 words, median 71, p95 119, max 254 — the long tail
  is worth accounting for in max-sequence-length choice once we pick a
  tokenizer/model, since a 254-word biomedical window will run well past
  a typical 512-subword-token BERT budget once split.

## 9. Model training

Per the supervisor's requirement to compare several models, `train_model.py`
fine-tunes each as a binary sequence classifier (encoder + classification
head, via `AutoModelForSequenceClassification`) on `train.csv`, evaluates
on `eval.csv`. One shared script, parameterized by model name, so every
model trains under identical conditions (same hyperparameters, same data,
same metric computation) — that matters for the comparison to be fair.

**Models compared:**
- `bert` → `bert-base-cased` — generic baseline, not biomedical-specific.
  Included so domain adaptation can be *shown* to help, not assumed.
- `biobert` → `dmis-lab/biobert-base-cased-v1.2` — BERT continued-pretrained
  on PubMed abstracts + PMC full text.
- `pubmedbert` → `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext`
  (renamed from "PubMedBERT"; old identifier still resolves) — trained
  from scratch on PubMed text with its own biomedical vocabulary, rather
  than continued-pretrained from generic BERT. Only ships **uncased** — no
  cased variant exists, unlike the other two, so this one model loses
  case information (e.g. `NPRL3` vs lowercase) that the tokenizer would
  otherwise preserve. A real, model-specific limitation, not a choice.

**No extra text cleanup at tokenization time.** The artifact repair
(glued sentences, headings, citation markers) already happened during
dataset construction — see sections 3 and 5 above — specifically because
it required sentence-level understanding a subword tokenizer doesn't
have. Tokenization itself is just subword splitting + special tokens +
truncation, all handled by `AutoTokenizer`. No classic NLP preprocessing
(lowercasing, stopword removal, stemming) is applied — that would remove
signal these models were pretrained to expect and use.

**Usage:**
```
python train_model.py --model bert
python train_model.py --model biobert
python train_model.py --model pubmedbert
```
Writes `results/<model>/results.json` (hyperparameters, eval metrics,
confusion matrix, per-epoch training log), `results/<model>/confusion_matrix.png`,
and `results/<model>/eval_predictions.csv` (every `eval.csv` row plus the
model's `predicted_label`, `positive_probability`, and a `correct`
boolean — filter to `correct == False` for error analysis: false
positives are `label == 0` there, false negatives are `label == 1`).
Model weights are **not** saved by default (pass `--save-model-dir` to
save them somewhere outside the repo) — a fine-tuned checkpoint is large
(100s of MB) and doesn't belong in git; only the small metrics/logs do.

**Running it**: this needs real GPU-friendly compute, so the intended
flow is Colab, not local: clone the repo there, `pip install -r
requirements.txt`, then run the commands above (prefixed with `!` in a
notebook cell, since a bare `python ...` line is interpreted as Python
code by the cell, not a shell command). Download the resulting
`results/<model>/` folder back and commit it from a local clone — Colab
has no saved GitHub credentials, so this avoids setting up token auth
inside a notebook.

**Verification status**: executed end-to-end for real, twice, independently.
First on CPU (no GPU available in that session — each model took roughly
1.5-3 hours; this run also confirmed Hugging Face model-weight downloads,
which had previously failed with a 403 in an earlier, more restrictive
session). Then for real on a Colab T4 GPU by following the "Running it"
steps above (each model ~15-20 min). Both runs used the script's default
hyperparameters (4 epochs, batch size 8, lr 2e-5, max_length 512, seed 42)
and produced the same model ranking with metrics within ~1 point of each
other — a solid independent replication. **The numbers below are from the
Colab GPU run**, kept as the canonical result since it matches the
intended "Running it" workflow above; see git history for the CPU run's
numbers if useful for comparison. No errors or silent failures in either
run.

## 10. Results

**Stale as of the section-2 annotation correction**: the numbers below
were trained/evaluated against the *previous* `eval.csv` (682 windows,
before the `10195938` reclassification). `eval.csv` now has 683 windows
with a slightly different positive/negative split, so these results,
`results/<model>/results.json`, and `results/<model>/eval_predictions.csv`
all need one more `train_model.py` run per model to reflect the
corrected data. Kept here until that re-run happens.

| model | checkpoint | eval accuracy | precision | recall | F1 |
|---|---|---|---|---|---|
| bert | `bert-base-cased` | 0.889 | 0.895 | 0.854 | 0.874 |
| biobert | `dmis-lab/biobert-base-cased-v1.2` | **0.914** | 0.896 | **0.916** | **0.906** |
| pubmedbert | `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext` | 0.911 | **0.916** | 0.884 | 0.900 |

Confusion matrices (rows = actual [neg, pos], cols = predicted [neg, pos],
682 eval windows: 309 positive / 373 negative):
- bert: `[[342, 31], [45, 264]]`
- biobert: `[[340, 33], [26, 283]]`
- pubmedbert: `[[348, 25], [36, 273]]`

**biobert wins** on accuracy, recall, and F1, with pubmedbert a close
second (best precision of the three) — continued-pretraining BERT on
PubMed text helps more here than training a biomedical vocabulary from
scratch (pubmedbert), and both biomedical models clearly beat the generic
`bert-base-cased` baseline, confirming domain adaptation helps as expected
rather than just being assumed.

All three models show eval loss climbing every epoch (1 through 4) even as
accuracy/F1 keep improving through epoch 4 — the classic early signature
of overfitting (growing confidence, including on mistakes) that hasn't yet
cost accuracy on this run, but is worth watching. Worth factoring into any
hyperparameter tuning (e.g. early stopping on eval loss, or comparing
epoch-3 vs epoch-4 checkpoints) before assuming more epochs would keep
helping.

Full per-epoch metrics and hyperparameters are in
`results/<model>/results.json`; confusion matrix plots are in
`results/<model>/confusion_matrix.png`.

## Next steps

- **Error analysis**: read through false positives/negatives on
  `eval.csv` (particularly biobert's, as the best-performing model) to
  see whether mistakes cluster around the three easy-to-mistake-for-a-gap
  categories the annotation guidelines call out (practical
  recommendations, bare speculation, "unknown" as a classification
  label), or elsewhere.
- **Overfitting**: given the climbing eval loss noted above, try early
  stopping (on eval loss) or compare earlier-epoch checkpoints against the
  current epoch-4 numbers.
- **Hyperparameter tuning**: learning rate / batch size sweep for
  biobert specifically, since it's the strongest candidate to take
  forward.
- Write up the 3-model comparison and this analysis for the thesis.
