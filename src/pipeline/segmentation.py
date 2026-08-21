"""Sentence segmentation tuned for extracted biomedical article sections.

The raw section text has three recurring artifacts that a plain sentence
segmenter mishandles:

1. Missing sentence boundaries where the extraction dropped a space after a
   period, e.g. "...enhanced NaV1.1 activity.Here, we aim...".
2. Structural headings glued directly onto the surrounding text in
   structured abstracts, e.g. "AbstractObjectiveMonogenic epilepsies...".
3. Biomedical notation that looks like a missing boundary but isn't, e.g.
   the HGVS variant prefixes "p.Asp920Glu", "c.400G>A".

Strategy: compute sentence-boundary cut points ourselves (heading spans,
repaired missing boundaries) and union them with pysbd's own boundary
detection, all as character offsets into the *original, untouched* text.
Nothing is ever inserted or deleted, so offsets never need remapping.
"""
import re

import pysbd

HEADING_WORDS = [
    "Background", "Objective", "Objectives", "Purpose", "Aim", "Aims",
    "Abstract", "Materials and Methods", "Methods", "Method", "Results",
    "Discussion", "Discussions", "Conclusion", "Conclusions",
    "Limitations", "Introduction", "Rationale", "Interpretation",
    "Summary", "Highlights",
    # CARE case-report structured-abstract labels
    "Patient concern", "Patient concerns", "Diagnoses", "Diagnosis",
    "Intervention", "Interventions", "Outcomes", "Outcome", "Lessons",
]

ADMIN_WORDS = [
    "Ethical approval", "Ethics approval", "Ethics statement",
    "Informed consent", "Conflict of interest", "Conflicts of interest",
    "Competing interests", "Declaration of competing interest",
    "Funding", "Author contributions", "Authors contributions",
    "Data availability", "Acknowledgments", "Acknowledgements",
    "Supplementary Information", "Supplementary information",
    "Trial registration",
]

_ALL_HEADINGS = sorted(HEADING_WORDS + ADMIN_WORDS, key=len, reverse=True)
_HEADING_SET_LOWER = {h.lower() for h in _ALL_HEADINGS}
_HEADING_SET_LOWER |= {h.lower() + ":" for h in _ALL_HEADINGS}

# Matches a heading token glued to what follows -- either directly by an
# uppercase letter ("MethodsA total...") or via a colon with no space
# ("Rationale:CUL3...", the CARE case-report style) -- and not preceded
# by a letter (so it starts a "word"). The colon, if present, is folded
# into the match so it's stripped along with the heading.
_GLUED_HEADING_RE = re.compile(
    r"(?<![A-Za-z])(" + "|".join(re.escape(h) for h in _ALL_HEADINGS) + r")(:)?(?=[A-Z])"
)

# HGVS reference-sequence prefixes: c. (coding DNA), g. (genomic),
# m. (mitochondrial), n. (non-coding RNA), p. (protein), r. (RNA).
_HGVS_RE = re.compile(r"\b[cgmnpr]\.(?=[A-Z])")

# Common abbreviations that can appear glued to a following capitalized
# word without triggering a real sentence boundary.
_ABBREV_RE = re.compile(
    r"\b(al|Fig|vs|approx|Dr|Mr|Mrs|Prof|Ref|Suppl|etc|cf)\.(?=[A-Z])",
    re.IGNORECASE,
)

# The general "missing boundary" signature: a word character (letter or
# digit, so gene symbols like "NPRL3." and figure/table numbers like
# "Table 1." are covered too) followed by a period and immediately an
# uppercase letter, no space.
_MISSING_BOUNDARY_RE = re.compile(r"[A-Za-z0-9]\.(?=[A-Z])")

# Stray citation-superscript reference numbers left behind by extraction,
# e.g. "...enhanced NaV1.1 activity.20\n, \n21\n It is hypothesized..."
# (originally inline superscript "activity.20,21" citing references 20 and
# 21). Two safe-to-strip shapes, both anchored right after sentence-ending
# punctuation and right before the next sentence's capital letter:
#   (a) TWO OR MORE short comma/newline-separated digit groups -- a shape
#       real prose doesn't produce, so it's unambiguous regardless of
#       surrounding whitespace.
#   (b) a SINGLE digit group, but only when bookended by a literal newline
#       on both sides (".\n19\n Similar..."). A lone number with no
#       newline bookend (".9 However...") is deliberately left alone --
#       real gene/variant identifiers routinely end in one digit followed
#       directly by a comma (e.g. "PCDH19,", "Cys182,"), so without the
#       double-newline signal there's no safe way to tell those apart from
#       a citation marker. With it, checked against the full corpus: 0
#       collisions with gene/variant-identifier text.
_CITATION_MARKER_RE = re.compile(
    r"(?<=[.!?])(?:"
    r"\s*\d{1,3}\s*(?:[,\n]\s*\d{1,3}\s*){1,5}"
    r"|"
    r"\n\d{1,3}\n\s*"
    r")(?=[A-Z])"
)

_segmenter = pysbd.Segmenter(language="en", clean=False, char_span=True)


def find_glued_heading_spans(text):
    """Return (start, end) spans of structural headings glued to adjacent text.

    Runs to a fixed point so chained headings with zero separator between
    them (e.g. "AbstractObjectiveMonogenic...") are all found: matching a
    heading can only reveal the next one once the first is masked out,
    since the lookbehind requires a non-letter immediately before it.
    """
    spans = set()
    scratch = text
    while True:
        found_new = False
        for m in _GLUED_HEADING_RE.finditer(scratch):
            span = (m.start(), m.end())
            if span not in spans:
                spans.add(span)
                found_new = True
        if not found_new:
            break
        chars = list(scratch)
        for s, e in spans:
            for i in range(s, e):
                chars[i] = " "
        scratch = "".join(chars)
    return sorted(spans)


def _protected_positions(text):
    """Character offsets of periods that must NOT be treated as sentence
    boundaries even though they match the lower.Upper glue pattern."""
    protected = set()
    for m in _HGVS_RE.finditer(text):
        protected.add(m.end() - 1)  # offset of the period itself
    for m in _ABBREV_RE.finditer(text):
        protected.add(m.end() - 1)
    return protected


def find_missing_boundary_cuts(text):
    """Character offsets where a sentence boundary must be inserted to
    repair a glued 'word.Word' pair, excluding protected biomedical
    notation and abbreviations."""
    protected = _protected_positions(text)
    cuts = set()
    for m in _MISSING_BOUNDARY_RE.finditer(text):
        period_pos = m.start() + 1
        if period_pos in protected:
            continue
        cuts.add(m.end())  # boundary right after the period
    return cuts


def find_citation_marker_spans(text):
    """Return (start, end) spans of stray multi-number citation markers
    (see _CITATION_MARKER_RE). These sit between two real sentences but
    neither pysbd nor find_missing_boundary_cuts creates a cut point next
    to them -- pysbd doesn't split before a digit (avoids breaking
    decimals/numbered lists), and the digit isn't uppercase so the missing-
    boundary repair doesn't fire either. Adding cut points at both ends is
    enough: once isolated, the marker fragment is pure digits/punctuation
    and segment_sentences' existing "no alphabetic content" filter drops it."""
    return [(m.start(), m.end()) for m in _CITATION_MARKER_RE.finditer(text)]


_WHITESPACE_RE = re.compile(r"\s")


def _normalize_whitespace_same_length(text):
    """Replace every whitespace character (including stray embedded
    newlines from citation-superscript extraction artifacts, e.g.
    "UNC13B,6\\nCELSR3,7\\n...") with a single regular space, one-for-one,
    so pysbd stops treating them as paragraph breaks. Length-preserving,
    so offsets stay valid against the original text."""
    return _WHITESPACE_RE.sub(" ", text)


def segment_sentences(text):
    """Return a list of (start, end) offsets into `text` for real sentences,
    with structural headings and admin labels excluded entirely."""
    if not text or not text.strip():
        return []

    heading_spans = find_glued_heading_spans(text)
    citation_marker_spans = find_citation_marker_spans(text)
    repair_cuts = find_missing_boundary_cuts(text)
    baseline_cuts = {
        span.end for span in _segmenter.segment(_normalize_whitespace_same_length(text))
    }

    cut_points = {0, len(text)}
    for start, end in heading_spans:
        cut_points.add(start)
        cut_points.add(end)
    for start, end in citation_marker_spans:
        cut_points.add(start)
        cut_points.add(end)
    cut_points |= repair_cuts
    cut_points |= baseline_cuts
    cut_points = sorted(cut_points)

    sentences = []
    for i in range(len(cut_points) - 1):
        start, end = cut_points[i], cut_points[i + 1]
        raw = text[start:end]
        trimmed = raw.strip()
        if not trimmed:
            continue
        # Skip fragments that are pure structural heading/admin labels.
        if trimmed.lower() in _HEADING_SET_LOWER:
            continue
        # Skip fragments with no alphabetic content (stray punctuation).
        if not re.search(r"[A-Za-z]", trimmed):
            continue
        # Recompute trimmed offsets relative to the original text.
        lead_ws = len(raw) - len(raw.lstrip())
        trail_ws = len(raw) - len(raw.rstrip())
        sentences.append((start + lead_ws, end - trail_ws))

    return sentences
