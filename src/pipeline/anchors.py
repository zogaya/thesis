"""Locate annotated knowledge-gap statements ("anchors") inside their
source section and map them onto the sentence-segmentation produced by
`segmentation.segment_sentences`.
"""


class AnchorNotFoundError(Exception):
    pass


class AnchorAmbiguousError(Exception):
    pass


def locate_anchor(section_text, statement):
    """Return (start, end) character offsets of `statement` inside
    `section_text`. Raises if it's missing or not unique."""
    first = section_text.find(statement)
    if first == -1:
        raise AnchorNotFoundError(statement[:80])
    second = section_text.find(statement, first + 1)
    if second != -1:
        raise AnchorAmbiguousError(statement[:80])
    return first, first + len(statement)


def snap_to_sentence_indices(sentence_spans, anchor_start, anchor_end):
    """Return (first_idx, last_idx) into `sentence_spans` for the smallest
    contiguous run of sentences that fully covers [anchor_start, anchor_end).

    The annotator's exact character boundaries don't always line up with
    our automatic sentence boundaries, so we expand outward to whichever
    sentences overlap the anchor at all, then take the full span.
    """
    overlapping = [
        i for i, (s, e) in enumerate(sentence_spans)
        if s < anchor_end and anchor_start < e
    ]
    if not overlapping:
        return None
    return min(overlapping), max(overlapping)
