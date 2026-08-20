"""Turn per-article sentence segmentation + anchor positions into the
augmented positive/negative windows used to train the classifier.

Window size rule: for anchor-centered windows (both positive anchors and
standalone negative anchors), the window is the full anchor plus 2 extra
sentences of context, distributed 3 ways: 2-before, 1-before+1-after,
2-after -- clipped (not skipped) at section boundaries.

Nearby-negative windows (the blocks flanking a positive anchor) are a
different, fixed-size mechanism: exactly 3 sentences immediately before
and exactly 3 immediately after the anchor, independent of the anchor's
own length. A side is only skipped if zero sentences are available there.

After generation, every negative-labeled window (from either mechanism)
is checked against every positive anchor in the same section: full
containment promotes it to positive, partial overlap discards it
(ambiguous windows are dropped rather than guessed at).
"""

POSITIVE_VARIANTS = [
    (2, 0, "2before_0after"),
    (1, 1, "1before_1after"),
    (0, 2, "0before_2after"),
]

NEARBY_NEGATIVE_BLOCK_SIZE = 3


def _clip_variant_windows(n_sentences, anchor_range):
    """Anchor + 2 extra sentences, 3 position variants, clipped at bounds."""
    first, last = anchor_range
    windows = []
    for n_before, n_after, variant_name in POSITIVE_VARIANTS:
        start = max(0, first - n_before)
        end = min(n_sentences - 1, last + n_after)
        windows.append((start, end, variant_name))
    return windows


def _nearby_negative_blocks(n_sentences, positive_anchor_range):
    """Fixed 3-sentence blocks flanking a positive anchor, skipped on the
    side where zero sentences are available."""
    first, last = positive_anchor_range
    blocks = []

    before_end = first - 1
    if before_end >= 0:
        before_start = max(0, before_end - NEARBY_NEGATIVE_BLOCK_SIZE + 1)
        blocks.append((before_start, before_end, "before_block"))

    after_start = last + 1
    if after_start <= n_sentences - 1:
        after_end = min(n_sentences - 1, after_start + NEARBY_NEGATIVE_BLOCK_SIZE - 1)
        blocks.append((after_start, after_end, "after_block"))

    return blocks


def _fully_contains(outer, inner):
    return outer[0] <= inner[0] and inner[1] <= outer[1]


def _overlaps(range_a, range_b):
    return range_a[0] <= range_b[1] and range_b[0] <= range_a[1]


def generate_section_windows(n_sentences, positive_anchors, negative_anchors):
    """positive_anchors / negative_anchors: list of dicts with at least
    'range': (first_idx, last_idx) and 'anchor_text'.

    Returns (windows, stats):
      windows: list of candidate window dicts
        {range: (start, end), label: 0/1, source: str, variant: str,
         anchor_text: str, promoted: bool}
        Partial-overlap candidates are already discarded here.
      stats: dict with candidates_generated / discarded_partial_overlap /
        promoted_positive_wins, for reporting.

    Text-level dedup is NOT done here -- window text depends on the
    original section text, which the caller has but this function
    doesn't, so `dedup_windows` is applied by the caller once text is
    attached.
    """
    candidates = []

    for anchor in positive_anchors:
        for start, end, variant in _clip_variant_windows(n_sentences, anchor["range"]):
            candidates.append({
                "range": (start, end),
                "label": 1,
                "source": "positive_anchor",
                "variant": variant,
                "anchor_text": anchor["anchor_text"],
                "promoted": False,
            })
        for start, end, variant in _nearby_negative_blocks(n_sentences, anchor["range"]):
            candidates.append({
                "range": (start, end),
                "label": 0,
                "source": "nearby_negative",
                "variant": variant,
                "anchor_text": anchor["anchor_text"],
                "promoted": False,
            })

    for anchor in negative_anchors:
        for start, end, variant in _clip_variant_windows(n_sentences, anchor["range"]):
            candidates.append({
                "range": (start, end),
                "label": 0,
                "source": "negative_anchor",
                "variant": variant,
                "anchor_text": anchor["anchor_text"],
                "promoted": False,
            })

    positive_ranges = [a["range"] for a in positive_anchors]

    resolved = []
    n_discarded = 0
    n_promoted = 0
    for cand in candidates:
        if cand["label"] == 1:
            resolved.append(cand)
            continue
        fully_contains_any = False
        partial_overlap_any = False
        for pr in positive_ranges:
            if not _overlaps(cand["range"], pr):
                continue
            if _fully_contains(cand["range"], pr):
                fully_contains_any = True
            else:
                partial_overlap_any = True
        if fully_contains_any:
            # Full containment wins even if the window also brushes a
            # different anchor only partially -- unambiguously positive.
            cand["label"] = 1
            cand["promoted"] = True
            n_promoted += 1
            resolved.append(cand)
        elif partial_overlap_any:
            n_discarded += 1
            continue  # ambiguous, discard
        else:
            resolved.append(cand)

    stats = {
        "candidates_generated": len(candidates),
        "discarded_partial_overlap": n_discarded,
        "promoted_positive_wins": n_promoted,
    }
    return resolved, stats


def dedup_windows(windows_with_text):
    """windows_with_text: list of dicts including a 'text' key (final
    window string). Collapses exact-text duplicates; if a text appears
    with both labels, the positive copy wins.

    Returns the deduplicated list, preserving first-seen order.
    """
    best_by_text = {}
    order = []
    for w in windows_with_text:
        key = w["text"]
        if key not in best_by_text:
            best_by_text[key] = w
            order.append(key)
        else:
            existing = best_by_text[key]
            if w["label"] == 1 and existing["label"] == 0:
                best_by_text[key] = w
    return [best_by_text[key] for key in order]
