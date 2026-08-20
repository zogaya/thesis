#!/usr/bin/env python3
"""Build the augmented-window training dataset from the manually annotated
knowledge-gap JSON file.

Usage:
    python build_dataset.py \
        --input data/raw/ongoing_ignorannotations_balanced.json \
        --outdir data/processed

Produces data/processed/{train,eval}.{jsonl,csv} and prints a summary of
counts and class balance.
"""
import argparse
import csv
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from pipeline.anchors import (
    AnchorAmbiguousError,
    AnchorNotFoundError,
    locate_anchor,
    snap_to_sentence_indices,
)
from pipeline.segmentation import segment_sentences
from pipeline.split import split_article_ids
from pipeline.windows import dedup_windows, generate_section_windows


def merge_sections(article):
    merged = {}
    for section_dict in article["sections"]:
        merged.update(section_dict)
    return merged


def build_windows_for_section(article_id, section_name, section_text, positives, negatives):
    """Returns a deduplicated list of window dicts for one (article, section)."""
    sentence_spans = segment_sentences(section_text)
    if not sentence_spans:
        return []

    def to_anchor_records(annotations):
        records = []
        for ann in annotations:
            stmt = ann["knowledge gap statement"]
            try:
                a_start, a_end = locate_anchor(section_text, stmt)
            except (AnchorNotFoundError, AnchorAmbiguousError) as exc:
                print(f"  [WARN] {article_id}/{section_name}: {type(exc).__name__}: {exc}",
                      file=sys.stderr)
                continue
            anchor_range = snap_to_sentence_indices(sentence_spans, a_start, a_end)
            if anchor_range is None:
                print(f"  [WARN] {article_id}/{section_name}: anchor has no overlapping sentence",
                      file=sys.stderr)
                continue
            records.append({"range": anchor_range, "anchor_text": stmt})
        return records

    positive_records = to_anchor_records(positives)
    negative_records = to_anchor_records(negatives)

    candidates, gen_stats = generate_section_windows(len(sentence_spans), positive_records, negative_records)

    windows = []
    for cand in candidates:
        start_idx, end_idx = cand["range"]
        text = " ".join(
            section_text[sentence_spans[i][0]:sentence_spans[i][1]]
            for i in range(start_idx, end_idx + 1)
        )
        windows.append({
            "article_id": article_id,
            "section": section_name,
            "text": text,
            "label": cand["label"],
            "source": cand["source"],
            "variant": cand["variant"],
            "promoted": cand["promoted"],
            "anchor_text": cand["anchor_text"],
            "n_sentences": end_idx - start_idx + 1,
        })

    deduped = dedup_windows(windows)
    gen_stats["deduped_away"] = len(windows) - len(deduped)
    return deduped, gen_stats


def build_dataset(data):
    all_windows = []
    stats = defaultdict(int)

    for article_id, article in data.items():
        sections = merge_sections(article)
        by_section = defaultdict(lambda: {"positives": [], "negatives": []})
        for ann in article.get("positives", []):
            by_section[ann["source_section"]]["positives"].append(ann)
        for ann in article.get("negatives", []):
            by_section[ann["source_section"]]["negatives"].append(ann)

        for section_name, anns in by_section.items():
            section_text = sections.get(section_name)
            if not section_text:
                stats["missing_section"] += 1
                continue
            windows, gen_stats = build_windows_for_section(
                article_id, section_name, section_text,
                anns["positives"], anns["negatives"],
            )
            all_windows.extend(windows)
            for k, v in gen_stats.items():
                stats[k] += v

    return all_windows, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/ongoing_ignorannotations_balanced.json")
    parser.add_argument("--outdir", default="data/processed")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        data = json.load(f)

    all_windows, stats = build_dataset(data)

    train_ids, eval_ids = split_article_ids(data.keys(), args.train_fraction, args.seed)

    train_windows = [w for w in all_windows if w["article_id"] in train_ids]
    eval_windows = [w for w in all_windows if w["article_id"] in eval_ids]

    os.makedirs(args.outdir, exist_ok=True)

    def write_jsonl(path, windows):
        with open(path, "w", encoding="utf-8") as f:
            for w in windows:
                f.write(json.dumps(w, ensure_ascii=False) + "\n")

    fieldnames = [
        "article_id", "section", "text", "label", "source", "variant",
        "promoted", "anchor_text", "n_sentences",
    ]

    def write_csv(path, windows):
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
            writer.writeheader()
            for w in windows:
                writer.writerow(w)

    write_jsonl(os.path.join(args.outdir, "train.jsonl"), train_windows)
    write_jsonl(os.path.join(args.outdir, "eval.jsonl"), eval_windows)
    write_csv(os.path.join(args.outdir, "train.csv"), train_windows)
    write_csv(os.path.join(args.outdir, "eval.csv"), eval_windows)

    def summarize(name, windows):
        n_pos = sum(1 for w in windows if w["label"] == 1)
        n_neg = len(windows) - n_pos
        n_promoted = sum(1 for w in windows if w["promoted"])
        print(f"{name}: {len(windows)} windows ({n_pos} positive, {n_neg} negative, "
              f"{n_promoted} promoted via positive-wins)")

    print(f"Articles: {len(data)} total, {len(train_ids)} train, {len(eval_ids)} eval")
    summarize("Train", train_windows)
    summarize("Eval", eval_windows)
    print(f"Candidate windows generated: {stats.get('candidates_generated', 0)}")
    print(f"  discarded (partial positive-anchor overlap): {stats.get('discarded_partial_overlap', 0)}")
    print(f"  promoted to positive (positive-wins rule): {stats.get('promoted_positive_wins', 0)}")
    print(f"  collapsed as duplicates: {stats.get('deduped_away', 0)}")
    if stats.get("missing_section"):
        print(f"  missing sections skipped: {stats['missing_section']}")


if __name__ == "__main__":
    main()
