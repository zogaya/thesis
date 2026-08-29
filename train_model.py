#!/usr/bin/env python3
"""Fine-tune a binary epistemic-gap classifier on train.csv/eval.csv.

One shared script for all three models being compared, so every model is
trained and evaluated under identical conditions (same hyperparameters,
same data, same metric computation) -- only the checkpoint name differs.

Usage:
    python train_model.py --model bert
    python train_model.py --model biobert
    python train_model.py --model pubmedbert

Writes results/<model>/results.json (hyperparameters, eval metrics,
confusion matrix, per-epoch training log) and
results/<model>/confusion_matrix.png. Does NOT save model weights by
default -- pass --save-model-dir to save the fine-tuned checkpoint
somewhere outside the repo (e.g. a mounted Google Drive path in Colab).
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

MODEL_CHECKPOINTS = {
    "bert": "bert-base-cased",
    "biobert": "dmis-lab/biobert-base-cased-v1.2",
    "pubmedbert": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
}

DEFAULT_MAX_LENGTH = 512


class WindowDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", pos_label=1, zero_division=0
    )
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def plot_confusion_matrix(cm, path, model_name):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    labels = ["negative (0)", "positive (1)"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix -- {model_name}")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i][j]), ha="center", va="center",
                     color="white" if cm[i][j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=sorted(MODEL_CHECKPOINTS))
    parser.add_argument("--train-csv", default="data/processed/train.csv")
    parser.add_argument("--eval-csv", default="data/processed/eval.csv")
    parser.add_argument("--outdir", default=None,
                         help="Defaults to results/<model>")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-model-dir", default=None,
                         help="If set, save the fine-tuned model+tokenizer here "
                              "(outside the repo -- these files are large).")
    args = parser.parse_args()

    checkpoint = MODEL_CHECKPOINTS[args.model]
    outdir = args.outdir or os.path.join("results", args.model)
    os.makedirs(outdir, exist_ok=True)

    train_df = pd.read_csv(args.train_csv)
    eval_df = pd.read_csv(args.eval_csv)

    print(f"Model: {args.model} ({checkpoint})")
    print(f"Train: {len(train_df)} rows, Eval: {len(eval_df)} rows")

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    def tokenize(texts):
        return tokenizer(
            list(texts),
            truncation=True,
            padding=True,
            max_length=args.max_length,
            return_tensors="pt",
        )

    train_dataset = WindowDataset(tokenize(train_df["text"]), train_df["label"].tolist())
    eval_dataset = WindowDataset(tokenize(eval_df["text"]), eval_df["label"].tolist())

    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

    training_args = TrainingArguments(
        output_dir=os.path.join(outdir, "_checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        eval_strategy="epoch",
        save_strategy="no",
        logging_strategy="epoch",
        seed=args.seed,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    eval_metrics = trainer.evaluate()

    predictions = trainer.predict(eval_dataset)
    logits = predictions.predictions
    preds = np.argmax(logits, axis=-1)
    probs = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)
    labels = eval_df["label"].tolist()
    cm = confusion_matrix(labels, preds, labels=[0, 1])

    eval_predictions_df = eval_df.copy()
    eval_predictions_df["predicted_label"] = preds
    eval_predictions_df["positive_probability"] = probs[:, 1]
    eval_predictions_df["correct"] = eval_predictions_df["label"] == eval_predictions_df["predicted_label"]
    eval_predictions_df.to_csv(os.path.join(outdir, "eval_predictions.csv"), index=False)

    results = {
        "model": args.model,
        "checkpoint": checkpoint,
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "max_length": args.max_length,
            "seed": args.seed,
        },
        "eval_metrics": {k: v for k, v in eval_metrics.items() if k.startswith("eval_")},
        "confusion_matrix": {
            "row_order": ["actual_negative", "actual_positive"],
            "col_order": ["predicted_negative", "predicted_positive"],
            "matrix": cm.tolist(),
        },
        "training_log_history": trainer.state.log_history,
    }

    results_path = os.path.join(outdir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    plot_confusion_matrix(cm, os.path.join(outdir, "confusion_matrix.png"), args.model)

    print(json.dumps(results["eval_metrics"], indent=2))
    print(f"Wrote {results_path}")
    print(f"Wrote {os.path.join(outdir, 'eval_predictions.csv')}")

    if args.save_model_dir:
        os.makedirs(args.save_model_dir, exist_ok=True)
        trainer.save_model(args.save_model_dir)
        tokenizer.save_pretrained(args.save_model_dir)
        print(f"Saved model + tokenizer to {args.save_model_dir}")


if __name__ == "__main__":
    main()
