"""Split articles into train/eval sets before augmentation, so windows
from the same paper never appear on both sides."""
import random

DEFAULT_SEED = 42
DEFAULT_TRAIN_FRACTION = 0.8


def split_article_ids(article_ids, train_fraction=DEFAULT_TRAIN_FRACTION, seed=DEFAULT_SEED):
    ids = sorted(article_ids)
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_train = round(len(ids) * train_fraction)
    train_ids = set(ids[:n_train])
    eval_ids = set(ids[n_train:])
    return train_ids, eval_ids
