import math
import numpy as np
import pandas as pd
import string
import time
import torch

from nltk.tokenize.regexp import WordPunctTokenizer
from torch.utils.data import Dataset

tokenizer = WordPunctTokenizer()


def load_embeddings(path, embedding_dim):
    with open(path) as file:
        lines = file.readlines()

        index = []
        embeddings = np.zeros((len(lines), embedding_dim))

        for i, l in enumerate(lines):
            tokens = l.split(' ')
            index.append(tokens[0])
            embeddings[i, :] = tokens[1:]

        return pd.DataFrame(embeddings, index=index)


def cleanup_and_tokenize_text(text):
    cleaned = ''.join([c for c in text if c not in string.punctuation]).lower()
    return tokenizer.tokenize(cleaned)


def tokenize_rows(df):
    tokenized_headlines = df['headline'].apply(
        cleanup_and_tokenize_text).tolist()
    tokenized_desc = df['short_description'].apply(
        cleanup_and_tokenize_text).tolist()

    return [tokens1 + tokens2 for tokens1, tokens2 in zip(tokenized_headlines, tokenized_desc)]


def downsample(df, c):
    category_counts = df['category'].value_counts()
    min_count = category_counts.min()

    # Calculate the probability of keeping a row
    # of a given category.
    category_probs = (min_count / category_counts) ** (1/c)

    # This is a series used to determine the probability that each
    # row is kept. Each rows mask depends on its category.
    prob_mask = np.zeros(len(df))

    for i, category in enumerate(category_counts.index.tolist()):
        category_prob = category_probs[i]
        category_keep_mask = (df['category'] == category) * category_prob
        prob_mask = prob_mask + category_keep_mask

    keep_mask = np.random.rand(len(df)) <= prob_mask

    return df[keep_mask].reset_index(drop=True)


def create_unigram_counts(rows):
    # Flatten
    tokens = [t for tokens in rows for t in tokens]

    counts = {}

    for token in tokens:
        if token not in counts:
            counts[token] = 0
        counts[token] += 1

    return counts


class WordTokenDatasetSample():
    def __init__(self, sequence, offset, label, vocab_size):
        self.sequence = sequence
        self.offset = offset
        self.label = label
        self.vocab_size = vocab_size

    def __len__(self):
        return len(self.label)


class WordTokenDataset(Dataset):
    __TOKEN_UNK__ = '__TOKEN_UNK__'

    __TOKEN_LOW_FREQ__ = '__TOKEN_LOW_FREQ__'

    def __init__(self, data, downsample_c=None, accepted_tokens=None, include_special_tokens=False, min_word_freq=0):
        """
        data: A pandas data frame where each row is a news article.

        downsample_c: An optional downsampling constant. This downsampling constant configures
                      the downsampling of imbalanced classifications.

        accepted_tokens: Can optionally specify a set of tokens that are recognized by the
                         dataset. If specified, any tokens not in this set will be assigned
                         to the special unknown token.

        include_special_tokens: Whether to include special tokens in the final data samples.
                                This includes the unknown special token and the low freq token.

        min_word_freq: The minimum frequency allows for any word in the data corpus. Any word
                       with a frequency less than the min frequency will get mapped to the
                       low frequency token.
        """

        super().__init__()

        self.data = data
        self.downsample_c = downsample_c
        self.accepted_tokens = accepted_tokens
        self.include_special_tokens = include_special_tokens
        self.min_word_freq = min_word_freq

        self._is_prepared = False
        self._unigram_counts = None
        self._token_encoder = None
        self._encoded_to_idx = None
        self._label_encoder = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if type(idx) == int:
            idx = slice(idx, idx+1)

        assert(self._is_prepared)

        sub_data = self.data.iloc[idx]

        if len(sub_data) == 0:
            return WordTokenDatasetSample(sequence=torch.LongTensor([]),
                                          offset=torch.LongTensor([]),
                                          label=torch.LongTensor([]),
                                          vocab_size=len(self._encoded_to_idx))

        tokenized_rows = tokenize_rows(sub_data)

        offset = []
        sequence = []

        for i, tokens in enumerate(tokenized_rows):
            sub_sequence = [self._encoded_to_idx[self._token_encoder[t]]
                            for t in tokens if t in self._encoded_to_idx]
            sequence.extend(sub_sequence)
            offset.append(len(sequence) - len(sub_sequence))

        label = [self._label_encoder[l] for l in sub_data['category']]

        return WordTokenDatasetSample(sequence=torch.LongTensor(sequence),
                                      offset=torch.LongTensor(offset),
                                      label=torch.LongTensor(label),
                                      vocab_size=len(self._encoded_to_idx))

    def prepare(self):
        if self.downsample_c is not None:
            self.data = downsample(self.data, self.downsample_c)

        tokenized_rows = tokenize_rows(self.data)

        self._unigram_counts = create_unigram_counts(tokenized_rows)

        if not self.include_special_tokens:
            # Filter out any tokens that do not get encoded into themselves.
            tokenized_rows = [[t for t in tokens if self._encoded_token(
                t) == t] for tokens in tokenized_rows]
            # Need to re-generate the unigram counts after
            # performing this filtering.
            self._unigram_counts = create_unigram_counts(tokenized_rows)

        self._token_encoder = {t: self._encoded_token(
            t) for t in self._unigram_counts.keys()}
        self._encoded_to_idx = {t: i for i,
                                t in enumerate(self._token_encoder.values())}
        self._label_encoder = {l: i for i, l in enumerate(
            self.data['category'].unique())}

        # Remove any rows in data that have no tokens.
        keep_mask = np.zeros(len(tokenized_rows))
        for i, ts in enumerate(tokenized_rows):
            # This will be true if there exists a token that is encoded into itself.
            # (i.e. not an unknown token or low freq token).
            keep_mask[i] = len(
                [True for t in ts if self._encoded_token(t) == t]) > 0

        keep_mask = keep_mask.astype(bool)
        self.data = self.data.iloc[keep_mask]
        self._is_prepared = True

    def _encoded_token(self, token):
        assert(self._unigram_counts is not None)

        if self.accepted_tokens is not None and token not in self.accepted_tokens:
            return self.__TOKEN_UNK__
        elif token not in self._unigram_counts:
            return self.__TOKEN_UNK__
        elif self._unigram_counts[token] < self.min_word_freq:
            return self.__TOKEN_LOW_FREQ__
        return token


def collate_samples(samples):
    if len(samples) == 0:
        return WordTokenDatasetSample(sequence=torch.LongTensor([]),
                                      offset=torch.LongTensor([]),
                                      label=torch.LongTensor([]),
                                      vocab_size=0)

    label = torch.cat([s.label for s in samples])
    sequence = torch.cat([s.sequence for s in samples])
    vocab_size = samples[0].vocab_size

    offset = torch.zeros_like(label, dtype=torch.int64)
    iter = 0
    shift_val = 0

    for i, sample in enumerate(samples):
        print(iter)
        sample_offset = sample.offset
        offset[iter:(iter+len(sample_offset))] = (sample_offset + shift_val)

        iter = iter + len(sample_offset)
        shift_val = shift_val + len(samples[i].sequence)

    return WordTokenDatasetSample(sequence=sequence,
                                  offset=offset,
                                  label=label,
                                  vocab_size=vocab_size)
