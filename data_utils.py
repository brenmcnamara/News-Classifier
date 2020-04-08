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


class BOWEncoding():
    LOW_FREQ_TOKEN = '__LOW_FREQ_TOKEN__'

    def __init__(self, data, min_word_freq=0):
        self.data = data
        self.min_word_freq = min_word_freq

        self._is_prepared = False
        self.vocab_size = None
        self._label_encoder = None
        self._token_encoder = None
        self._token_decoder = None

    def prepare(self):
        tokenized_rows = tokenize_rows(self.data)
        unigram_counts = create_unigram_counts(tokenized_rows)

        if self.min_word_freq > 0:
            tokenized_rows = [[t if unigram_counts[t] >=
                               self.min_word_freq else self.LOW_FREQ_TOKEN for t in tokens] for tokens in tokenized_rows]
            unigram_counts = create_unigram_counts(tokenized_rows)

        self._unigram_counts = unigram_counts
        self._label_encoder = {l: i for i, l in enumerate(
            self.data['category'].unique())}
        self._token_encoder = {t: i for i,
                               t in enumerate(unigram_counts.keys())}
        self._token_decoder = {i: t for t, i in self._token_encoder.items()}

        self.vocab_size = len(unigram_counts)

        self._is_prepared = True

    def encode_token(self, token):
        assert(self._is_prepared)
        return self._token_encoder[token]

    def encode_label(self, label):
        assert(self._is_prepared)
        return self._label_encoder[label]

    def n_classes(self):
        assert(self._is_prepared)
        return len(self._label_encoder)

    def is_valid_token(self, token):
        assert(self._is_prepared)
        return token in self._token_encoder


class WordEmbeddingEncoding():
    def __init__(self, data, embeddings, min_word_freq=0):
        self.data = data
        self.embeddings = embeddings
        self.min_word_freq = min_word_freq
        self.vocab_size = len(embeddings)

        self._is_prepared = False
        self._embedding_tokens = None
        self._unigram_counts = None
        self._token_encoder = None
        self._label_encoder = None

    def prepare(self):
        self._embedding_tokens = {t for t in self.embeddings.index}

        tokenized_rows = tokenize_rows(self.data)
        self._unigram_counts = create_unigram_counts(tokenized_rows)

        # Filter out any tokens that are invalid.
        tokenized_rows = [
            [t for t in tokens if self.is_valid_token(t)] for tokens in tokenized_rows]

        # Need to re-generate the unigram counts after
        # performing this filtering.
        self._unigram_counts = create_unigram_counts(tokenized_rows)

        self._token_encoder = {t: i for i,
                               t in enumerate(self.embeddings.index)}
        self._label_encoder = {l: i for i, l in enumerate(
            self.data['category'].unique())}

        self._is_prepared = True

    def encode_token(self, token):
        assert(self._is_prepared)
        return self._token_encoder[token]

    def encode_label(self, label):
        assert(self._is_prepared)
        return self._label_encoder[label]

    def n_classes(self):
        assert(self._is_prepared)
        return len(self._label_encoder)

    def is_valid_token(self, token):
        assert(self._embedding_tokens is not None)
        assert(self._unigram_counts is not None)

        if token not in self._embedding_tokens:
            return False
        elif token not in self._unigram_counts or self._unigram_counts[token] < self.min_word_freq:
            return False
        return True


class WordTokenDatasetSample():
    def __init__(self, sequence, offset, label, vocab_size, doc_count, doc_freq):
        self.sequence = sequence
        self.offset = offset
        self.label = label
        self.vocab_size = vocab_size
        self.doc_count = doc_count
        self.doc_freq = doc_freq

    def __len__(self):
        return len(self.label)

    def create_bow_matrix(self):
        bow = torch.zeros(
            size=(len(self.offset), self.vocab_size), dtype=torch.int64)
        offset_with_end = torch.cat(
            [self.offset, torch.LongTensor([len(self.sequence)])])

        for i in range(len(offset_with_end) - 1):
            start = offset_with_end[i].item()
            end = offset_with_end[i+1]
            sub_seq = self.sequence[start:end]
            for idx in sub_seq:
                bow[i, idx.item()] += 1

        return bow

    def create_uniform_weights(self):
        if len(self) == 0:
            return torch.FloatTensor([])

        weights = torch.zeros_like(self.sequence, dtype=torch.float)

        offset_with_end = torch.cat(
            [self.offset, torch.LongTensor([len(self.sequence)])])

        for i in range(len(offset_with_end) - 1):
            start = offset_with_end[i].item()
            end = offset_with_end[i+1].item()
            weight = 1. / (end - start)

            weights[start:end] = weight

        return weights

    def create_tf_idf_weights(self):
        if len(self) == 0:
            return torch.FloatTensor([])

        weights = torch.zeros_like(self.sequence, dtype=torch.float)

        offset_with_end = torch.cat(
            [self.offset, torch.LongTensor([len(self.sequence)])])
        idf = torch.log(torch.FloatTensor(
            [float(self.doc_count) / self.doc_freq[idx.item()] for idx in self.sequence]))

        for i in range(len(offset_with_end) - 1):
            start = offset_with_end[i].item()
            end = offset_with_end[i+1].item()
            doc_len = end - start

            # Generate term frequencies for each element in sequence.
            freq_map = {}
            for idx in self.sequence[start:end]:
                if idx not in freq_map:
                    freq_map[idx.item()] = 1
                else:
                    freq_map[idx.item()] += 1

            tf = torch.FloatTensor(
                [float(freq_map[idx.item()]) / doc_len for idx in self.sequence[start:end]])
            tf_idf = tf * idf[start:end]

            # normalize tf_idf weights between documents to account for
            # documents of very different sizes.
            total = torch.sum(tf_idf)

            weights[start:end] = tf_idf / total

        return weights


class WordTokenDataset(Dataset):
    def __init__(self, data, encoding):
        """
        data: A pandas data frame where each row is a news article.

        encoding: The word token encoding that contains information
                  about the corpus and the encoding of words
                  and labels.
        """

        super().__init__()
        self.data = data
        self.encoding = encoding

        self._is_prepared = False
        self._doc_freq = None

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
                                          vocab_size=self.encoding.vocab_size,
                                          doc_count=len(self.data),
                                          doc_freq=self._doc_freq)

        tokenized_rows = tokenize_rows(sub_data)

        offset = []
        sequence = []

        for i, tokens in enumerate(tokenized_rows):
            sub_sequence = [self.encoding.encode_token(
                t) for t in tokens if self.encoding.is_valid_token(t)]
            sequence.extend(sub_sequence)
            offset.append(len(sequence) - len(sub_sequence))

        label = [self.encoding.encode_label(l) for l in sub_data['category']]

        return WordTokenDatasetSample(sequence=torch.LongTensor(sequence),
                                      offset=torch.LongTensor(offset),
                                      label=torch.LongTensor(label),
                                      vocab_size=self.encoding.vocab_size,
                                      doc_count=len(self.data),
                                      doc_freq=self._doc_freq)

    def prepare(self):
        tokenized_rows = tokenize_rows(self.data)

        doc_freq = {}
        for i, tokens in enumerate(tokenized_rows):
            new_tokens = [t for t in tokens if self.encoding.is_valid_token(t)]
            tokenized_rows[i] = new_tokens

            seen = set()
            for token in new_tokens:
                if token in seen:
                    continue

                idx = self.encoding.encode_token(token)
                if idx not in doc_freq:
                    doc_freq[idx] = 1
                else:
                    doc_freq[idx] += 1

                seen.add(token)

        # Remove any token rows that are empty.
        keep_mask = np.array([len(ts) > 0 for ts in tokenized_rows])

        self.data = self.data.iloc[keep_mask]
        self._doc_freq = doc_freq
        self._is_prepared = True


def collate_samples(samples):
    if len(samples) == 0:
        return WordTokenDatasetSample(sequence=torch.LongTensor([]),
                                      offset=torch.LongTensor([]),
                                      label=torch.LongTensor([]),
                                      vocab_size=0,
                                      doc_freq={},
                                      doc_count=0)

    label = torch.cat([s.label for s in samples])
    sequence = torch.cat([s.sequence for s in samples])
    vocab_size = samples[0].vocab_size
    doc_freq = samples[0].doc_freq
    doc_count = samples[0].doc_count

    offset = torch.zeros_like(label, dtype=torch.int64)
    iter = 0
    shift_val = 0

    for i, sample in enumerate(samples):
        sample_offset = sample.offset
        offset[iter:(iter+len(sample_offset))] = (sample_offset + shift_val)

        iter = iter + len(sample_offset)
        shift_val = shift_val + len(samples[i].sequence)

    return WordTokenDatasetSample(sequence=sequence,
                                  offset=offset,
                                  label=label,
                                  vocab_size=vocab_size,
                                  doc_freq=doc_freq,
                                  doc_count=doc_count)
