import numpy as np
import pandas as pd
import string
import time

from nltk.tokenize.regexp import WordPunctTokenizer

# Special token for tokens that occur MIN_WORD_FREQ or fewer times in the
# entire corpus.
__LOW_FREQ_TOKEN__ = '__LOW_FREQ_TOKEN__'

MIN_WORD_FREQ = 5


def cleanup_and_tokenize_text(text):
    tokenizer = WordPunctTokenizer()
    cleaned = ''.join([c for c in text if c not in string.punctuation]).lower()
    return tokenizer.tokenize(cleaned)


def tokenize_rows(df):
    tokenized_headlines = df['headline'].apply(
        cleanup_and_tokenize_text).tolist()
    tokenized_desc = df['short_description'].apply(
        cleanup_and_tokenize_text).tolist()

    return [tokens1 + tokens2 for tokens1, tokens2 in zip(tokenized_headlines, tokenized_desc)]


def create_encoder_and_decoder(unigram_counts):
    encoder = {t: i for i, t in enumerate(unigram_counts.keys())}
    decoder = {i: t for t, i in encoder.items()}

    return encoder, decoder


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


def create_bow_dataframe(encoded_token_rows, encoder, decoder):
    bows = np.zeros((len(encoded_token_rows), len(encoder)))

    for i, encoded_tokens in enumerate(encoded_token_rows):
        for encoded in encoded_tokens:
            bows[i, encoded] += 1

    df = pd.DataFrame(data=bows)
    df.columns = [decoder[i] for i in range(len(decoder))]

    return df


def process_data(data_raw, min_word_freq, should_downsample, log=True):
    def print_if_logging(x): return print(x) if log else None

    print_if_logging(f'[1/7] Downsampling...')
    if should_downsample:
        data = downsample(data_raw, c=3)
    else:
        data = data_raw
        print_if_logging(f'     Skipping Downsampling')

    print_if_logging(f'[2/8] Generating labels...')
    labels = data['category']

    print_if_logging(f'[3/8] Tokenizing rows...')
    token_rows = tokenize_rows(data)

    print_if_logging('[4/8] Generating global unigram count ...')
    unigram_counts = create_unigram_counts(token_rows)
    print_if_logging(
        '[5/8] Filtering out low-frequency words (only small dataset) ...')
    if min_word_freq is not None:
        token_rows = [[token if unigram_counts[token] >
                       min_word_freq else __LOW_FREQ_TOKEN__ for token in tokens] for tokens in token_rows]
        unigram_counts = create_unigram_counts(token_rows)
    else:
        print_if_logging(f'     Skipping low-frequency filtering')

    print_if_logging('[6/8] Create encoder / decoder ...')
    encoder, decoder = create_encoder_and_decoder(unigram_counts)

    print_if_logging('[7/8] Encoding Token Rows ...')
    encoded_token_rows = [[encoder[t] for t in tokens]
                          for tokens in token_rows]

    print_if_logging('[8/8] Creating Bag Of Words DataFrame ...')
    data_bow = create_bow_dataframe(encoded_token_rows, encoder, decoder)

    return data_bow, labels, encoder, decoder


def main():
    # The dataset provided is malformed JSON. Need to fix up the JSON formatting
    # so that it can be ingested by pandas.
    print('Loading the Raw Data...')

    TRAIN_FRACTION = 0.8

    with open('./data/News_Category_Dataset_v2.json') as file:
        lines = file.readlines()

        # Move some of those lines aside for a test set. Want to make sure that
        # all algorithms have a consistent test set for benchmarking purposes.
        train_mask = np.random.rand(len(lines)) < TRAIN_FRACTION

        train_lines = []
        test_lines = []

        for i, line in enumerate(lines):
            if train_mask[i]:
                train_lines.append(line)
            else:
                test_lines.append(line)

        train_json = f'[{",".join(train_lines)}]'
        test_json = f'[{",".join(test_lines)}]'

    # Saving json after format has been fixed and
    # has been split into train / test data.

    with open('./data/train_data.json', 'a') as file:
        file.write(train_json)

    with open('./data/test_data.json', 'a') as file:
        file.write(test_json)

    # Load training data into dataframe.
    data = pd.read_json(train_json, orient='records')

    start_time = time.time()

    print('Processing large dataset...')
    data_large, labels_large, _encoder_large, _decoder_large = process_data(
        data, min_word_freq=None, should_downsample=False)

    print('Processing small dataset...')
    data_small, labels_small, _encoder_small, _decoder_small = process_data(
        data, min_word_freq=MIN_WORD_FREQ, should_downsample=True)

    end_time = time.time()

    print(f'Processed data in {(end_time - start_time)/60:.02f}m')

    print(f'len(data_large) == {len(data_large)}')
    print(f'len(data_small) == {len(data_small)}')
    print()
    print(f'len(data_large.columns) == {len(data_large.columns)}')
    print(f'len(data_small.columns) == {len(data_small.columns)}')

    print('Saving data frames to disk...')

    start_time = time.time()

    print(f'[1/2] Saving large data...')
    data_large.to_pickle('./data/data_large.pickle')
    labels_large.to_pickle('./data/label_large.pickle')

    print(f'[2/2] Saving small data...')
    data_small.to_pickle('./data/data_small.pickle')
    labels_small.to_pickle('./data/label_small.pickle')

    end_time = time.time()

    print(f'Saved data frames in {(end_time - start_time)/60:.02f}m')


if __name__ == '__main__':
    main()
