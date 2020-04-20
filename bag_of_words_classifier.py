import data_utils
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import string
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

from guppy import hpy
from data_utils import BOWEncoding, WordTokenDataset
from nltk.tokenize.regexp import WordPunctTokenizer
from torch.utils.data import DataLoader

EPOCHS = 20


class Model(nn.Module):
    def __init__(self, vocab_size, n_classes):
        super(Model, self).__init__()
        self.vocab_size = vocab_size

        self.linear = nn.Linear(vocab_size, n_classes)

    def forward(self, samples):
        bow = samples.create_bow_matrix().type(torch.float32)
        output = self.linear(bow)
        return output

    def predict(self, samples):
        with torch.no_grad():
            outputs = self(samples)
            predictions = torch.argmax(outputs, axis=1)

        return predictions


def train(model, criterion, optimizer, dataset, data_loader, epochs, hp, log=True):
    train_losses = []
    print_every = 1
    train_loss_estimator_size = 10000

    for epoch in range(epochs):
        losses = []

        for i, samples in enumerate(data_loader):
            optimizer.zero_grad()
            output = model(samples)
            loss = criterion(output, samples.label)
            loss.backward()
            optimizer.step()

            losses.append(loss)

        train_loss = torch.mean(torch.stack(losses))
        train_losses.append(train_loss)

        if log and (epoch + 1) % print_every == 0:
            train_loss_estimator_start = max(
                1, len(dataset) - train_loss_estimator_size)
            random_start = torch.randint(
                high=train_loss_estimator_start, size=(1,)).item()

            samples = dataset[random_start:(
                random_start+train_loss_estimator_size)]
            predictions = model.predict(samples)
            labels = samples.label

            total = len(labels)
            correct = torch.sum(labels == predictions)

            print(f'Epoch {epoch + 1}')
            print(f'Accuracy: {float(correct)/total*100:.02f}%.')
            print(f'Training Loss: {train_loss.item()}')
            print()

            h = hp.heap()
            print(h)
            print()

    return train_losses


def train_multiple(hyperparams_list, train_dataset, valid_dataset, encoding, epochs, hp):
    models = []
    train_losses_list = []
    valid_losses = []

    for i, hyperparams in enumerate(hyperparams_list):
        print(f'Starting training Model {i+1} / {len(hyperparams_list)}...')

        print(hp.heap())
        print()

        start_time = time.time()

        batch_size = hyperparams['batch_size']
        lr = hyperparams['lr']

        # 1. Setup Data Loader

        data_loader = DataLoader(dataset=train_dataset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 collate_fn=data_utils.collate_samples)

        # 2. Create the Model

        model = Model(vocab_size=encoding.vocab_size,
                      n_classes=encoding.n_classes())

        # 3. Setup Criterion and Optimizer

        criterion = torch.nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # 4. Train the Model

        train_losses = train(model,
                             criterion,
                             optimizer,
                             train_dataset,
                             data_loader,
                             epochs,
                             hp)

        # 5. Calculate Validation Loss

        with torch.no_grad():
            valid_samples = valid_dataset[:]
            outputs = model(valid_samples)
            valid_loss = criterion(outputs, valid_samples.label)
            valid_losses.append(valid_loss)

        end_time = time.time()

        models.append(model)
        train_losses_list.append(train_losses)

        print(f'Model completed in {(end_time - start_time)/60:.02f}m.')
        print()

    return models, train_losses_list, valid_losses


def create_confusion_matrix(labels, predictions):
    # Displaying a confusion matrix of the validation results for our model.

    categories = labels.unique()
    category_encoder = {c.item(): i for i, c in enumerate(categories)}

    confusion_matrix = np.random.rand(len(categories), len(categories))

    for i, category in enumerate(categories):
        row = np.zeros(len(categories))

        cat_mask = (labels == category.item()).tolist()
        cat_preds = predictions[cat_mask]

        for category in categories:
            pred_count = torch.sum(cat_preds == category)
            row[category_encoder[category.item()]] = pred_count

        confusion_matrix[i, :] = row / len(cat_preds)

    return confusion_matrix, category_encoder


def top_k_labeling_errors(confusion_matrix, category_decoder, k):

    # ZSubtract 1 from diagonal so that the values along
    # diagonal do not show up as top values.
    diag = np.eye(confusion_matrix.shape[0])
    mat = confusion_matrix - diag

    label_errors = []

    for i in range(k):
        argmax = np.argmax(mat)

        # Getting row and col components. Note we are assuming
        # matrix has 2 dimensions.
        row = math.floor(argmax / mat.shape[0])
        col = argmax % mat.shape[1]

        label_error = (category_decoder[row], category_decoder[col])
        label_errors.append(label_error)

        # Zero out the element so we find a different argmax
        # on next pass.
        mat[row, col] = 0

    return label_errors


def main():
    hp = hpy()
    hp.setrelheap()

    print('Loading and setting up the data...')

    data = pd.read_json('./data/train_data.json', orient='records')
    data = data.sample(frac=1)

    train_test_split = 0.95
    split_idx = math.floor(len(data) * train_test_split)

    train_data = data.iloc[0:split_idx]
    valid_data = data.iloc[split_idx:]

    encoding = BOWEncoding(data, min_word_freq=5)
    encoding.prepare()

    train_dataset = WordTokenDataset(train_data, encoding)
    train_dataset.prepare()

    valid_dataset = WordTokenDataset(valid_data, encoding)
    valid_dataset.prepare()

    print('Done setting up data')
    print(hp.heap())
    print()

    print('Training...')

    hyperparams_list = [
        {'batch_size': 100, 'lr': 1e-3},
        {'batch_size': 10,  'lr': 1e-3},
        {'batch_size': 100, 'lr': 1e-2},
        {'batch_size': 10,  'lr': 1e-2},
    ]

    models, train_loss_list, valid_losses = train_multiple(hyperparams_list,
                                                           train_dataset,
                                                           valid_dataset,
                                                           encoding,
                                                           epochs=EPOCHS,
                                                           hp=hp)

    print('Viewing results of training...')
    best_model_idx = torch.argmin(torch.FloatTensor(valid_losses)).item()

    best_model = models[best_model_idx]

    print(f'Best Model: {best_model_idx+1}')

    valid_samples = valid_dataset[:]

    predictions = best_model.predict(valid_samples)

    total = len(valid_samples.label)
    correct = torch.sum(predictions == valid_samples.label)
    accuracy = float(correct) / total

    print(f'Accuracy of Best Model: {accuracy*100:.02f}%.')

    confusion_matrix, category_encoder = create_confusion_matrix(
        valid_samples.label,
        predictions)

    category_decoder = {i: c for c, i in category_encoder.items()}

    labeling_errors = top_k_labeling_errors(
        confusion_matrix, category_decoder, k=5)
    label_decoder = {i: l for l, i in encoding._label_encoder.items()}

    # Looking at the most frequent labeling errors.
    for i, error in enumerate(labeling_errors):
        error_0 = label_decoder[error[0]]
        error_1 = label_decoder[error[1]]
        print(f'{i+1}. "{error_0}" confused for "{error_1}"')

    print('Persisting Model...')
    torch.save(best_model.state_dict(), './models/bow_model.torch')


if __name__ == '__main__':
    main()
