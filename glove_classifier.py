import data_utils
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from data_utils import WordEmbeddingEncoding, WordTokenDataset
from time import time
from torch.utils.data import Dataset, DataLoader

EPOCHS = 25

def main():

    print('Loading and Setting Up Data...')

    embeddings = data_utils.load_embeddings('./data/glove.6B/glove.6B.100d.txt',
                                            embedding_dim=100)

    data = pd.read_json('./data/train_data.json', orient='records')

    train_test_split = 0.95
    split_idx = math.floor(len(data) * train_test_split)

    train_data = data.iloc[0:split_idx]
    valid_data = data.iloc[split_idx:]

    encoding = WordEmbeddingEncoding(data, embeddings)
    encoding.prepare()

    train_dataset = WordTokenDataset(train_data, encoding)
    train_dataset.prepare()

    valid_dataset = WordTokenDataset(valid_data, encoding)
    valid_dataset.prepare()

    print('Creating Model...')

    hyperparams_list = [
        {'weighting': 'uniform', 'lr': 0.001,  'batch_size': 100},
        {'weighting': 'uniform', 'lr': 0.01,   'batch_size': 100},
        {'weighting': 'uniform', 'lr': 0.001,  'batch_size': 10},
        {'weighting': 'uniform', 'lr': 0.01,   'batch_size': 10},
        {'weighting': 'tf_idf',  'lr': 0.001,  'batch_size': 100},
        {'weighting': 'tf_idf',  'lr': 0.01,   'batch_size': 100},
        {'weighting': 'tf_idf',  'lr': 0.001,  'batch_size': 10},
        {'weighting': 'tf_idf',  'lr': 0.01,   'batch_size': 10},
    ]

    class Model(torch.nn.Module):
        def __init__(self, embeddings, n_classes, weighting):
            super(Model, self).__init__()

            self.weighting = weighting

            torch_embeddings = torch.FloatTensor(embeddings.values)
            self.embedding_bag = torch.nn.EmbeddingBag.from_pretrained(
                torch_embeddings, mode='sum')
            self.linear = torch.nn.Linear(
                self.embedding_bag.embedding_dim, n_classes)

        def forward(self, samples):
            if weighting == 'tf_idf':
                weights = samples.create_tf_idf_weights()
            else:
                weights = samples.create_uniform_weights()

            x = self.embedding_bag(
                samples.sequence, samples.offset, per_sample_weights=weights)
            output = self.linear(x)
            return output

        def predict(self, samples):
            with torch.no_grad():
                outputs = self(samples)
                predictions = torch.argmax(outputs, axis=1)

            return predictions

    print('Training the Model...')

    def train(model, criterion, optimizer, dataset, data_loader, epochs, log=True):
        train_losses = []

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

            if log and (epoch + 1) % 10 == 0:
                train_loss_estimator_size = 10000
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

        return train_losses

    models = []
    train_losses_list = []
    valid_losses = []

    accepted_tokens = {t for t in embeddings.index}

    for i, hyperparams in enumerate(hyperparams_list):
        print(f'Starting training Model {i+1} / {len(hyperparams_list)}...')

        start_time = time()

        batch_size = hyperparams['batch_size']
        lr = hyperparams['lr']
        weighting = hyperparams['weighting']

        # 1. Setup Data Loader

        data_loader = DataLoader(dataset=train_dataset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 collate_fn=data_utils.collate_samples)

        # 2. Create the Model

        model = Model(embeddings=embeddings,
                      n_classes=encoding.n_classes(),
                      weighting=weighting)

        # 3. Setup Criterion and Optimizer

        criterion = torch.nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # 4. Train the Model

        train_losses = train(model,
                             criterion,
                             optimizer,
                             train_dataset,
                             data_loader,
                             epochs=EPOCHS)

        # 5. Calculate Validation Loss

        with torch.no_grad():
            valid_samples = valid_dataset[:]

            outputs = model(valid_samples)

            valid_loss = criterion(outputs, valid_samples.label)
            valid_losses.append(valid_loss)

        end_time = time()

        models.append(model)
        train_losses_list.append(train_losses)

        print(f'Model completed in {(end_time - start_time)/60:.02f}m.')
        print()

    print('Checking Results...')

    uniform_mask = [hp['weighting'] == 'uniform' for hp in hyperparams_list]

    uniform_models = [m for i, m in enumerate(models) if uniform_mask[i]]
    uniform_train_losses_list = [losses for i, losses in enumerate(
        train_losses_list) if uniform_mask[i]]
    uniform_valid_losses = [loss.item() for i, loss in enumerate(
        valid_losses) if uniform_mask[i]]

    tf_idf_models = [m for i, m in enumerate(models) if not uniform_mask[i]]
    tf_idf_train_losses_list = [losses for i, losses in enumerate(
        train_losses_list) if not uniform_mask[i]]
    tf_idf_valid_losses = [loss.item() for i, loss in enumerate(
        valid_losses) if not uniform_mask[i]]

    best_uniform_model_idx = uniform_valid_losses.index(
        min(uniform_valid_losses))
    best_uniform_model = uniform_models[best_uniform_model_idx]

    best_tf_idf_model_idx = tf_idf_valid_losses.index(min(tf_idf_valid_losses))
    best_tf_idf_model = tf_idf_models[best_tf_idf_model_idx]

    print(f'Best Uniform Model: {best_uniform_model_idx+1}')
    print(f'Best TF-IDF Model:  {best_tf_idf_model_idx+1}')

    print('Computing Uniform Model Accuracy...')

    samples = valid_dataset[:]

    uniform_predictions = best_uniform_model.predict(valid_samples)

    total = len(valid_samples.label)
    correct = torch.sum(uniform_predictions == valid_samples.label)

    print(f'Accuracy of Uniform Model: {(float(correct) / total)*100:.02f}%.')

    print('Computing TF-IDF Model Accuracy...')

    tf_idf_predictions = best_tf_idf_model.predict(samples)

    total = len(samples.label)
    correct = torch.sum(tf_idf_predictions == samples.label)

    print(f'Accuracy of TF-IDF Model: {(float(correct) / total)*100:.02f}%.')

    print('Persisting Models...')

    torch.save(best_uniform_model.state_dict(),
               './models/uniform_glove_model.torch')
    torch.save(best_tf_idf_model.state_dict(), './models/tf_idf_model.torch')

    print('Done!')


if __name__ == main():
    start_time = time()
    main()
    end_time = time()

    print(f'Script ran in {(end_time - start_time)/60:.02f}m.')
