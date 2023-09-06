import sys

sys.path.append('../python')
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)


def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    fn = nn.Sequential(nn.Linear(dim, hidden_dim),
                       norm(hidden_dim),
                       nn.ReLU(),
                       nn.Dropout(drop_prob),
                       nn.Linear(hidden_dim, dim),
                       norm(dim))
    return nn.Sequential(nn.Residual(fn), nn.ReLU())


def MLPResNet(dim, hidden_dim=100, num_blocks=3, num_classes=10, norm=nn.BatchNorm1d, drop_prob=0.1):
    return nn.Sequential(nn.Linear(dim, hidden_dim),
                         nn.ReLU(),
                         *[ResidualBlock(hidden_dim, hidden_dim // 2, norm, drop_prob) for _ in range(num_blocks)],
                         nn.Linear(hidden_dim, num_classes))


def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    if opt:
        model.train()
    else:
        model.eval()
    loss_fn = nn.SoftmaxLoss()
    errors = 0
    losses = 0
    num_samples = 0
    for batch in dataloader:
        x, y = batch[0], batch[1]
        predict = model(x)
        loss = loss_fn(predict, y)
        err = predict.numpy().argmax(axis=1) - y.numpy()
        errors += np.sum(err != 0)
        losses += loss.numpy() * len(err)
        num_samples += len(err)
        if opt:
            loss.backward()
            opt.step()

    return errors / num_samples, losses / num_samples


def train_mnist(batch_size=100, epochs=10, optimizer=ndl.optim.Adam,
                lr=0.001, weight_decay=0.001, hidden_dim=100, data_dir="data"):
    np.random.seed(4)
    training_dataset = ndl.data.MNISTDataset(data_dir + "/train-images-idx3-ubyte.gz",
                                             data_dir + "/train-labels-idx1-ubyte.gz")
    training_dataloader = ndl.data.DataLoader(training_dataset, batch_size, shuffle=True)
    test_dataset = ndl.data.MNISTDataset(data_dir + "/t10k-images-idx3-ubyte.gz",
                                         data_dir + "/t10k-labels-idx1-ubyte.gz")
    test_dataloader = ndl.data.DataLoader(test_dataset, batch_size)

    model = MLPResNet(784, hidden_dim)
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)

    training_err, training_loss = None, None
    for _ in range(epochs):
        training_err, training_loss = epoch(training_dataloader, model=model, opt=opt)
    test_err, test_loss = epoch(test_dataloader, model=model, opt=None)

    return training_err, training_loss, test_err, test_loss


if __name__ == "__main__":
    train_mnist(data_dir="../data")
