import argparse
import sys

import torch
import click

from data import mnist
from model import MyAwesomeModel
from torch import optim, nn


@click.group()
def cli():
    pass


@click.command()
@click.option("--lr", default=1e-3, help='learning rate to use for training')
def train(lr):
    print("Training day and night")
    print(lr)

    # TODO: Implement training loop here
    model = MyAwesomeModel()
    train_set, test_set = mnist()
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train(model, train_set, test_set, criterion, optimizer, 10)

    torch.save(model.state_dict(), 'model.pth')

@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    print("Evaluating until hitting the ceiling")
    print(model_checkpoint)

    # TODO: Implement evaluation logic here
    model = torch.load(model_checkpoint)
    _, test_set = mnist()

    criterion = nn.NLLLoss()
    model.eval()

    # Turn off gradients for validation, will speed up inference
    with torch.no_grad():
        test_loss, accuracy = model.validation(model, test_set, criterion)
    
    print("Test Loss: {:.3f}.. ".format(test_loss/len(test_set)),
            "Test Accuracy: {:.3f}".format(accuracy/len(test_set)))


cli.add_command(train)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()

  