from torch import flatten
from torch import nn
import matplotlib.pyplot as plt


class MyAwesomeModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=50,
                               kernel_size=5,
                               stride=1,
                               padding=0)

        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv2 = nn.Conv2d(in_channels=50,
                               out_channels=100,
                               kernel_size=(5, 5))

        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.fc1 = nn.Linear(in_features=2500, out_features=500)
        self.relu3 = nn.ReLU()

        self.fc2 = nn.Linear(in_features=500, out_features=10)
        self.logSoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # Your code here!
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        x = flatten(x, 1)
        x = self.fc1(x)
        x = self.relu3(x)

        x = self.fc2(x)
        x = self.logSoftmax(x)

        return x


def train(model, trainloader, testloader, criterion, optimizer=None, epochs=5, print_every=40):
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    steps = 0
    running_loss = 0
    loss_total = []
    for e in range(epochs):
        # Model in training mode, dropout is on
        model.train()
        for images, labels in trainloader:
            steps += 1

            images = images.float()
            labels = labels.long()

            # Flatten images into a 784 long vector
            # images.resize_(images.size()[0], 784)

            optimizer.zero_grad()

            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            loss_total.append(loss.item())
            if steps % print_every == 0:
                # Model in inference mode, dropout is off
                model.eval()

                # Turn off gradients for validation, will speed up inference
                with torch.no_grad():
                    test_loss, accuracy = validation(
                        model, testloader, criterion)

                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(
                          running_loss/print_every),
                      "Test Loss: {:.3f}.. ".format(test_loss/len(testloader)),
                      "Test Accuracy: {:.3f}".format(accuracy/len(testloader)))

                running_loss = 0

                # Make suredropout and grads are on for training
                model.train()
    fig, ax = plt.subplots()
    ax.plot(loss_total, label="Training Loss")
    legend = ax.legend(loc='upper center', fontsize='x-large')
    legend.get_frame().set_facecolor('C0')
    plt.show()


def validation(model, testloader, criterion):
    accuracy = 0
    test_loss = 0
    for images, labels in testloader:

        # images = images.resize_(images.size()[0], 784)
        images = images.float()
        labels = labels.long()

        output = model.forward(images)
        test_loss += criterion(output, labels).item()

        # Calculating the accuracy
        # Model's output is log-softmax, take exponential to get the probabilities
        ps = torch.exp(output)
        # Class with highest probability is our predicted class, compare with true label
        equality = (labels.data == ps.max(1)[1])
        # Accuracy is number of correct predictions divided by all predictions, just take the mean
        accuracy += equality.type_as(torch.FloatTensor()).mean()

    return test_loss, accuracy
