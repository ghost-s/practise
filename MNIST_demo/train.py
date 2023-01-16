from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import model
import torch
import torch.nn as nn

training_data = datasets.MNIST(
    root="train",
    train=True,
    transform=ToTensor(),
    download=True)
test_data = datasets.MNIST(
    root="test",
    train=False,
    transform=ToTensor(),
    download=True)


train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)

test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

MODEL = model.MNIST_MODEL();

# Initialize the loss function
loss_fn = nn.CrossEntropyLoss()

learning_rate = 1e-3
batch_size = 64
epochs = 5


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):

    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(MODEL.parameters(), lr=learning_rate)

epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, MODEL, loss_fn, optimizer)
    test_loop(test_dataloader, MODEL, loss_fn)
print("Done!")
torch.save(MODEL.state_dict(), 'model.pth')