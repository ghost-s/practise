import torch
import numpy as np
import torchvision
from torch import optim
from torchvision.transforms import transforms
import model
import torch.nn as nn
import os

transform = transforms.Compose(
    [
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    ]
);
# trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
#                                         download=True, transform=transform);
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=36,
#                                           shuffle=True, num_workers=0);
#
# testset = torchvision.datasets.CIFAR10(root='./data', train=False,
#                                         download=True, transform=transform);
# testloader = torch.utils.data.DataLoader(testset, batch_size=10000,
#                                           shuffle=False, num_workers=0);

def build_dataset(is_train):

    path = path = os.path.join('D:/codingsoftware/PyCharm/resnet18', 'train' if is_train else 'test');
    dataset = torchvision.datasets.ImageFolder(path, transform=transform);
    return dataset

trainset = build_dataset(True);

testset = build_dataset(False);

trainloader = torch.utils.data.DataLoader(trainset, batch_size=36,
                                           shuffle=True, num_workers=0);

testloader = torch.utils.data.DataLoader(testset, batch_size=1000,
                                          shuffle=False, num_workers=0);

test_data_iter = iter(testloader);
test_images, test_label = test_data_iter.next();


net = model.LeNet();
loss_function = nn.CrossEntropyLoss();
optimizer = optim.Adam(net.parameters(), lr=0.001);
for epoch in range(5):
    running_loss = 0.0;
    for step, data in enumerate(trainloader, start = 0):

        inputs, labels = data;

        optimizer.zero_grad();

        outputs = net(inputs);
        loss = loss_function(outputs, labels);

        loss.backward();
        optimizer.step();

        running_loss += loss.item();
        if step % 10 == 9:
            with torch.no_grad():
                outputs = net(test_images);
                print("outputs.shape:", outputs.shape);
                print("outputs", outputs);
                predict_y = torch.max(outputs, dim=1)[1];  #0返回概率 1是index
                print("torch.max(outputs, dim=1", torch.max(outputs, dim=1));
                accuracy = (predict_y == test_label).sum().item() / test_label.size(0);

                print('[%d %5d] train_loss: %.3f test_accuracy: %.3f'%
                      (epoch + 1, step + 1, running_loss / 500, accuracy));
                running_loss = 0.0;
print('Finish Training');

save_path = './Lenet6.pth';
torch.save(net.state_dict(), save_path);










