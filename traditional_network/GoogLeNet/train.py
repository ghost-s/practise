import torch
import numpy as np
import torchvision
from torch import optim
from torchvision.transforms import transforms
import model
import torch.nn as nn
import os
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu");
print(device);
# input 3*224*224
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    ]
);


def build_dataset(is_train):
    path = path = os.path.join('D:/codingsoftware/PyCharm/resnet18', 'train' if is_train else 'test');
    dataset = torchvision.datasets.ImageFolder(path, transform=transform);
    return dataset


trainset = build_dataset(True);
testset = build_dataset(False);
trainloader = torch.utils.data.DataLoader(trainset, batch_size=36,
                                           shuffle=True, num_workers=0);
testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                          shuffle=False, num_workers=0);

# test_data_iter = iter(testloader);
# test_images, test_label = test_data_iter.next();


net = model.GoogLeNet(in_channels=3, num_classes=36, aux_logits=True);
loss_function = nn.CrossEntropyLoss();
optimizer = optim.Adam(net.parameters(), lr=0.001);
for epoch in range(5):
    running_loss = 0.0;
    acc = 0;
    net.train();
    for step, data in enumerate(trainloader, start=0):

        inputs, labels = data;

        optimizer.zero_grad();

        outputs, aux1, aux2 = net(inputs);
        loss0 = loss_function(aux1, labels.to(device));
        loss1 = loss_function(aux2, labels.to(device));
        loss2 = loss_function(outputs, labels.to(device));
        loss = loss0 + loss1*0.3 + loss2*0.3;

        loss.backward();
        optimizer.step();

        running_loss += loss.item();

    net.eval();
    with torch.no_grad():
        for data_test in testloader:
            test_images, test_label = data_test;
            outputs = net(test_images.to(device));
            print("outputs.shape:", outputs.shape);
            print("outputs", outputs);
            predict_y = torch.max(outputs, dim=1)[1];  #0返回概率 1是index
            print("torch.max(outputs, dim=1", torch.max(outputs, dim=1));
            acc += (predict_y == test_label.to(device)).sum().item()
        accuracy = acc/len(testset);
        print('[epoch %d] train_loss: %.3f test_accuracy: %.3f'%
              (epoch + 1, running_loss/len(trainloader), acc));

print('Finish Training');

save_path = './googLeNet.pth';
torch.save(net.state_dict(), save_path);