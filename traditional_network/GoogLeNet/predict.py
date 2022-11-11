import PIL.ImageShow
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import model
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu");
    print(device);
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    );

    class_dict = {'apple': 0, 'banana': 1, 'beetroot': 2, 'bell pepper': 3, 'cabbage': 4, 'capsicum': 5, 'carrot': 6, 'cauliflower': 7,
                  'chilli pepper': 8, 'corn': 9, 'cucumber': 10, 'eggplant': 11, 'garlic': 12, 'ginger': 13, 'grapes': 14, 'jalepeno': 15,
                  'kiwi': 16, 'lemon': 17, 'lettuce': 18, 'mango': 19, 'onion': 20, 'orange': 21, 'paprika': 22, 'pear': 23, 'peas': 24,
                  'pineapple': 25, 'pomegranate': 26, 'potato': 27, 'raddish': 28, 'soy beans': 29, 'spinach': 30, 'sweetcorn': 31,
                  'sweetpotato': 32, 'tomato': 33, 'turnip': 34, 'watermelon': 35};

    net = model.GoogLeNet(in_channels=3, num_classes=36, aux_logits=False, _init_weight_=False);
    missing_keys, unexpected_keys = net.load_state_dict(torch.load('googLeNet.pth'), strict=False);

    img = Image.open('apple.jpg');
    img = transform(img);
    img = torch.unsqueeze(img, dim=0);

    with torch.no_grad():
        outputs = net(img);
        predict = torch.max(outputs, dim=1)[1].data.numpy();
    print(int(predict));


if __name__ == '__main__':
    main();