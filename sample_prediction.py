import os
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    img_path = "./data/samples/roses.jpg"
    model = torch.load('./models/ViT_pre_train_20_epochs.pth', map_location=device)

    img = Image.open(img_path)
    plt.imshow(img)
    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    img = data_transform(img)

    pred = model(img)
    print(pred)
    plt.title(pred)
    plt.show()