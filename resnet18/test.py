import torch as th
import torchvision
from model import my_res18, matplotlib_imshow
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch import nn, optim

PATH = './weights/model-20.pth'

test_data = torchvision.datasets.CIFAR10('cifar', False, transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]), download=True)
test_loader = DataLoader(test_data, batch_size=128, shuffle=False)

img, lbl = iter(test_loader).next()

model = my_res18()
model_dict = model.load_state_dict(th.load(PATH), strict=False)
predict = model(img)
print('predict is : ', predict, 'while the real label is : ', lbl)
