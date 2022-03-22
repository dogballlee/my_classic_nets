import torch as th
import torchvision
from model import my_res18, matplotlib_imshow
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # 为了解决报OMP错误的问题


def main():
    # ------------------------------------------data and data_loader------------------------------------------
    train_data = torchvision.datasets.CIFAR10('cifar', True, transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]), download=True)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

    test_data = torchvision.datasets.CIFAR10('cifar', False, transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]), download=True)
    test_loader = DataLoader(test_data, batch_size=128, shuffle=False)

    # ------------------------------------------------device------------------------------------------------
    device = 'cuda:0' if th.cuda.is_available() else 'cpu'
    print('*****************Now we are using {} to train the model*****************'.format(device))

    writer = SummaryWriter(log_dir='logs')

    x, x_label = iter(train_loader).next()
    print('train_data_shape:', x.shape, 'train_label_shape:', x_label.shape)

    model = my_res18().to(device)
    # --------------------------------------将模型结构传入模型tensorboard--------------------------------------
    init_img = th.zeros((1, 3, 32, 32), device=device)
    writer.add_graph(model, init_img)

    # display first batch's image from CIFAR10 in tensorboard
    dataiter = iter(train_loader)
    imgs, lbls = dataiter.next()
    img_grid = torchvision.utils.make_grid(imgs)
    matplotlib_imshow(img_grid, one_channel=True)
    writer.add_image('64_CIFAR10_images', img_grid)

    criteon = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    print(model)

    for epoch in range(1):
        model.train()
        for batchidx, (x, label) in enumerate(train_loader):
            x, label = x.to(device), label.to(device)
            logits = model(x)
            loss = criteon(logits, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(epoch, ' loss:', loss.item())

        model.eval()
        with th.no_grad():
            total_correct = 0
            total_num = 0
            for x, label in test_loader:
                x, label = x.to(device), label.to(device)
                logits = model(x)
                pred = logits.argmax(dim=1)
                correct = th.eq(pred, label).float().sum().item()
                total_correct += correct
                total_num += x.size(0)

            acc = total_correct / total_num
            print(epoch, ' test:', acc)
            tags = ['train_loss', 'accuracy', 'learning_rate']
        writer.add_scalar(tags[0], loss.item(), epoch)
        writer.add_scalar(tags[1], acc, epoch)
        # writer.add_scalar(tags[2], optimizer.param_groups[0]['lr'], epoch)
        writer.add_histogram(tag='conv0',
                             values=model.conv0.weight,
                             global_step=epoch)
        writer.add_histogram(tag='layer0',
                             values=model.layer0[0].conv0.weight,
                             global_step=epoch)

        th.save(model.state_dict(), './weights.pth')


if __name__ == '__main__':
    main()
