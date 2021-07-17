import torch
import torch.nn as nn
# import torch.autograd as Autograd
import torch.utils.data as Data
import torchvision
import my_LeNet5
# import torch.utils.tensorboard as tensorboard
from tqdm import tqdm


# 超参数
def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


device = get_device()
print(device)

EPOCH = 35
BATCH_SIZE = 64
BATCH_SIZE_TEST = 1000
LR = 2e-3
WEIGHT_DECAY = 1e-3
download_mnist = True

train_data = torchvision.datasets.MNIST(
    root='./mnist',
    train=True,
    transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                              torchvision.transforms.Normalize((0.1307,), (0.3081,))]),
    download=download_mnist
)
train_loader = Data.DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True
)


test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(
        './mnist',
        train=False,
        download=True,
        transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                  torchvision.transforms.Normalize((0.1307,), (0.3081,))])),
        batch_size=BATCH_SIZE_TEST,
        shuffle=True)

my_le_net5 = my_LeNet5.LeNet5()
optimizer = torch.optim.Adam(my_le_net5.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
loss_func = nn.CrossEntropyLoss()


# 训练&验证
def main():
    net = my_le_net5
    net = net.to(device)
    best_acc = 0.0
    model_path = './pre_lenet5.ckpt'
    print(net)

    # -------------------------------TRAIN---------------------------------- #
    for epoch in range(EPOCH):
        train_loss = []
        train_accs = []
        for batch in tqdm(train_loader):
            data, labels = batch
            data, labels = data.to(device), labels.to(device)
            logits = net(data)
            loss = loss_func(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = (logits.argmax(dim=-1) == labels).float().mean()

            train_loss.append(loss.item())
            train_accs.append(acc)

        train_loss = sum(train_loss) / len(train_loss)
        train_acc = sum(train_accs) / len(train_accs)
        print(f"[Train | {epoch + 1:03d}/{EPOCH:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

        # -------------------------------Test----------------------------------- #
        net.eval()
        valid_loss = []
        valid_accs = []

        for batch in tqdm(test_loader):
            data, labels = batch
            data, labels = data.to(device), labels.to(device)

            with torch.no_grad():
                logits = net(data)
            loss = loss_func(logits, labels)

            acc = (logits.argmax(dim=-1) == labels).float().mean()
            valid_loss.append(loss.item())
            valid_accs.append(acc)

        valid_loss = sum(valid_loss) / len(valid_loss)
        valid_acc = sum(valid_accs) / len(valid_accs)

        print(f"[ Valid | {epoch + 1:03d}/{EPOCH:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")

        # 如果acc有提升，则保存当前模型
        if valid_acc > best_acc:
            best_acc = valid_acc
            torch.save(net.state_dict(), model_path)
            print('save model with acc :{}'.format(best_acc))


if __name__ == "__main__":
    main()
