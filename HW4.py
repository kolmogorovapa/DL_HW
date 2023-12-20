import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader
from torch.utils.tensorboard import SummaryWriter

torch.manual_seed(2023)

writer = SummaryWriter(comment='CNN')
train_indices = torch.arange(50000)
test_indices = torch.arange(10000)

transform = transforms.Compose(
    [
        # transforms.ColorJitter(hue=0.05, saturation=0.05),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ]
)

base_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ]
)

train_cifar_dataset = Subset(datasets.CIFAR100(
    train=True,
    download=True, root='./', transform=transform), train_indices)

test_cifar_dataset = Subset(datasets.CIFAR100(
    train=False, download=True, root='./', transform=base_transform),test_indices)

train_dl = DataLoader(train_cifar_dataset, shuffle=True, batch_size=100)
test_dl = DataLoader(test_cifar_dataset, batch_size=100)

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.conv3 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.05)
        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 100)


    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x =  self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)


        return F.softmax(x, dim=1)

model = ConvNet()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
loss_fn = torch.nn.CrossEntropyLoss()

num_epochs = 20

for epoch in range(num_epochs):
    error = 0
    for x, y in train_dl:
        model.train()
        optimizer.zero_grad()
        prediction = model(x)
        loss = loss_fn(prediction, y)
        error += loss

        loss.backward()
        optimizer.step()

    writer.add_scalar('Loss', error/len(train_cifar_dataset), epoch)
    print('Loss', error/len(train_cifar_dataset))

    correct_guess = 0
    for x, y in test_dl:
        model.eval()
        prediction = model(x)
        predicted_indices = torch.argmax(prediction)
        correct_guess += (predicted_indices == y).float().sum()

    writer.add_scalar('Accuracy', correct_guess/len(test_cifar_dataset), epoch)
    print('Accuracy', correct_guess/len(test_cifar_dataset))