from model import UNetWithResnet50Encoder, criterion
from utils import ImageDataset
from torch import nn, optim
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(model, criterion, optimizer, train_dataloader, test_dataloader, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        dt_size = len(train_dataloader.dataset)
        epoch_loss = 0
        step = 0
        for x, y in train_dataloader:
            step += 1
            inputs = x.to(device)
            labels = y.to(device)
            optimizer.zero_grad()
            output_labels, output_masks = model(inputs)
            loss = criterion(output_masks, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            print("%d/%d,train_loss:%0.3f" % (step, (dt_size - 1) // train_dataloader.batch_size + 1, loss.item()))

        print('-' * 10)
        print("epoch %d loss:%0.3f" % (epoch, epoch_loss / step))
        print('-' * 10)
        test_model(model, criterion, test_dataloader)

        torch.save(model, 'model_%d.pth' % epoch)
    return model


def test_model(model, criterion, test_dataloader):
    model.eval()
    epoch_loss = 0
    step = 0
    for x, y in test_dataloader:
        step += 1
        inputs = x.to(device)
        labels = y.to(device)
        with torch.no_grad():
            output_labels, output_masks = model(inputs)
        loss = criterion(output_masks, labels)
        epoch_loss += loss.item()
    print("test set loss:%0.3f" % (epoch_loss / step))


def train():
    model = UNetWithResnet50Encoder().to(device)
    optimizer = optim.Adam(model.parameters())

    x_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    # x_transforms = transforms.ToTensor()
    y_transforms = transforms.ToTensor()
    dataSet = ImageDataset("./data/modified_train_images", "./data/mask", transform=x_transforms, target_transform=y_transforms)
    train_size = int(0.8 * len(dataSet))
    test_size = len(dataSet) - train_size
    train_dataset, test_dataset = random_split(dataSet, [train_size, test_size])

    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)
    test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=True, num_workers=0)

    train_model(model, criterion, optimizer, train_dataloader, test_dataloader)


if __name__ == "__main__":
    train()
