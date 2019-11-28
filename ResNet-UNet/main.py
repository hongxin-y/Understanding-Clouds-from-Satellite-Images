from model import UNetWithResnet50Encoder, criterion
from utils import ImageDataset, printMask0
from torch import nn, optim
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
import numpy as np
import pandas as pd
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(model, criterion, optimizer, train_dataloader, test_dataloader, num_epochs=30):
    model.train()
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        dt_size = len(train_dataloader.dataset)
        epoch_loss = 0
        step = 0
        for x, y, imgName, labels in train_dataloader:
            step += 1
            inputs = x.to(device)
            truth_masks = y.to(device)
            optimizer.zero_grad()
            output_labels, output_masks = model(inputs)
            loss = criterion(output_masks, truth_masks)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            print("%d/%d,train_loss:%0.3f" % (step, (dt_size - 1) // train_dataloader.batch_size + 1, loss.item()))
            if step % 100 == 0:
                torch.cuda.empty_cache()
            if step % 500 == 0:
                torch.save(model, './checkpoint/model_%d_step.pth' % step)

        print('-' * 10)
        print("epoch %d loss:%0.3f" % (epoch, epoch_loss / step))
        print('-' * 10)

        state = {'net':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch}
        torch.save(state, './checkpoint/model_%d_epoch.pth' % epoch)

        test_model(model, criterion, test_dataloader)

    return model


def test_model(model, criterion, test_dataloader, print_out = False):
    model.eval()
    epoch_loss = 0
    step = 0
    data = []
    for x, y, imgName, labels in test_dataloader:
        step += 1
        inputs = x.to(device)
        truth_masks = y.to(device)
        with torch.no_grad():
            output_labels, output_masks = model(inputs)
        loss = criterion(output_masks, truth_masks)
        if print_out:
            for batch in range(len(imgName)):
                output_label = output_labels[batch].cpu().data.numpy()
                label = labels[batch].cpu().data.numpy()
                line = ["{}.jpg".format(imgName[batch])] + list(output_label) + list(label)
                data.append(line)

                print_output_masks = output_masks[batch].cpu()
                print_label_masks = truth_masks[batch].cpu()

                print_output_masks = print_output_masks.data.numpy()
                print_output_masks *= (print_output_masks > 0)
                print_output_masks = print_output_masks * (print_output_masks <= 1) + 1 * (print_output_masks > 1)
                print_output_masks = (print_output_masks * 255).astype(np.uint8)

                print_label_masks = print_label_masks.data.numpy()
                print_label_masks = (print_label_masks * 255).astype(np.uint8)

                printMask0("./data/output/{}_output".format(imgName[batch]), print_output_masks)
                printMask0("./data/output/{}_truth".format(imgName[batch]), print_label_masks)


        epoch_loss += loss.item()
    df = pd.DataFrame(data, columns=["file_name", "fish_output", "flower_output", "gravel_output", "sugar_output",
                                     "fish_real", "flower_real", "gravel_output", "sugar_output"])
    df.to_csv("./data/output.csv", index=False)
    print("test set loss:%0.3f" % (epoch_loss / step))


def train():
    model = UNetWithResnet50Encoder().to(device)
    optimizer = optim.Adam(model.parameters())

    x_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    y_transforms = transforms.ToTensor()
    dataSet = ImageDataset("./data/modified_train_images", "./data/train_mask", transform=x_transforms, target_transform=y_transforms, augment_transform=True)
    train_size = int(0.8 * len(dataSet))
    test_size = len(dataSet) - train_size
    train_dataset, test_dataset = random_split(dataSet, [train_size, test_size])

    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)
    test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=True, num_workers=0)

    train_model(model, criterion, optimizer, train_dataloader, test_dataloader)


def test():
    checkpoint = torch.load("./checkpoint/model_6_epoch.pth")
    model = UNetWithResnet50Encoder().to(device)
    model.load_state_dict(checkpoint['net'])
    x_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    y_transforms = transforms.ToTensor()
    dataSet = ImageDataset("./data/modified_test_images", "./data/test_mask", transform=x_transforms, target_transform=y_transforms)
    test_dataloader = DataLoader(dataSet, batch_size=1, shuffle=True, num_workers=0)
    test_model(model, criterion, test_dataloader, True)

if __name__ == "__main__":
    print("start!")
    train()
    # test()