import torch.nn as nn
import torch.nn.functional as F
import torch

'''
    The structure of UNet is referenced from https://github.com/JavisPeng/u_net_liver/blob/master/unet.py
    adding criterion and change the way of residual.
'''

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return x


class Model(nn.Module):
    def __init__(self, in_channels=3, out_channels=4):
        super(Model, self).__init__()
        self.conv1 = DoubleConv(in_channels, 64)
        self.maxPool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(64, 128)
        self.maxPool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(128, 256)
        self.maxPool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(256, 512)
        self.maxPool4 = nn.MaxPool2d(2)

        self.conv5 = DoubleConv(512, 1024)

        self.up6 = Up(1024, 512)
        self.conv6 = DoubleConv(1024, 512)
        self.up7 = Up(512, 256)
        self.conv7 = DoubleConv(512, 256)
        self.up8 = Up(256, 128)
        self.conv8 = DoubleConv(256, 128)
        self.up9 = Up(128, 64)
        self.conv9 = DoubleConv(128, 64)
        self.conv10 = nn.Conv2d(64, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        batch_size = x.shape[0]
        c1 = self.conv1(x)
        p1 = self.maxPool1(c1)
        c2 = self.conv2(p1)
        p2 = self.maxPool2(c2)
        c3 = self.conv3(p2)
        p3 = self.maxPool3(c3)
        c4 = self.conv4(p3)
        p4 = self.maxPool4(c4)
        c5 = self.conv5(p4)

        up6 = self.up6(c5, c4)
        c6 = self.conv6(up6)
        up7 = self.up7(c6, c3)
        c7 = self.conv7(up7)
        up8 = self.up8(c7, c2)
        c8 = self.conv8(up8)
        up9 = self.up9(c8, c1)
        c9 = self.conv9(up9)
        c10 = self.conv10(c9)
        probability_mask  = torch.sigmoid(c10)
        probability_label = F.adaptive_max_pool2d(probability_mask,1).view(batch_size,-1)
        return probability_label, probability_mask

def criterion(pred, truth):
    loss_mask = F.binary_cross_entropy(pred, truth, reduction='mean')
    return loss_mask

if __name__ == "__main__":
    print(0)
    model = Model()
    print(1)
    inp = torch.rand((2, 3, 320, 480))
    print(inp.shape)
    out = model(inp)
    print(out[0].shape)
    print(out[1].shape)
    criterion(out[1], out[1])
