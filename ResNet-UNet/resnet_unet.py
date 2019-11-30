import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torch

class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, padding=1, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, padding=1, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, up_channels, res_channels):
        super().__init__()

        in_channels = up_channels
        out_channels = up_channels // 2
        block_channels = out_channels + res_channels

        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv_block = Block(block_channels, out_channels)

    def forward(self, up_x, down_x):
        x = self.upsample(up_x)
        x = torch.cat([x, down_x], 1)
        x = self.conv_block(x)
        return x


class Model(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.depth = 6
        down_blocks = []
        up_blocks = []
        resnet = models.resnet50(pretrained=True)
        for p in resnet.parameters():
            p.requires_grad = False

        self.input_block = nn.Sequential(*list(resnet.children()))[:3]
        self.input_pool = list(resnet.children())[3]
        for bottleneck in list(resnet.children()):
            if isinstance(bottleneck, nn.Sequential):
                down_blocks.append(bottleneck)
        self.down_blocks = nn.ModuleList(down_blocks)

        self.bridge = Block(2048, 2048)

        up_blocks.append(UpBlock(2048, 1024))
        up_blocks.append(UpBlock(1024, 512))
        up_blocks.append(UpBlock(512, 256))
        up_blocks.append(UpBlock(256, 64))
        up_blocks.append(UpBlock(128, 3))

        self.up_blocks = nn.ModuleList(up_blocks)

        self.out = nn.Conv2d(64, num_classes, kernel_size=1, stride=1)

    def forward(self, x):
        batch_size = x.shape[0]
        layers = {}

        layers["layer_0"] = x
        x = self.input_block(x)
        layers["layer_1"] = x
        x = self.input_pool(x)
        for i in range(0, self.depth-2):
            x = self.down_blocks[i](x)
            if i < self.depth - 3:
                layers["layer_{}".format(i+2)] = x
        x = self.bridge(x)

        for i in range(0, self.depth-1):
            key = "layer_{}".format(self.depth - 2 - i)
            x = self.up_blocks[i](x, layers[key])
        x = self.out(x)
        del layers

        probability_mask  = torch.sigmoid(x)
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
