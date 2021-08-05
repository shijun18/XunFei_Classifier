import torch
from torch import nn



def BasicBlock(cin, cout, kenerl_size=(1,12), stride=1, use_norm=False):
    """
    Construct basic block with BatchNorm placed after each Conv2d
    :param cin: Num of input channels
    :param cout: Num of output channels
    :param n: Num of conv layers
    """
    layers = [nn.Conv2d(cin, cout, kenerl_size, stride=stride, padding=1)]
    if use_norm:
        layers.append(nn.BatchNorm2d(cout))
    layers.append(nn.ReLU(True))
    return nn.Sequential(*layers)


class SimpleNet(nn.Module):

    def __init__(self, block, num_features, depth=4, num_classes=2, input_channels=1,final_drop=0.0):
        """
        Construct a SimpleNet
        """
        super(SimpleNet, self).__init__()
        self.backbone = []
        self.backbone.append(block(input_channels, num_features[0], (1,24),stride=2))
        for i in range(1,depth):
            self.backbone.append(block(num_features[i-1], num_features[i], use_norm=i%2==0))
            self.backbone.append(nn.MaxPool2d(kernel_size=(1,12), stride=2))
        self.backbone = nn.Sequential(*self.backbone)
        self.bridge = nn.AdaptiveAvgPool2d((1, 1))
        self.drop = nn.Dropout(final_drop) if final_drop > 0.0 else None
        self.cls = nn.Sequential(
            nn.Linear(num_features[-1], num_classes)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.backbone(x)
        x = self.bridge(x)
        x = torch.flatten(x, 1)
        if self.drop:
            x = self.drop(x)
        x = self.cls(x)
        return x


def simple_net(**kwargs):
    
    model = SimpleNet(BasicBlock, [64,128,256], 3, **kwargs)
    return model
