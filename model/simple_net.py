import torch
from torch import nn



def BasicBlock(cin, cout, kenerl_size=3):
    """
    Construct basic block with BatchNorm placed after each Conv2d
    :param cin: Num of input channels
    :param cout: Num of output channels
    :param n: Num of conv layers
    """
    layers = [
        nn.Conv2d(cin, cout, kenerl_size, stride=2, padding=1),
        nn.BatchNorm2d(cout),
        nn.ReLU(True)
    ]
    return nn.Sequential(*layers)


class SimpleNet(nn.Module):

    def __init__(self, block, num_features, depth=4, num_classes=2, input_channels=1,final_drop=0.5):
        """
        Construct a SimpleNet
        """
        super(SimpleNet, self).__init__()
        self.backbone = []
        self.backbone.append(block(input_channels, num_features[0],7))
        for i in range(1,depth):
            self.backbone.append(block(num_features[i-1], num_features[i]))

        self.backbone = nn.Sequential(*self.backbone)
        self.bridge = nn.AdaptiveMaxPool2d((1, 1))
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
    """Constructs a simplenet with 6 layers. 
    """
    model = SimpleNet(BasicBlock, [64,128,256,512], 4, **kwargs)
    return model
