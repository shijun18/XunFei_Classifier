import torch
import torch.nn as nn
import model.resnest as resnest
import model.resnet as resnet

FINAL_CHANNLE = {
    'efficientnet-b0':1280,
    'efficientnet-b1':1280,
    'efficientnet-b2':1408,
    'efficientnet-b3':1536,
    'efficientnet-b4':1792,
    'efficientnet-b5':2048,
    'efficientnet-b6':2304,
    'efficientnet-b7':2560,
}


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BilinearNet(nn.Module):
    
    def __init__(self,encode_net,input_channels=1,num_classes=8,final_drop=0.5):
        super(BilinearNet,self).__init__()
        self.num_classes = num_classes
        self.in_channels = input_channels
        self.encode_net = self.get_encoder(encode_net)

        self.reduction = [conv1x1(FINAL_CHANNLE[encode_net],256),nn.BatchNorm2d(256),nn.ReLU(inplace=True)]
        self.reduction = nn.Sequential(*self.reduction)

        self.drop = nn.Dropout(final_drop) if final_drop > 0.0 else None
        self.fc = nn.Sequential(
                nn.Linear(256**2, 256),
                nn.Dropout(final_drop) if final_drop > 0.0 else nn.Identity(),
                nn.ReLU(inplace=True),
                nn.Linear(256, self.num_classes)
            )

    def forward(self,x):
        x = self.encode_net.extract_features(x)
        # print(x.size())
        x = self.reduction(x)
        N,C,H,W = x.size()
        x = x.view(N,C,H*W)
        x = (x @ x.permute(0, 2, 1).contiguous()) / (H*W)
        x = x.view(N,C**2)
        x = torch.sqrt(x + 1e-5)
        x = torch.nn.functional.normalize(x)
        x = self.fc(x)
        return x

    def get_encoder(self,net_name):
        if net_name.startswith('efficientnet'):
            from efficientnet_pytorch import EfficientNet
            net = EfficientNet.from_pretrained(model_name=net_name,
                                                in_channels=self.in_channels,
                                                num_classes=self.num_classes,
                                                advprop=True)
        return net

    
def bilinearnet_b3(**kwargs):
    net = BilinearNet('efficientnet-b3',**kwargs)

    return net



def bilinearnet_b5(**kwargs):
    net = BilinearNet('efficientnet-b5',**kwargs)

    return net
