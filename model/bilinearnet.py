import torch
import torch.nn as nn
import model.resnest as resnest
import model.resnet as resnet

class BilinearNet(nn.Module):
    
    def __init__(self,encode_net,encode_dim=128,in_channels=1,num_classes=8,final_drop=0.5):
        super(BilinearNet,self).__init__()
        self.encode_dim = encode_dim
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.encode_net = self.get_encoder(encode_net)

        self.features =[nn.Sequential(self.encode_net.conv1, self.encode_net.bn1, self.encode_net.relu),
                        nn.Sequential(self.encode_net.maxpool, self.encode_net.layer1),
                        self.encode_net.layer2,
                        self.encode_net.layer3,
                        self.encode_net.layer4
                        ]
        
        self.features = nn.Sequential(*self.features)

        self.drop = nn.Dropout(final_drop) if final_drop > 0.0 else None
        self.fc = nn.Sequential(
                nn.Linear(512**2, 256) if '50' not in encode_net else nn.Linear(2048**2, 256),
                nn.Dropout(final_drop) if final_drop > 0.0 else nn.Identity(),
                nn.ReLU(inplace=True),
                nn.Linear(256, self.num_classes)
            )

    def forward(self,x):
        x = self.features(x)
        # print(x.size())
        N,C,H,W = x.size()
        x = x.view(N,C,H*W)
        x = (x @ x.permute(0, 2, 1).contiguous()) / (H*W)
        x = x.view(N,C**2)
        x = torch.sqrt(x + 1e-5)
        x = torch.nn.functional.normalize(x)
        x = self.fc(x)
        return x

    def get_encoder(self,net_name):
        if net_name.startswith('resnest'):
            net = resnest.__dict__[net_name](input_channels=self.in_channels,num_classes=self.encode_dim)
        if net_name.startswith('resnet'):
            net = resnet.__dict__[net_name](input_channels=self.in_channels,num_classes=self.encode_dim)
        else:
            raise ValueError('the {} is unavailable!!'%net_name)
        return net



def bilinearneSt18(**kwargs):
    net = BilinearNet('resnest18',**kwargs)

    return net


def bilinearneSt50(**kwargs):
    net = BilinearNet('resnest50',**kwargs)

    return net


def bilinearnet18(**kwargs):
    net = BilinearNet('resnet18',**kwargs)

    return net


def bilinearnet50(**kwargs):
    net = BilinearNet('resnet50',**kwargs)

    return net