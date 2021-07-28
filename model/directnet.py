import torch
import torch.nn as nn
import model.resnest as resnest

class DirectNet(nn.Module):
    
    def __init__(self,encode_net,encode_dim=128,in_channels=1,num_classes=8,final_drop=0.5):
        super(DirectNet,self).__init__()
        self.encode_dim = encode_dim
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.encode_net = self.get_encoder(encode_net)

        self.drop = nn.Dropout(final_drop) if final_drop > 0.0 else None
        self.fc = nn.Sequential(
                nn.Linear(10 * self.encode_dim, 256),
                nn.Dropout(final_drop) if final_drop > 0.0 else nn.Identity(),
                nn.ReLU(inplace=True),
                nn.Linear(256, self.num_classes)
            )

    def forward(self,x):
        output_list = []
        for i in range(x.size(1)):
            input_x = torch.unsqueeze(x[:,i],1)
            output_list.append(self.encode_net(input_x))
        cat_x = torch.cat(output_list,1)
        
        x = self.fc(cat_x)

        return x

    def get_encoder(self,net_name):
        if net_name.startswith('resnest'):
            net = resnest.__dict__[net_name](input_channels=self.in_channels,num_classes=self.encode_dim)
        else:
            raise ValueError('the {} is unavailable!!'%net_name)
        return net



def directnet18(**kwargs):
    net = DirectNet('resnest18',encode_dim=128,**kwargs)

    return net


def directnet50(**kwargs):
    net = DirectNet('resnest50',encode_dim=128,**kwargs)

    return net