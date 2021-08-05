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



class DirectNet_V2(nn.Module):
    
    def __init__(self,main_encode_net,aux_encode_net,encode_dim=128,in_channels=1,num_classes=8,final_drop=0.5,pretrained=True):
        super(DirectNet_V2,self).__init__()
        self.encode_dim = encode_dim
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.aux_encode_net = self.get_encoder(aux_encode_net)
        self.main_encode_net = self.get_encoder(main_encode_net)
        self.pretrained = pretrained

        self.drop = nn.Dropout(final_drop) if final_drop > 0.0 else None
        self.fc = nn.Sequential(
                nn.Linear(10 * self.encode_dim, 256),
                nn.Dropout(final_drop) if final_drop > 0.0 else nn.Identity(),
                nn.ReLU(inplace=True),
                nn.Linear(256, self.num_classes)
            )

    def forward(self,x):
        output_list = []
        for i in range(x.size(1)-1):
            input_x = torch.unsqueeze(x[:,i],1)
            output_list.append(self.aux_encode_net(input_x))
        output_list.append(self.main_encode_net(torch.unsqueeze(x[:,-1],1)))
        cat_x = torch.cat(output_list,1)
        
        x = self.fc(cat_x)

        return x

    def get_encoder(self,net_name):
        if net_name.startswith('resnest'):
            net = resnest.__dict__[net_name](input_channels=self.in_channels,num_classes=self.encode_dim)
        elif net_name.startswith('efficientnet'):
            from efficientnet_pytorch import EfficientNet
            if self.pretrained:
                net = EfficientNet.from_pretrained(model_name=net_name,
                                                in_channels=self.in_channels,
                                                num_classes=self.encode_dim,
                                                advprop=True)
            else:
                net = EfficientNet.from_name(model_name=net_name)
                num_ftrs = net._fc.in_features
                net._fc = nn.Linear(num_ftrs, self.encode_dim)
        else:
            raise ValueError('the {} is unavailable!!'%net_name)
        return net



def directnet18(**kwargs):
    net = DirectNet('resnest18',encode_dim=128,**kwargs)

    return net


def directnet50(**kwargs):
    net = DirectNet('resnest50',encode_dim=128,**kwargs)

    return net


def directnetv2_b5(**kwargs):
    net = DirectNet('efficientnet-b5','efficientnet-b5',encode_dim=128,**kwargs)

    return net


def directnetv2_b0(**kwargs):
    net = DirectNet('efficientnet-b0','efficientnet-b0',encode_dim=128,**kwargs)

    return net