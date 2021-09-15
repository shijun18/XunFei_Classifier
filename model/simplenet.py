import torch
from torch import nn



def BasicBlock(cin, cout, kenerl_size=(5,12), stride=1, use_norm=False,padding=0):
    """
    Construct basic block with BatchNorm placed after each Conv2d
    :param cin: Num of input channels
    :param cout: Num of output channels
    :param n: Num of conv layers
    """
    layers = [nn.Conv2d(cin, cout, kenerl_size, stride=stride, padding=padding, bias=False)]
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
        kenerl_size_list = [(15,15),(15,15),(15,15),(15,15)]
        self.backbone = []
        self.backbone.append(block(input_channels, num_features[0], kenerl_size=kenerl_size_list[0],stride=2))
        for i in range(1,depth):
            self.backbone.append(block(num_features[i-1], num_features[i], kenerl_size=kenerl_size_list[i], stride=2, use_norm=i%2==1))
        self.backbone = nn.Sequential(*self.backbone)
        
        # self.bridge = nn.AdaptiveAvgPool2d((1, 1))
        self.bridge = nn.AdaptiveMaxPool2d((1, 1))
        self.drop = nn.Dropout(final_drop) if final_drop > 0.0 else None
        self.cls = nn.Sequential(
            nn.Linear(num_features[-1], num_classes)
        )

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.backbone(x)
        x = self.bridge(x).view(x.size(0),-1)
        if self.drop:
            x = self.drop(x)
        x = self.cls(x)
        return x



class SimpleNetV2(nn.Module):

    def __init__(self, block, num_features, depth=4, num_classes=2, input_channels=1,final_drop=0.0):
        """
        Construct a SimpleNet
        """
        super(SimpleNetV2, self).__init__()
        kenerl_size_list = [(17,17),(17,17),(17,17),(17,17)]
        self.backbone = []
        self.backbone.append(block(input_channels, num_features[0], kenerl_size=kenerl_size_list[0],stride=2))
        for i in range(1,depth):
            self.backbone.append(block(num_features[i-1], num_features[i], kenerl_size=kenerl_size_list[i], stride=2, use_norm=i%2==1))
        self.backbone = nn.Sequential(*self.backbone)
        # print(self.backbone)
        # self.bridge = nn.AdaptiveAvgPool2d((1, 1))
        self.bridge = nn.AdaptiveMaxPool2d((1, 1))
        self.drop = nn.Dropout(final_drop) if final_drop > 0.0 else None
        self.cls = nn.Sequential(
            nn.Linear(num_features[-1], num_classes)
        )

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.backbone(x)
        x = self.bridge(x).view(x.size(0),-1)
        if self.drop:
            x = self.drop(x)
        x = self.cls(x)
        return x



class SimpleNetV3(nn.Module):

    def __init__(self, block, num_features, depth=4, num_classes=2, input_channels=1,final_drop=0.0):
        """
        Construct a SimpleNet
        """
        super(SimpleNetV3, self).__init__()
        self.fea_extractor = []
        kenerl_size_list = [[(19,19),(19,19),(19,19),(19,19)],\
                           [(15,15),(15,15),(15,15),(15,15)],\
                           [(11,11),(11,11),(11,11),(11,11)],\
                           [(7,7),(7,7),(7,7),(7,7)],\
                           [(3,3),(3,3),(3,3),(3,3)]]

        for kenerl_size in kenerl_size_list:
            backbone = []
            for i in range(depth):
                if i == 0:
                    backbone.append(block(input_channels, num_features[i], kenerl_size=kenerl_size[i],stride=2,use_norm=i%2==1,padding=(kenerl_size[i][0]//2,kenerl_size[i][1]//2)))
                else:
                    backbone.append(block(num_features[i-1], num_features[i], kenerl_size=kenerl_size[i],stride=2,use_norm=i%2==1,padding=(kenerl_size[i][0]//2,kenerl_size[i][1]//2)))
            self.fea_extractor.append(nn.Sequential(*backbone))
        self.fea_extractor = nn.ModuleList(self.fea_extractor)
        self.bridge =nn.AdaptiveMaxPool2d((1, 1))
        self.drop = nn.Dropout(final_drop) if final_drop > 0.0 else None
        self.cls = nn.Sequential(
            nn.Linear(num_features[-1]*len(kenerl_size_list), num_classes)
        )

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

    def forward(self, x):
        fea_list = []
        for extractor in self.fea_extractor:
            fea = extractor(x)
            fea_list.append(fea)
        x = torch.cat(fea_list,dim=1)
        x = self.bridge(x)
        x = torch.flatten(x, 1)
        if self.drop:
            x = self.drop(x)
        x = self.cls(x)
        return x


class SimpleNetV4(nn.Module):

    def __init__(self, block, num_features, depth=4, num_classes=2, input_channels=1,final_drop=0.0):
        """
        Construct a SimpleNet
        """
        super(SimpleNetV4, self).__init__()
        self.fea_extractor = []
        kenerl_size_list =[[(21,21),(21,21),(21,21)],\
                           [(17,17),(17,17),(17,17)],\
                           [(13,13),(13,13),(13,13)],\
                           [(9,9),(9,9),(9,9)],\
                           [(5,5),(5,5),(5,5)]]

        for kenerl_size in kenerl_size_list:
            backbone = []
            for i in range(depth-1):
                if i == 0:
                    backbone.append(block(input_channels, num_features[i], kenerl_size=kenerl_size[i],stride=2,use_norm=i%2==1,padding=(kenerl_size[i][0]//2,kenerl_size[i][1]//2)))
                else:
                    backbone.append(block(num_features[i-1], num_features[i], kenerl_size=kenerl_size[i],stride=2,use_norm=i%2==1,padding=(kenerl_size[i][0]//2,kenerl_size[i][1]//2)))
            self.fea_extractor.append(nn.Sequential(*backbone))
        self.fea_extractor = nn.ModuleList(self.fea_extractor)
        self.fusion = block(num_features[-2]*len(kenerl_size_list),num_features[-1],kenerl_size=(5,5),stride=2,use_norm=True)
        self.bridge = nn.AdaptiveMaxPool2d((1, 1))
        self.drop = nn.Dropout(final_drop) if final_drop > 0.0 else None
        self.cls = nn.Sequential(
            nn.Linear(num_features[-1], num_classes)
        )

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

    def forward(self, x):
        fea_list = []
        for extractor in self.fea_extractor:
            fea = extractor(x)
            fea_list.append(fea)
        x = torch.cat(fea_list,dim=1)
        x = self.fusion(x)
        x = self.bridge(x)
        x = torch.flatten(x, 1)
        if self.drop:
            x = self.drop(x)
        x = self.cls(x)
        return x

'''
class SimpleNetV5(nn.Module):

    def __init__(self, block, num_features, depth=4, num_classes=2, input_channels=1,final_drop=0.0):
        """
        Construct a SimpleNet
        """
        super(SimpleNetV5, self).__init__()
        self.fea_extractor = []
        kenerl_size_list =[[(5,15),(5,15),(5,15)],\
                           [(15,5),(15,5),(15,5)]]

        for kenerl_size in kenerl_size_list:
            backbone = []
            for i in range(depth-1):
                if i == 0:
                    backbone.append(block(input_channels, num_features[i], kenerl_size=kenerl_size[i],stride=2,use_norm=i%2==1,padding=(kenerl_size[i][0]//2,kenerl_size[i][1]//2)))
                else:
                    backbone.append(block(num_features[i-1], num_features[i], kenerl_size=kenerl_size[i],stride=2,use_norm=i%2==1,padding=(kenerl_size[i][0]//2,kenerl_size[i][1]//2)))
            self.fea_extractor.append(nn.Sequential(*backbone))
        self.fea_extractor = nn.ModuleList(self.fea_extractor)
        self.fusion = block(num_features[-2]*len(kenerl_size_list),num_features[-1],kenerl_size=(1,1),stride=1,use_norm=True)
        self.bridge = nn.AdaptiveMaxPool2d((1, 1))
        self.drop = nn.Dropout(final_drop) if final_drop > 0.0 else None
        self.cls = nn.Sequential(
            nn.Linear(num_features[-1], num_classes)
        )

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

    def forward(self, x):
        fea_list = []
        for extractor in self.fea_extractor:
            fea = extractor(x)
            fea_list.append(fea)
        x = torch.cat(fea_list,dim=1)
        x = self.fusion(x)
        x = self.bridge(x)
        x = torch.flatten(x, 1)
        if self.drop:
            x = self.drop(x)
        x = self.cls(x)
        return x
'''

class SimpleNetV5(nn.Module):

    def __init__(self, block, num_features, depth=4, num_classes=2, input_channels=1,final_drop=0.0):
        """
        Construct a SimpleNet
        """
        super(SimpleNetV5, self).__init__()
        self.fea_extractor = []
        kenerl_size_list =[[(15,15),(15,15),(15,15),(15,15)],\
                           [(11,11),(11,11),(11,11),(11,11)],\
                           [(7,7),(7,7),(7,7),(7,7)],\
                           [(3,3),(3,3),(3,3),(3,3)]]

        for kenerl_size in kenerl_size_list:
            backbone = []
            for i in range(depth):
                if i == 0:
                    backbone.append(block(input_channels, num_features[i], kenerl_size=kenerl_size[i],stride=2,use_norm=i%2==1,padding=(kenerl_size[i][0]//2,kenerl_size[i][1]//2)))
                else:
                    backbone.append(block(num_features[i-1], num_features[i], kenerl_size=kenerl_size[i],stride=2,use_norm=i%2==1,padding=(kenerl_size[i][0]//2,kenerl_size[i][1]//2)))
            self.fea_extractor.append(nn.Sequential(*backbone))
        self.fea_extractor = nn.ModuleList(self.fea_extractor)
        self.bridge =nn.AdaptiveMaxPool2d((1, 1))
        self.drop = nn.Dropout(final_drop) if final_drop > 0.0 else None
        self.cls = nn.Sequential(
            nn.Linear(num_features[-1]*len(kenerl_size_list), num_classes)
        )

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

    def forward(self, x):
        fea_list = []
        for extractor in self.fea_extractor:
            fea = extractor(x)
            fea_list.append(fea)
        x = torch.cat(fea_list,dim=1)
        x = self.bridge(x)
        x = torch.flatten(x, 1)
        if self.drop:
            x = self.drop(x)
        x = self.cls(x)
        return x



def simplenet(**kwargs):
    
    model = SimpleNet(BasicBlock, [64,128,256,512], 4, **kwargs)
    return model


def simplenetv2(**kwargs):
    
    model = SimpleNetV2(BasicBlock, [64,128,256,512], 4, **kwargs)
    return model


def simplenetv3(**kwargs):
    
    model = SimpleNetV3(BasicBlock, [64,128,256,512], 4, **kwargs)
    return model


def simplenetv4(**kwargs):
    
    model = SimpleNetV4(BasicBlock, [64,128,256,512], 4, **kwargs)
    return model


def simplenetv5(**kwargs):
    
    model = SimpleNetV5(BasicBlock, [64,128,256,512], 4, **kwargs)
    return model


if __name__ == "__main__":
  
  net = simplenetv3(input_channels=1,num_classes=2)

  from torchsummary import summary
  import os 
  os.environ['CUDA_VISIBLE_DEVICES'] = '2'
  net = net.cuda()
  summary(net,input_size=(1,512,512),batch_size=1,device='cuda')