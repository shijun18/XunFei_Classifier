import os
from torch import nn
import torch
import pandas as pd

from reg_run import scaler_normalize
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast,GradScaler

os.environ['CUDA_VISIBLE_DEVICES'] = '7'

class MLP_Regressor(nn.Module):
    def __init__(self, input_size, output_size=1, depth=3, depth_list=[256,128,64], drop_prob=0.0):
        super(MLP_Regressor, self).__init__()

        assert len(depth_list) == depth
        self.linear_list = []
        for i in range(depth):
            if i == 0:
                self.linear_list.append(nn.Linear(input_size,depth_list[i]))
            else:
                self.linear_list.append(nn.Linear(depth_list[i-1],depth_list[i]))
            # self.linear_list.append(nn.ReLU(depth_list[i]))
            # self.linear_list.append(nn.Tanh())

        self.linear = nn.Sequential(*self.linear_list)
        self.drop = nn.Dropout(drop_prob) if drop_prob > 0.0 else None
        self.reg_head = nn.Linear(depth_list[-1],output_size)

    def forward(self, x):
        x = self.linear(x) #N*C
        if self.drop:
            x = self.drop(x)
        x = self.reg_head(x)
        return x


train_path = './dataset/air/pre_train/保定2016年.csv'
train_df = pd.read_csv(train_path)

test_path = './dataset/air/pre_test/石家庄20160701-20170701.csv'
test_df = pd.read_csv(test_path)


# del train_df['日期']
# del test_df['日期']

# del train_df['质量等级']
# del test_df['质量等级']

# scale_list = ["AQI","PM2.5", "PM10", "SO2", "CO", "NO2", "O3_8h"]
scale_list = ["AQI","PM2.5", "PM10", "SO2", "CO", "NO2", "O3_8h",'month','day']
# scale_list = None
train_df,test_df = scaler_normalize(train_df,test_df,scale_list,'IPRC')

fea_list = [f for f in train_df.columns if f not in ['IPRC','month','day']]

y_train = np.asarray(train_df['IPRC']).astype(np.float32)
x_train =  np.asarray(train_df[fea_list]).astype(np.float32)
x_test =  np.asarray(test_df[fea_list]).astype(np.float32)

print(x_train[:10],y_train[:10])

input_size = x_test.shape[1]

net = MLP_Regressor(input_size,1,3,[32,32,32])
loss_func = nn.MSELoss()
optim = torch.optim.LBFGS(net.parameters())
# optim = torch.optim.Adam(net.parameters(),lr=0.05,weight_decay=0.0001)

# lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, [1000,5000,8000], gamma=0.1)
# lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optim, base_lr=0.01, max_lr=0.1)

net = net.cuda()
loss_func = loss_func.cuda()

epoch_num = 10000
max_loss = 1.0
early_stop = 0

net.train()
for epoch in range(epoch_num):
    # lr_scheduler.step()
    data = torch.from_numpy(x_train)
    target = torch.from_numpy(y_train)

    data = data.cuda()
    target = target.cuda()

    def closure():
        optim.zero_grad()
        output = net(data)
        loss = loss_func(output,target)
        loss.backward()
        return loss
    
    optim.step(closure)
    loss = closure()
    if epoch % 100 == 0:
        print('epoch:{},loss:{:.5f},lr:{:.5f}'.format(
            epoch,loss.item(),optim.param_groups[0]['lr']))
    
    early_stop += 1

    if loss.item() < max_loss:
        torch.save({'state_dict': net.state_dict()},'./mlp_model.pth')
        print('Save Model!! When train loss = {:.5f}'.format(loss.item()))
        max_loss = loss.item()
        early_stop = 0

    # if early_stop > 1000:
    #     break