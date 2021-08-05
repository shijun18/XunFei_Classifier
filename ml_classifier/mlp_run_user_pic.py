import os
import shutil
import pickle
from numpy.core.fromnumeric import shape

from pandas._config import config
from torch import nn
import torch
import pandas as pd
import random
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast,GradScaler
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,f1_score
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from torchvision import transforms

from utils import dfs_remove_weight
from feature_selection import select_feature_linesvc

import warnings
warnings.filterwarnings('ignore')

ACTIVATION = {
    'relu':nn.ReLU(inplace=True),
    'elu':nn.ELU(inplace=True),
    'leakyrelu':nn.LeakyReLU(inplace=True),
    'prelu':nn.PReLU(),
    'gelu':nn.GELU(),
    'tanh':nn.Tanh()
}


class RandomMask1d(object):
    def __init__(self,factor=0.1,p=0.5):
        self.factor = factor
        self.p = p
    
    def __call__(self,seq):
        """
        seq: vector of 1d
        """
        if np.random.rand() > self.p:
            mask = (np.random.random_sample(seq.shape) > self.factor).astype(np.float32)
            seq = seq * mask

        return seq 

class RandomScale1d(object):
    def __init__(self,factor=0.2,p=0.5):
        self.factor = factor
        self.p = p
    
    def __call__(self,seq):
        """
        seq: vector of 1d
        """
        if np.random.rand() > self.p:
            mask = (np.random.random_sample(seq.shape)*self.factor + (1.0-self.factor/2)).astype(np.float32)
            seq = seq * mask

        return seq 


class MyDataset(Dataset):
    def __init__(self, X, Y=None,transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self,index):
        data = self.X[index]
        if self.transform is not None:
            data = self.transform(data)
        if self.Y is not None:
            target = self.Y[index]
            sample = {'data':torch.from_numpy(data), 'target':int(target)}
        else:
            sample = {'data':torch.from_numpy(data)}
        
        return sample


class MLP_CLASSIFIER(nn.Module):
    def __init__(self, input_size, output_size=245, depth=3, depth_list=[256,128,64], drop_prob=0.5, use_norm=True,activation=None):
        super(MLP_CLASSIFIER, self).__init__()

        assert len(depth_list) == depth
        self.linear_list = []
        for i in range(depth):
            if i == 0:
                self.linear_list.append(nn.Linear(input_size,depth_list[i]))
            else:
                self.linear_list.append(nn.Linear(depth_list[i-1],depth_list[i]))
            if use_norm:
                self.linear_list.append(nn.BatchNorm1d(depth_list[i]))
            if activation is None:
                self.linear_list.append(nn.ReLU(inplace=True))
            else:
                self.linear_list.append(activation)
            # self.linear_list.append(nn.Tanh())
            # self.linear_list.append(nn.Dropout(0.2))
            

        self.linear = nn.Sequential(*self.linear_list)
        self.drop = nn.Dropout(drop_prob) if drop_prob > 0.0 else None
        self.cls_head = nn.Linear(depth_list[-1],output_size)

    def forward(self, x):
        x = self.linear(x) #N*C
        if self.drop:
            x = self.drop(x)
        x = self.cls_head(x)
        return x


class AverageMeter(object):
    '''
    Computes and stores the average and current value
    '''

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    '''
    Computes the precision@k for the specified values of k
    '''
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(1/batch_size))
    return res


def train_epoch(epoch,net,criterion,optim,train_loader,scaler,use_fp16=True):
    net.train()
   
    train_loss = AverageMeter()
    train_acc = AverageMeter()

    for step, sample in enumerate(train_loader):

        data = sample['data']
        target = sample['target']

        # print(data.size())
        data = data.cuda()
        target = target.cuda()
        with autocast(use_fp16):
            output = net(data)
            loss = criterion(output, target)
        
        optim.zero_grad()
        if use_fp16:
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            loss.backward()
            optim.step()

        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        acc = accuracy(output.data, target)[0]
        train_loss.update(loss.item(), data.size(0))
        train_acc.update(acc.item(), data.size(0))

        torch.cuda.empty_cache()

        # if step % 10 == 0:
        #     print('epoch:{},step:{},train_loss:{:.5f},train_acc:{:.5f},lr:{}'
        #             .format(epoch, step, loss.item(), acc.item(), optim.param_groups[0]['lr']))

    return train_acc.avg,train_loss.avg


def val_epoch(epoch,net,criterion,val_loader,use_fp16=True):
    
    net.eval()

    val_loss = AverageMeter()
    val_acc = AverageMeter()

    with torch.no_grad():
        for step, sample in enumerate(val_loader):
            data = sample['data']
            target = sample['target']

            data = data.cuda()
            target = target.cuda()
            with autocast(use_fp16):
                output = net(data)
                loss = criterion(output, target)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            acc = accuracy(output.data, target)[0]
            val_loss.update(loss.item(), data.size(0))
            val_acc.update(acc.item(), data.size(0))

            torch.cuda.empty_cache()

            # print('epoch:{},step:{},val_loss:{:.5f},val_acc:{:.5f}'
            #     .format(epoch, step, loss.item(), acc.item()))

    return val_acc.avg,val_loss.avg


def evaluation(test_data,net,weight_path,use_fp16=True):
    ckpt = torch.load(weight_path)
    net.load_state_dict(ckpt['state_dict'])
    net.eval()
    test_data = torch.from_numpy(test_data) #N*fea_len
    with torch.no_grad():
        data = test_data.cuda()
        with autocast(use_fp16):
            output = net(data)
        output = output.float()
        prob_output = torch.softmax(output,dim=1) # N*C
        output = torch.argmax(prob_output,dim=1) #N
        torch.cuda.empty_cache()

    return output.cpu().numpy().tolist(),prob_output.cpu().numpy()


def evaluation_tta(test_data,net,weight_path,use_fp16=True,tta=5):
    ckpt = torch.load(weight_path)
    net.load_state_dict(ckpt['state_dict'])
    net.eval()

    transform = transforms.Compose([RandomScale1d()])

    vote_out = []
    prob_out = []
    for _ in range(tta):
        input_data = transform(test_data)
        input_data = torch.from_numpy(input_data) #N*fea_len
        with torch.no_grad():
            data = input_data.cuda()
            with autocast(use_fp16):
                output = net(data)
            output = output.float()
            prob_output = torch.softmax(output,dim=1) #N*C
            output = torch.argmax(prob_output,dim=1) #N
            vote_out.append(output.cpu().numpy().tolist())
            prob_out.append(prob_output.cpu().numpy()) # tta*N*C
            torch.cuda.empty_cache()
    vote_array = np.asarray(vote_out).astype(np.uint8) # tta*N
    vote_out = [max(list(vote_array[:,i]),key=list(vote_array[:,i]).count) for i in range(vote_array.shape[1])]
    return vote_out, np.mean(prob_out, axis=0)


def eval_metric(result_dict,test_id,pred_result):
    if result_dict is not None:
        true_result = [result_dict[case] for case in test_id]
        acc = accuracy_score(true_result,pred_result)
        f1 = f1_score(true_result,pred_result,average='macro')

        print('Evaluation:')
        print('Accuracy:%.5f'%acc)
        print('F1 Score:%.5f'%f1)

        return acc,f1
    else:
        return np.nan,np.nan

def fea_encoder(train_df,test_df,fea_name=None,label=None):
    
    target = train_df[label]
    del train_df[label]
    data = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    data = data.fillna(0)
    le = LabelEncoder()
    data[fea_name] = le.fit_transform(data[fea_name])

    train_df = data[:train_df.shape[0]]
    train_df[label] = target
    test_df = data[train_df.shape[0]:]
    # print(train_df)
    # print(test_df)
    return train_df,test_df


def fea_pad(train_df,test_df,fea_name=None,label=None,pad_val=0):
    
    target = train_df[label]
    del train_df[label]
    data = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    data[fea_name] = data[fea_name].apply(lambda x: pad_val if x == 'NULL' else x)

    train_df = data[:train_df.shape[0]]
    train_df[label] = target
    test_df = data[train_df.shape[0]:]
    # print(train_df)
    # print(test_df)
    return train_df,test_df


def time_encoder(train_df,test_df,fea_name=None,label=None):
    
    target = train_df[label]
    del train_df[label]
    data = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    data = data.fillna(0)
    data[fea_name] = data[fea_name].apply(lambda x:len(eval(x)))

    train_df = data[:train_df.shape[0]]
    train_df[label] = target
    test_df = data[train_df.shape[0]:]
    # print(train_df)
    # print(test_df)
    return train_df,test_df


def run(train_path,test_path,result_path,output_dir,net_depth=3,exclude_list=None,scale_flag=True,select_flag=False,dim_reduction=False,aug_flag=False,tta=False,**aux_config):

    torch.manual_seed(0)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        shutil.rmtree(output_dir)
        os.makedirs(output_dir)
    
    # load training and testing data
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    test_id = test_df['pid']

    # print(len(list(test_id)))

    # data preprocessing
    fea_list = [f for f in train_df.columns if f not in exclude_list] 
    test_df = test_df[fea_list]
    train_df = train_df[fea_list + ['label']]

    # train_df,test_df = fea_pad(train_df,test_df,'gender','label',pad_val=2)
    train_df,test_df = fea_pad(train_df,test_df,'age','label',pad_val=0)
    train_df,test_df = time_encoder(train_df,test_df,'time','label')
    train_df,test_df = fea_encoder(train_df,test_df,'province','label')
    train_df,test_df = fea_encoder(train_df,test_df,'city','label')
    train_df,test_df = fea_encoder(train_df,test_df,'model','label')


    # result
    if result_path is not None:
        from utils import csv_reader_single
        result_dict = csv_reader_single(result_path,'image_id','category_id')    
    else:
        result_dict = None
    

    # convert to numpy array
    Y = np.asarray(train_df['label']).astype(np.uint8)
    X = np.asarray(train_df[fea_list]).astype(np.float32)
    test = np.asarray(test_df[fea_list]).astype(np.float32)
    
    # feature selection
    if select_flag:
        select_model_path = './select_model.pkl'
        if os.path.exists(select_model_path):
            with open(select_model_path, 'rb') as f:
                select_model = pickle.load(f)
        else:
            select_model = select_feature_linesvc(X, Y, select_model_path)

        X = select_model.transform(X)
        test = select_model.transform(test)

    # data scale
    if scale_flag:
        X_len = X.shape[0]
        data_scaler = StandardScaler()
        cat_data = np.concatenate([X,test],axis=0)
        cat_data= data_scaler.fit_transform(cat_data)

        X = cat_data[:X_len]
        test = cat_data[X_len:]
    

    # dim reduction
    if dim_reduction:
        X_len = X.shape[0]
        cat_data = np.concatenate([X,test],axis=0)
        # fastica = FastICA(n_components=int(X.shape[1]*0.5),random_state=0)
        # cat_data= fastica.fit_transform(cat_data)

        pca = PCA(n_components=int(X.shape[1]*0.5))
        cat_data= pca.fit_transform(cat_data)
        
        X = cat_data[:X_len]
        test = cat_data[X_len:]

    # print(Y)
    print(X.shape,test.shape)

    num_classes = len(set(train_df['label'])) #245
    fea_len = X.shape[1]

    total_result = []
    total_prob_result = []
    kfold = KFold(n_splits=5,shuffle=True,random_state=21)
    for fold_num,(train_index,val_index) in enumerate(kfold.split(X)):
        print(f'***********fold {fold_num+1} start!!***********')
        fold_dir = os.path.join(output_dir,f'fold{fold_num+1}')
        if not os.path.exists(fold_dir):
            os.makedirs(fold_dir)

        # initialization
        epoch_num = 100
        acc_threshold = 0.0

        depth = net_depth
        depth_list = [fea_len*(2**(depth-1-i)) for i in range(depth)]
        
        net = MLP_CLASSIFIER(fea_len,num_classes,depth,depth_list,**aux_config)
        criterion = nn.CrossEntropyLoss()
        optim = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=0.0001)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, [25,60,80], gamma=0.1)
        scaler = GradScaler()
        
        net = net.cuda()
        criterion = criterion.cuda()

        # data loader
        x_train, x_val = X[train_index], X[val_index]
        y_train, y_val = Y[train_index], Y[val_index]

        print('Train Data Size:',x_train.shape)
        print('Val Data Size:',x_val.shape)

        if aug_flag:
            transform = transforms.Compose([RandomScale1d()])
        else:
            transform = None

        train_dataset = MyDataset(X=x_train,Y=y_train,transform=transform)
        val_dataset = MyDataset(X=x_val,Y=y_val,transform=transform)

        train_loader = DataLoader(
                train_dataset,
                batch_size=256,
                shuffle=True,
                num_workers=2)
        
        val_loader = DataLoader(
                val_dataset,
                batch_size=256,
                shuffle=False,
                num_workers=2)

        # main processing
        for epoch in range(epoch_num):
            train_acc, train_loss = train_epoch(epoch,net,criterion,optim,train_loader,scaler)
            val_acc,val_loss = val_epoch(epoch,net,criterion,val_loader)
            torch.cuda.empty_cache()

            if epoch % 10 == 0:
                print('Train epoch:{},train_loss:{:.5f},train_acc:{:.5f}'
                    .format(epoch, train_loss, train_acc))

                print('Val epoch:{},val_loss:{:.5f},val_acc:{:.5f}'
                    .format(epoch, val_loss, val_acc))


            if lr_scheduler is not None:
                lr_scheduler.step()
            
            if val_acc > acc_threshold:
                acc_threshold = val_acc
                saver = {
                    'state_dict': net.state_dict()
                }

                file_name = 'epoch:{}-val_acc:{:.5f}-val_loss:{:.5f}-mlp.pth'.format(epoch,val_acc,val_loss)
                save_path = os.path.join(fold_dir, file_name)
                print('Save as: %s'%file_name)
                torch.save(saver, save_path)
        
        # save top3 model
        dfs_remove_weight(fold_dir,retain=1)

        # generating test result using the best model
        if tta:
            fold_result,fold_prob_result = evaluation_tta(test,net,save_path,tta=9) #N,N*C
        else:
            fold_result,fold_prob_result = evaluation(test,net,save_path) #N,N*C
        total_result.append(fold_result)
        total_prob_result.append(fold_prob_result)
        
        fold_prob_result = np.argmax(fold_prob_result,axis=1).astype(np.uint8).tolist()
        print(f'\nfold {fold_num+1} vote:')
        acc_vote,f1_vote = eval_metric(result_dict,test_id,fold_result)
        print(f'\nfold {fold_num+1} prob:')
        acc_prob,f1_prob = eval_metric(result_dict,test_id,fold_prob_result)

        # csv save
        fold_vote_csv = {}
        fold_vote_csv = pd.DataFrame(fold_vote_csv)
        fold_vote_csv['user_id'] = list(test_id)
        fold_vote_csv['category_id'] = list(fold_result)
        fold_vote_csv.to_csv(os.path.join(output_dir, f'fold{fold_num}_acc-{round(acc_vote,4)}_f1-{round(f1_vote,4)}-vote.csv'),index=False)

        fold_prob_csv = {}
        fold_prob_csv = pd.DataFrame(fold_prob_csv)
        fold_prob_csv['user_id'] = list(test_id)
        fold_prob_csv['category_id'] = list(fold_prob_result)
        fold_vote_csv.to_csv(os.path.join(output_dir, f'fold{fold_num}_acc-{round(acc_prob,4)}_f1-{round(f1_prob,4)}-prob.csv'),index=False)

    # result fusion by voting
    final_vote_result = []
    vote_array = np.asarray(total_result).astype(np.uint8) #fold*N
    final_vote_result.extend([max(list(vote_array[:,i]),key=list(vote_array[:,i]).count) for i in range(vote_array.shape[1])])
    print('\nfusion vote:')
    acc_vote,f1_vote = eval_metric(result_dict,test_id,final_vote_result)

    # result fusion by avg prob
    final_prob_result = []
    prob_array = np.asarray(total_prob_result) # fold*N*C
    prob_array = np.mean(prob_array,axis=0) #N*C
    final_prob_result = np.argmax(prob_array,axis=1).astype(np.uint8).tolist()
    print('\nfusion prob:')
    acc_prob,f1_prob = eval_metric(result_dict,test_id,final_prob_result)

    # csv save
    total_vote_csv = {}
    total_vote_csv = pd.DataFrame(total_vote_csv)
    total_vote_csv['user_id'] = list(test_id)
    total_vote_csv['category_id'] = list(final_vote_result)
    total_vote_csv.to_csv(os.path.join(output_dir, f'fusion_acc-{round(acc_vote,4)}_f1-{round(f1_vote,4)}-vote.csv'),index=False)

    total_prob_csv = {}
    total_prob_csv = pd.DataFrame(total_prob_csv)
    total_prob_csv['user_id'] = list(test_id)
    total_prob_csv['category_id'] = list(final_prob_result)
    total_prob_csv.to_csv(os.path.join(output_dir, f'fusion_acc-{round(acc_prob,4)}_f1-{round(f1_prob,4)}-prob.csv'),index=False)


if __name__ == '__main__':
    feas = ['pid','label','gender','age','tagid','time','province','city','model','make']
    
    train_path = './dataset/user_pic/train.csv'
    test_path = './dataset/user_pic/test.csv'

    result_path = None
    output_dir = './result/user_pic'

    os.environ['CUDA_VISIBLE_DEVICES'] = '2'

    for i in range(7):
    # for i,act in enumerate(ACTIVATION):
        save_path = os.path.join(output_dir,f'trial_{str(i+1)}')
        # total
        exclude_list = ['label','pid','tagid','make']
    
        print('exclude list:',exclude_list)
        # print('activation fun:',act)
        run(train_path,test_path,result_path,save_path,net_depth=10,exclude_list=exclude_list,scale_flag=True,select_flag=False,dim_reduction=False,aug_flag=False,tta=False)
