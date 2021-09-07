from enum import auto
import os,shutil
import torch
import torch.nn as nn
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torchvision import transforms
import numpy as np
import math

from torch.nn import functional as F
from torch.autograd import Variable
import torchvision.transforms as tr
from data_utils.data_loader import DataGenerator as DataGenerator
# from data_utils.data_loader import SplitDataGenerator as DataGenerator

import torch.distributed as dist
from PIL import Image
from utils import remove_dir, make_dir,dfs_remove_weight
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler
from data_utils.transforms import RandomRotate,DeNoise,SquarePad,AddNoise,Trimming

from sklearn import metrics
import random
# GPU version.


class My_Classifier(object):
    '''
    Control the training, evaluation, test and inference process.
    Args:
    - net_name: string, __all__ = [''].
    - lr: float, learning rate.
    - n_epoch: integer, the epoch number
    - channels: integer, the channel number of the input
    - num_classes: integer, the number of class
    - input_shape: tuple of integer, input dim
    - crop: integer, cropping size
    - batch_size: integer
    - num_workers: integer, how many subprocesses to use for data loading.
    - device: string, use the specified device
    - pre_trained: True or False, default False
    - weight_path: weight path of pre-trained model
    '''

    def __init__(self, net_name=None, gamma=0.1, lr=1e-3, n_epoch=1, channels=1, num_classes=3, input_shape=None, crop=48,
                 batch_size=6, num_workers=0, device=None, pre_trained=False, weight_path=None, weight_decay=0.,
                 momentum=0.95, mean=(0.105393,), std=(0.203002,), milestones=None,use_fp16=False,transform=None,
                 drop_rate=0.0,smothing=0.1,external_pretrained=False,use_mixup=False,use_cutmix=False,mix_only=False):
        super(My_Classifier, self).__init__()

        self.net_name = net_name
        self.lr = lr
        self.n_epoch = n_epoch
        self.channels = channels
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.crop = crop
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device

        self.pre_trained = pre_trained
        self.weight_path = weight_path
        self.start_epoch = 0
        self.global_step = 0
        self.loss_threshold = 1.0
        self.metric = 0.0
        # save the middle output
        self.feature_in = []
        self.feature_out = []
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.mean = mean
        self.std = std
        self.gamma = gamma
        self.milestones = milestones
        self.use_fp16 = use_fp16
        self.drop_rate = drop_rate
        self.smothing = smothing
        self.external_pretrained = external_pretrained
        self.use_mixup = use_mixup
        self.use_cutmix = use_cutmix
        self.mix_only = mix_only

        os.environ['CUDA_VISIBLE_DEVICES'] = self.device
        self.net = self._get_net(self.net_name)
        if self.pre_trained:
            self._get_pre_trained(self.weight_path)
            self.loss_threshold = eval(os.path.splitext(os.path.basename(
                self.weight_path))[0].split(':')[3].split('-')[0])
            print('loss threshold:%.4f'%self.loss_threshold)
            self.metric = eval(os.path.splitext(os.path.basename(
                self.weight_path))[0].split(':')[-1])
            print('metric threshold:%.4f'%self.metric)
        self.transform_list = [
            DeNoise(3), #1
            tr.Resize(size=self.input_shape), #2
            tr.RandomAffine(0,(0.1,0.1),(0.8,1.2)), #3
            tr.ColorJitter(brightness=.3,contrast=.3), #4
            tr.RandomPerspective(distortion_scale=0.6, p=0.5), #5
            RandomRotate([-30, -60, -90, 0, 90, 60, 30]), #6
            # tr.RandomRotation((-15,+15)), #6
            tr.RandomHorizontalFlip(p=0.5), #7
            tr.RandomVerticalFlip(p=0.5), #8
            tr.ToTensor(), #9
            tr.Normalize(self.mean, self.std), #10
            SquarePad(), #11
            AddNoise(), #12
            tr.CenterCrop(size=self.input_shape),#13
            tr.RandomCrop(size=(256,256)), #14
            tr.FiveCrop(size=(self.input_shape[0]//2,self.input_shape[1]//2)),#15
            tr.Lambda(lambda crops: torch.stack([torch.squeeze(crop) for crop in crops])), #16
            tr.RandomEqualize(p=0.5), #17
            tr.GaussianBlur(3), #18
            tr.RandomErasing(scale=(0.01, 0.05), ratio=(1.1, 1.3)), #19
            Trimming(),#20
        ]

        self.transform = [self.transform_list[i-1] for i in transform]

    def trainer(self, train_path, val_path, label_dict, cur_fold, output_dir=None, log_dir=None, optimizer='Adam',
                loss_fun='Cross_Entropy', class_weight=None, lr_scheduler=None,balance_sample=False,monitor='val_acc'):

        torch.manual_seed(1)
        print('Device:{}'.format(self.device))
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        output_dir = os.path.join(output_dir, "fold"+str(cur_fold))
        log_dir = os.path.join(log_dir, "fold"+str(cur_fold))

        if os.path.exists(log_dir):
            if not self.pre_trained:
                shutil.rmtree(log_dir)
                os.makedirs(log_dir)
        else:
            os.makedirs(log_dir)

        if os.path.exists(output_dir):
            if not self.pre_trained:
                shutil.rmtree(output_dir)
                os.makedirs(output_dir)
        else:
            os.makedirs(output_dir)

        self.writer = SummaryWriter(log_dir)
        self.global_step = self.start_epoch * \
            math.ceil(len(train_path)/self.batch_size)

        net = self.net
        lr = self.lr
        loss = self._get_loss(loss_fun, class_weight)
        weight_decay = self.weight_decay
        momentum = self.momentum

        if len(self.device.split(',')) > 1:
            net = DataParallel(net)

        # dataloader setting
        train_transformer = transforms.Compose(self.transform)

        if balance_sample:
            from data_utils.data_loader import BalanceDataGenerator as BalanceDataGenerator
            train_dataset = BalanceDataGenerator(
                train_path, label_dict, channels=self.channels, transform=train_transformer)
        else:
            train_dataset = DataGenerator(
                train_path, label_dict, channels=self.channels, transform=train_transformer)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

        # copy to gpu
        net = net.cuda()
        loss = loss.cuda()

        # optimizer setting
        optimizer = self._get_optimizer(optimizer, net, lr, weight_decay, momentum)
        scaler = GradScaler()

        # if self.pre_trained:
        #     checkpoint = torch.load(self.weight_path)
        #     optimizer.load_state_dict(checkpoint['optimizer'])

        if lr_scheduler is not None:
            lr_scheduler = self._get_lr_scheduler(lr_scheduler, optimizer)

        # acc_threshold = 0.5
        min_loss = 1.
        max_acc = 0.


        early_stopping = EarlyStopping(patience=50,verbose=True,monitor=monitor,best_score=self.metric,op_type='max')
        for epoch in range(self.start_epoch, self.n_epoch):
            train_loss, train_acc,train_f1 = self._train_on_epoch(epoch, net, loss, optimizer, train_loader,scaler)

            val_loss, val_acc,val_f1 = self._val_on_epoch(epoch, net, loss, val_path, label_dict)

            torch.cuda.empty_cache()

            if lr_scheduler is not None:
                lr_scheduler.step()

            print('Train epoch:{},train_loss:{:.5f},train_acc:{:.5f},train_f1:{:.5f}'
                  .format(epoch, train_loss, train_acc,train_f1))

            print('Val epoch:{},val_loss:{:.5f},val_acc:{:.5f},val_f1:{:.5f}'
                  .format(epoch, val_loss, val_acc, val_f1))

            self.writer.add_scalars(
                'data/loss', {'train': train_loss, 'val': val_loss}, epoch
            )
            self.writer.add_scalars(
                'data/acc', {'train': train_acc, 'val': val_acc}, epoch
            )
            self.writer.add_scalars(
                'data/f1', {'train': train_f1, 'val': val_f1}, epoch
            )
            self.writer.add_scalar(
                'data/lr', optimizer.param_groups[0]['lr'], epoch
            )

            if monitor == 'val_acc':
                early_stopping(val_acc)
                val_metric = val_acc
            elif monitor == 'val_f1':
                early_stopping(val_f1)
                val_metric = val_f1
            # if val_loss < self.loss_threshold:
            if val_metric > self.metric:
                self.loss_threshold = val_loss
                self.metric = val_metric
                
                min_loss = min(min_loss, val_loss)
                max_acc = max(max_acc, val_acc)

                if len(self.device.split(',')) > 1:
                    state_dict = net.module.state_dict()
                else:
                    state_dict = net.state_dict()

                saver = {
                    'epoch': epoch,
                    'save_dir': output_dir,
                    'state_dict': state_dict,
                    'optimizer': optimizer.state_dict()
                }

                file_name = 'epoch:{}-train_loss:{:.5f}-val_loss:{:.5f}-train_acc:{:.5f}-val_acc:{:.5f}-val_f1:{:.5f}.pth'.format(
                    epoch, train_loss, val_loss, train_acc, val_acc, val_f1)
                print('Save as --- ' + file_name)
                save_path = os.path.join(output_dir, file_name)

                torch.save(saver, save_path)
            
            #early stopping
            if early_stopping.early_stop:
                print("Early stopping")
                break

        self.writer.close()
        dfs_remove_weight(output_dir,retain=3)
        return min_loss, max_acc

    def _train_on_epoch(self, epoch, net, criterion, optimizer, train_loader,scaler):

        net.train()

        train_loss = AverageMeter()
        train_acc = AverageMeter()
        train_f1 = AverageMeter()
        for step, sample in enumerate(train_loader):

            data = sample['image']
            target = sample['label']
            
            data = data.cuda()
            target = target.cuda()
            ######## use mixup or cutmix
            prob = random.random()
            mix_p = 0.5 if not self.mix_only else 0.0
            if prob > mix_p:
                if self.use_mixup and self.use_cutmix:
                    if prob > mix_p and prob <= (1.0-(1.0-mix_p)/2):
                        data, target_a, target_b, lam = mixup_data(data, target, 1.0)
                    elif prob > (1.0-(1.0-mix_p)/2):
                        data, target_a, target_b, lam = cutmix_data(data, target, 1.0)
                    data, target_a, target_b = map(Variable, (data, target_a, target_b))
                elif not self.use_mixup and self.use_cutmix:
                    data, target_a, target_b, lam = cutmix_data(data, target, 1.0)
                    data, target_a, target_b = map(Variable, (data, target_a, target_b))
                elif not self.use_cutmix and self.use_mixup:
                    data, target_a, target_b, lam = mixup_data(data, target, 1.0)
                    data, target_a, target_b = map(Variable, (data, target_a, target_b))
            ########
            with autocast(self.use_fp16):
                output = net(data)
                if (self.use_mixup or self.use_cutmix) and prob > mix_p :
                    loss = mixup_criterion(criterion, output, target_a, target_b, lam)
                else: 
                    loss = criterion(output, target)
            
            optimizer.zero_grad()
            if self.use_fp16:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            acc = accuracy(output.data, target)[0]
            f1 = f1_score(output.data, target)
            train_loss.update(loss.item(), data.size(0))
            train_acc.update(acc.item(), data.size(0))
            train_f1.update(f1,data.size(0))
            torch.cuda.empty_cache()

            print('epoch:{},step:{},train_loss:{:.5f},train_acc:{:.5f},train_f1:{:.5f},lr:{}'
                  .format(epoch, step, loss.item(), acc.item(), f1, optimizer.param_groups[0]['lr']))

            if self.global_step % 10 == 0:
                self.writer.add_scalars(
                    'data/train_loss_acc', {'train_loss': loss.item(),
                                            'train_acc': acc.item(),
                                            'train_f1':f1}, self.global_step
                )

            self.global_step += 1

        return train_loss.avg, train_acc.avg,train_f1.avg

    def _val_on_epoch(self, epoch, net, criterion, val_path, label_dict):

        net.eval()

        val_transformer = transforms.Compose(self.transform)

        val_dataset = DataGenerator(
            val_path, label_dict, channels=self.channels, transform=val_transformer)

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

        val_loss = AverageMeter()
        val_acc = AverageMeter()
        val_f1 = AverageMeter()

        with torch.no_grad():
            for step, sample in enumerate(val_loader):
                data = sample['image']
                target = sample['label']

                data = data.cuda()
                target = target.cuda()
                with autocast(self.use_fp16):
                    output = net(data)
                    loss = criterion(output, target)

                output = output.float()
                loss = loss.float()

                # measure accuracy and record loss
                acc = accuracy(output.data, target)[0]
                f1 = f1_score(output.data, target)
                val_loss.update(loss.item(), data.size(0))
                val_acc.update(acc.item(), data.size(0))
                val_f1.update(f1,data.size(0))

                torch.cuda.empty_cache()

                print('epoch:{},step:{},val_loss:{:.5f},val_acc:{:.5f},val_f1:{:.5f}'
                      .format(epoch, step, loss.item(), acc.item(),f1))

        return val_loss.avg, val_acc.avg, val_f1.avg

    def hook_fn_forward(self, module, input, output):
        # print(module)
        # print(input[0].size())
        # print(output.size())

        for i in range(input[0].size(0)):
            self.feature_in.append(input[0][i].cpu().numpy())
            self.feature_out.append(output[i].cpu().numpy())

    def test(self, test_path, label_dict, net=None, hook_fn_forward=False):

        if net is None:
            net = self.net

        if hook_fn_forward:
            net.avgpool.register_forward_hook(self.hook_fn_forward)

        net = net.cuda()
        net.eval()

        test_transformer = transforms.Compose(self.transform)

        test_dataset = DataGenerator(
            test_path, label_dict, channels=self.channels, transform=test_transformer)

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

        result = {
            'true': [],
            'pred': [],
            'prob': []
        }

        test_acc = AverageMeter()

        with torch.no_grad():
            for step, sample in enumerate(test_loader):
                data = sample['image']
                target = sample['label']

                data = data.cuda()
                target = target.cuda()  # N
                with autocast(self.use_fp16):
                    output = net(data)
                    output = output.float()  # N*C

                acc = accuracy(output.data, target)[0]
                test_acc.update(acc.item(), data.size(0))

                result['true'].extend(target.detach().tolist())
                result['pred'].extend(torch.argmax(
                    output, 1).detach().tolist())
                output = F.softmax(output, dim=1)
                result['prob'].extend(output.detach().tolist())

                print('step:{},test_acc:{:.5f}'
                      .format(step, acc.item()))

                torch.cuda.empty_cache()

        print('average test_acc:{:.5f}'.format(test_acc.avg))

        return result, np.array(self.feature_in), np.array(self.feature_out)

    def inference(self, test_path, net=None):

        if net is None:
            net = self.net

        net = net.cuda()
        net.eval()

        test_transformer = transforms.Compose(self.transform)

        test_dataset = DataGenerator(test_path, channels=self.channels, transform=test_transformer)

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

        result = {
            'pred': [],
            'prob': []
        }

        with torch.no_grad():
            for _, sample in enumerate(test_loader):
                data = sample['image']
                data = data.cuda()
                with autocast(self.use_fp16):
                    output = net(data)
                output = output.float()  # N*C

                result['pred'].extend(torch.argmax(
                    output, 1).detach().tolist())
                output = F.softmax(output, dim=1)
                result['prob'].extend(output.detach().tolist())

                torch.cuda.empty_cache()

        return result

    def inference_tta(self, test_path, tta_times=1, net=None):

        if net is None:
            net = self.net

        net = net.cuda()
        net.eval()

        test_transformer = transforms.Compose(self.transform)
        test_dataset = DataGenerator(test_path, channels=self.channels, transform=None)

        prob_output = []
        vote_output = []

        with torch.no_grad():
            for _, sample in enumerate(test_dataset):
                data = sample['image']

                tta_output = []
                binary_output = []

                for _ in range(tta_times):
                    if self.channels == 1:
                        img = Image.fromarray(np.copy(data)).convert('L')
                    elif self.channels == 3:
                        img = Image.fromarray(np.copy(data)).convert('RGB')
                    img_tensor = test_transformer(img)
                    img_tensor = torch.unsqueeze(img_tensor, 0)

                    img_tensor = img_tensor.cuda()
                    with autocast(self.use_fp16):
                        output = net(img_tensor)
                    output = F.softmax(output, dim=1)

                    output = output.float().squeeze().cpu().numpy()
                    tta_output.append(output)

                    binary_output.append(np.argmax(output))

                prob_output.append(np.mean(tta_output, axis=0))
                vote_output.append(max(binary_output,key=binary_output.count))

        return prob_output, vote_output

    def _get_net(self, net_name):
        if net_name == 'resnet18':
            from model.resnet import resnet18
            net = resnet18(pretrained=self.external_pretrained,input_channels=self.channels,
                           num_classes=self.num_classes,final_drop=self.drop_rate)
        elif net_name == 'resnet34':
            from model.resnet import resnet34
            net = resnet34(pretrained=self.external_pretrained,input_channels=self.channels,
                           num_classes=self.num_classes,final_drop=self.drop_rate)
        elif net_name == 'resnet50':
            from model.resnet import resnet50
            net = resnet50(pretrained=self.external_pretrained,input_channels=self.channels,
                           num_classes=self.num_classes,final_drop=self.drop_rate)
        elif net_name == 'resnest18':
            from model.resnest import resnest18
            net = resnest18(pretrained=self.external_pretrained,input_channels=self.channels,
                           num_classes=self.num_classes,final_drop=self.drop_rate)
        elif net_name == 'resnest50':
            from model.resnest import resnest50
            net = resnest50(pretrained=self.external_pretrained,input_channels=self.channels,
                           num_classes=self.num_classes,final_drop=self.drop_rate)
        elif net_name == 'resnext101_32x8d':
            from model.resnet import resnext101_32x8d
            net = resnext101_32x8d(pretrained=self.external_pretrained,input_channels=self.channels,
                           num_classes=self.num_classes,final_drop=self.drop_rate)
        elif net_name == 'se_resnet18':
            from model.se_resnet import se_resnet18
            net = se_resnet18(input_channels=self.channels,
                              num_classes=self.num_classes,final_drop=self.drop_rate)
        elif net_name == 'se_resnet10':
            from model.se_resnet import se_resnet10
            net = se_resnet10(input_channels=self.channels,
                              num_classes=self.num_classes,final_drop=self.drop_rate)
        elif net_name == 'simple_net':
            from model.simple_net import simple_net
            net = simple_net(input_channels=self.channels,
                             num_classes=self.num_classes,final_drop=self.drop_rate)
        elif net_name.startswith('densenet'):
            import model.densenet as densenet
            net = densenet.__dict__[net_name](pretrained=self.external_pretrained,input_channels=self.channels,
                                            num_classes=self.num_classes,drop_rate=self.drop_rate)
        elif net_name.startswith('vgg'):
            import model.vgg as vgg
            net = vgg.__dict__[net_name](pretrained=self.external_pretrained,input_channels=self.channels,num_classes=self.num_classes)
        
        elif net_name.startswith('mod_vgg'):
            import model.vgg_mod as mod_vgg
            net = mod_vgg.__dict__[net_name](input_channels=self.channels,num_classes=self.num_classes,final_drop=self.drop_rate)
        
        elif net_name == 'res2net50':
            from model.res2net import res2net50
            net = res2net50(input_channels=self.channels,
                        	num_classes=self.num_classes,final_drop=self.drop_rate)
        elif net_name == 'res2net18':
            from model.res2net import res2net18
            net = res2net18(input_channels=self.channels,
                            num_classes=self.num_classes,final_drop=self.drop_rate)
        elif net_name == 'res2next50':
            from model.res2next import res2next50
            net = res2next50(input_channels=self.channels,
                            num_classes=self.num_classes,final_drop=self.drop_rate)
        elif net_name == 'res2next18':
            from model.res2next import res2next18
            net = res2next18(input_channels=self.channels,
                            num_classes=self.num_classes,final_drop=self.drop_rate)
        elif 'efficientnet' in net_name:
            # import timm
            # net = timm.create_model(net_name, pretrained=True, in_chans=self.channels,num_classes=self.num_classes)
            from efficientnet_pytorch import EfficientNet
            if self.external_pretrained:
                net = EfficientNet.from_pretrained(model_name=net_name,
                                                in_channels=self.channels,
                                                num_classes=self.num_classes,
                                                advprop=True)
            else:
                net = EfficientNet.from_name(model_name=net_name)
                num_ftrs = net._fc.in_features
                net._fc = nn.Linear(num_ftrs, self.num_classes)
                
        elif 'directnet' in net_name:
            import model.directnet as directnet
            net = directnet.__dict__[net_name](in_channels=self.channels,num_classes=self.num_classes)
        
        elif 'finenet' in net_name:
            import model.finenet as finenet
            net = finenet.__dict__[net_name](input_channels=self.channels,num_classes=self.num_classes)

        elif net_name.startswith('bilinearnet'):
            import model.bilinearnet as bilinearnet
            net = bilinearnet.__dict__[net_name](input_channels=self.channels,num_classes=self.num_classes)

        return net


    def _get_loss(self, loss_fun, class_weight=None):
        if class_weight is not None:
            class_weight = torch.tensor(class_weight)

        if loss_fun == 'Cross_Entropy':
            loss = nn.CrossEntropyLoss(class_weight)

        elif loss_fun == 'SoftCrossEntropy':
            from losses.crossentropy import SoftCrossEntropy
            loss = SoftCrossEntropy(classes=self.num_classes,smoothing=self.smothing,weight=class_weight)

        elif loss_fun == 'TopkCrossEntropy':
            from losses.crossentropy import SoftCrossEntropy
            loss = SoftCrossEntropy(classes=self.num_classes,smoothing=0.0,weight=class_weight,reduction='topk',k=0.8)

        elif loss_fun == 'TopkSoftCrossEntropy':
            from losses.crossentropy import SoftCrossEntropy
            loss = SoftCrossEntropy(classes=self.num_classes,smoothing=self.smothing,weight=class_weight,reduction='topk',k=0.8)

        elif loss_fun == 'DynamicTopkCrossEntropy':
            from losses.crossentropy import DynamicTopkSoftCrossEntropy
            loss = DynamicTopkSoftCrossEntropy(classes=self.num_classes,smoothing=0.0,weight=class_weight)

        elif loss_fun == 'DynamicTopkSoftCrossEntropy':
            from losses.crossentropy import DynamicTopkSoftCrossEntropy
            loss = DynamicTopkSoftCrossEntropy(classes=self.num_classes,smoothing=self.smothing,weight=class_weight)
        
        elif loss_fun == 'F1_Loss':
            from losses.f1_loss import F1_Loss
            loss = F1_Loss()
        
        return loss

    def _get_optimizer(self, optimizer, net, lr, weight_decay, momentum):
        if optimizer == 'Adam':
            optimizer = torch.optim.Adam(
                net.parameters(), lr=lr, weight_decay=weight_decay)

        elif optimizer == 'SGD':
            optimizer = torch.optim.SGD(
                net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        
        elif optimizer == 'AdamW':
            optimizer = torch.optim.AdamW(net.parameters(), lr=lr,weight_decay=weight_decay, amsgrad=False)
        return optimizer

    def _get_lr_scheduler(self, lr_scheduler, optimizer):
        if lr_scheduler == 'ReduceLROnPlateau':
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                      mode='min', patience=5, verbose=True)
        elif lr_scheduler == 'MultiStepLR':
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, self.milestones, gamma=self.gamma)
        elif lr_scheduler == 'CosineAnnealingLR':
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                            optimizer, T_max=5)
        elif lr_scheduler == 'CosineAnnealingWarmRestarts':
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                        optimizer, 5, T_mult=2)
        return lr_scheduler

    def _get_pre_trained(self, weight_path):
        checkpoint = torch.load(weight_path)
        self.net.load_state_dict(checkpoint['state_dict'])
        # self.start_epoch = checkpoint['epoch'] + 1


# computing tools

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

def f1_score(output, target):
    y_pred = torch.softmax(output, dim=1) #N*C
    y_pred = torch.argmax(y_pred,dim=1).cpu().numpy().tolist()

    y_true = target.cpu().numpy().tolist()

    f1 = metrics.f1_score(y_true,y_pred,average='macro')
    
    return f1

class EarlyStopping(object):
    """Early stops the training if performance doesn't improve after a given patience."""
    def __init__(self, patience=10, verbose=True, delta=0, monitor='val_loss',best_score=None,op_type='min'):
        """
        Args:
            patience (int): How long to wait after last time performance improved.
                            Default: 10
            verbose (bool): If True, prints a message for each performance improvement. 
                            Default: True
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            monitor (str): Monitored variable.
                            Default: 'val_loss'
            op_type (str): 'min' or 'max'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = best_score
        self.early_stop = False
        self.delta = delta
        self.monitor = monitor
        self.op_type = op_type

        if self.op_type == 'min':
            self.val_score_min = np.Inf
        else:
            self.val_score_min = 0

    def __call__(self, val_score):

        score = -val_score if self.op_type == 'min' else val_score

        if self.best_score is None:
            self.best_score = score
            self.print_and_update(val_score)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.print_and_update(val_score)
            self.counter = 0

    def print_and_update(self, val_score):
        '''print_message when validation score decrease.'''
        if self.verbose:
           print(self.monitor, f'optimized ({self.val_score_min:.6f} --> {val_score:.6f}).  Saving model ...')
        self.val_score_min = val_score


def mixup_data(x, y, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def cutmix_data(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    rand_index = torch.randperm(x.size()[0]).cuda()
    target_a = y
    target_b = y[rand_index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))

    return x,target_a,target_b,lam


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2