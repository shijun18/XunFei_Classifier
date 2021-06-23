 
__all__ = ['resnet18','resnet34', 'resnet50','resnest18','resnest50','se_resnet18', 'se_resnet10', \
            'simple_net', 'tiny_net','densenet121','vgg16','res2net50','res2net18','res2next50', \
            'res2next18','efficientnet-b5']


data_config = {
    'Adver_Material':'./converter/csv_file/adver_material.csv',
}

num_classes = {
    'Adver_Material':137
}

TASK = 'Adver_Material'
NET_NAME = 'resnest50'
VERSION = 'v5.0'
DEVICE = '7'
# Must be True when pre-training and inference
PRE_TRAINED = False 
# 1,2,3,4
CURRENT_FOLD = 1
GPU_NUM = len(DEVICE.split(','))
FOLD_NUM = 5
TTA_TIMES = 18


NUM_CLASSES = num_classes[TASK]
from PIL.Image import NONE
from utils import get_weight_path,get_weight_list

CSV_PATH = data_config[TASK]
CKPT_PATH = './ckpt/{}/{}/fold{}'.format(TASK,VERSION,str(CURRENT_FOLD))
WEIGHT_PATH = get_weight_path(CKPT_PATH)
print(WEIGHT_PATH)

if PRE_TRAINED:
    WEIGHT_PATH_LIST = get_weight_list('./ckpt/{}/{}/'.format(TASK,VERSION))
else:
    WEIGHT_PATH_LIST = None



MEAN = {
    'Adver_Material':None,
}

STD = {
    'Adver_Material':None,
}

# Arguments when trainer initial
INIT_TRAINER = {
    'net_name': NET_NAME,
    'lr': 1e-3,
    'n_epoch': 150,
    'channels': 3,
    'num_classes': NUM_CLASSES,
    'input_shape': (128, 128),
    'crop': 0,
    'batch_size': 64,
    'num_workers': 2,
    'device': DEVICE,
    'pre_trained': PRE_TRAINED,
    'weight_path': WEIGHT_PATH,
    'weight_decay': 0,
    'momentum': 0.9,
    'mean': MEAN[TASK],
    'std': STD[TASK],
    'gamma': 0.1,
    'milestones': [30,60,90],
    'use_fp16':True
}

# no_crop     

# mean:0.131
# std:0.209

#crop
# mean:0.189
# std:0.185


# Arguments when perform the trainer
SETUP_TRAINER = {
    'output_dir': './ckpt/{}/{}'.format(TASK,VERSION),
    'log_dir': './log/{}/{}'.format(TASK,VERSION),
    'optimizer': 'Adam',
    'loss_fun': 'Cross_Entropy',
    'class_weight': None,
    'lr_scheduler': 'MultiStepLR'
}
