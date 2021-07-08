 
__all__ = ['resnet18','resnet34', 'resnet50','resnest18','resnest50','se_resnet18', 'se_resnet10', \
            'simple_net', 'tiny_net','densenet121','vgg16','res2net50','res2net18','res2next50', \
            'res2next18','efficientnet-b5']


data_config = {
    'Adver_Material':'./converter/csv_file/adver_material.csv',
    'Crop_Growth':'./converter/csv_file/crop_growth.csv',
    'Photo_Guide':'./converter/csv_file/photo_guide.csv',
}

num_classes = {
    'Adver_Material':137,
    'Crop_Growth':4,
    'Photo_Guide':11
}

TASK = 'Adver_Material'
NET_NAME = 'resnet50'
VERSION = 'v3.0-pretrained'
DEVICE = '3'
# Must be True when pre-training and inference
PRE_TRAINED = False 
# 1,2,3,4
CURRENT_FOLD = 4
GPU_NUM = len(DEVICE.split(','))
FOLD_NUM = 5
TTA_TIMES = 1


NUM_CLASSES = num_classes[TASK]
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
    'Crop_Growth':None,
    'Photo_Guide':None
}

STD = {
    'Adver_Material':None,
    'Crop_Growth':None,
    'Photo_Guide':None
}

MILESTONES = {
    'Adver_Material':[30,60,90],
    'Crop_Growth':[30,60,90],
    'Photo_Guide':[30,60,90]
}

EPOCH = {
    'Adver_Material':150,
    'Crop_Growth':120,
    'Photo_Guide':120
}

TRANSFORM = {
    'Adver_Material':[6,7,8,13,9],
    'Crop_Growth':[6,7,8,13,9],
    'Photo_Guide':[6,7,8,13,9]
}

# Arguments when trainer initial
INIT_TRAINER = {
    'net_name': NET_NAME,
    'lr': 1e-3 if not PRE_TRAINED else 5e-5, #1e-3
    'n_epoch': EPOCH[TASK],
    'channels': 3,
    'num_classes': NUM_CLASSES,
    'input_shape': (256, 256),
    'crop': 0,
    'batch_size': 32,
    'num_workers': 2,
    'device': DEVICE,
    'pre_trained': PRE_TRAINED,
    'weight_path': WEIGHT_PATH,
    'weight_decay': 0.0001, #0.0001
    'momentum': 0.9,
    'mean': MEAN[TASK],
    'std': STD[TASK],
    'gamma': 0.1,
    'milestones': MILESTONES[TASK],
    'use_fp16':True,
    'transform':TRANSFORM[TASK],
    'drop_rate': 0.5, #0.5
    'external_pretrained':True if 'pretrained' in VERSION else False#False
}

# no_crop     

# mean:0.131
# std:0.209

#crop
# mean:0.189
# std:0.185


# Arguments when perform the trainer
__loss__ = ['Cross_Entropy','TopkCrossEntropy','SoftCrossEntropy','TopkSoftCrossEntropy']
LOSS_FUN = 'Cross_Entropy'
# Arguments when perform the trainer
SETUP_TRAINER = {
    'output_dir': './ckpt/{}/{}'.format(TASK,VERSION),
    'log_dir': './log/{}/{}'.format(TASK,VERSION),
    'optimizer': 'Adam',
    'loss_fun': LOSS_FUN,
    'class_weight': None,
    'lr_scheduler': 'MultiStepLR' #'MultiStepLR'
}
