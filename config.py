 
__all__ = ['resnet18','resnet34', 'resnet50','resnest18','resnest50','efficientnet-b5'\
            'efficientnet-b7','efficientnet-b3','bilinearnet18','finenet50','directnet50','densenet121','densenet169','simple_net','vgg16','res2net50','res2net18','res2next50', \
            'res2next18','se_resnet18', 'se_resnet10', ]


data_config = {
    'Adver_Material':'./converter/csv_file/adver_material.csv',
    'Crop_Growth':'./converter/csv_file/crop_growth.csv',
    'Photo_Guide':'./converter/csv_file/photo_guide_merge.csv',
    'Leve_Disease':'./converter/csv_file/leve_disease.csv',
    'Temp_Freq':'./converter/csv_file/temp_freq.csv',
    'Farmer_Work':'./converter/csv_file/farmer_work_crop_external.csv'
}

num_classes = {
    'Adver_Material':137,
    'Crop_Growth':4,
    'Photo_Guide':8,
    'Leve_Disease':3,
    'Temp_Freq':24,
    'Farmer_Work':4
}

TASK = 'Farmer_Work'
NET_NAME = 'efficientnet-b5'
VERSION = 'v6.0-crop-external-pretrained'
DEVICE = '4,5'
# Must be True when pre-training and inference
PRE_TRAINED = False 
# 1,2,3,4
CURRENT_FOLD = 1
GPU_NUM = len(DEVICE.split(','))
FOLD_NUM = 5
TTA_TIMES = 5


NUM_CLASSES = num_classes[TASK]
from utils import get_weight_path,get_weight_list

CSV_PATH = data_config[TASK]
CKPT_PATH = './ckpt/{}/{}/fold{}'.format(TASK,VERSION,str(CURRENT_FOLD))
# CKPT_PATH = './ckpt/{}/{}/fold{}'.format(TASK,'v6.0-pretrained',str(CURRENT_FOLD))
WEIGHT_PATH = get_weight_path(CKPT_PATH)
print(WEIGHT_PATH)

if PRE_TRAINED:
    WEIGHT_PATH_LIST = get_weight_list('./ckpt/{}/{}/'.format(TASK,VERSION))
else:
    WEIGHT_PATH_LIST = None

# WEIGHT_PATH_LIST = None

MEAN = {
    'Adver_Material':[0.485, 0.456, 0.406], 
    'Crop_Growth':None,
    'Photo_Guide':(0.450),
    'Leve_Disease':(0.527),#(0.496,0.527,0.387),
    'Temp_Freq':None,
    'Farmer_Work':(0.486,0.496,0.403)
}

STD = {
    'Adver_Material':[0.229, 0.224, 0.225],
    'Crop_Growth':None,
    'Photo_Guide':(0.224),
    'Leve_Disease':(0.216),#(0.230,0.216,0.237),
    'Temp_Freq':None,
    'Farmer_Work':(0.237,0.231,0.246)
    
}

MILESTONES = {
    'Adver_Material':[30,60,90],
    'Crop_Growth':[30,60,90],
    'Photo_Guide':[30,60,90],
    'Leve_Disease':[30,60,90],
    'Temp_Freq':[30,60,90],
    'Farmer_Work':[30,60,90]
}

EPOCH = {
    'Adver_Material':150,
    'Crop_Growth':120,
    'Photo_Guide':200, #120
    'Leve_Disease':120,
    'Temp_Freq':120,
    'Farmer_Work':50
}

TRANSFORM = {
    'Adver_Material':[2,6,7,8,9],#[6,7,8,2,9]
    'Crop_Growth':[6,7,8,13,9],
    'Photo_Guide':[18,2,6,9,19],#[18,2,6,9,19]
    'Leve_Disease':[2,6,7,8,9,10,19], #[2,6,7,8,9,10,19] (6.2-pre)(8.0-pre)(7.2-pre) 
    'Temp_Freq':[2,4,9,19], #[2,3,4,9,19](6.0-pre) [2,4,9,19](6.1-pre) 
    'Farmer_Work':[2,6,7,8,9,10,19]
}

SHAPE = {
    'Adver_Material':(512, 512),
    'Crop_Growth':(256, 256),
    'Photo_Guide':(256, 256),
    'Leve_Disease':(512, 512),
    'Temp_Freq':(512,512),
    'Farmer_Work':(512,512)
}


CHANNEL = {
    'Adver_Material':3,
    'Crop_Growth':3,
    'Photo_Guide':1,
    'Leve_Disease':3,
    'Temp_Freq':3,
    'Farmer_Work':3
}

# Arguments when trainer initial
INIT_TRAINER = {
    'net_name': NET_NAME,
    'lr': 1e-3 if not PRE_TRAINED else 5e-4, #1e-3
    'n_epoch': EPOCH[TASK],
    'channels': CHANNEL[TASK],
    'num_classes': NUM_CLASSES,
    'input_shape': SHAPE[TASK],
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
    'smothing':0.15,
    'external_pretrained':True if 'pretrained' in VERSION else False#False
}

# no_crop     

# mean:0.131
# std:0.209

#crop
# mean:0.189
# std:0.185


# Arguments when perform the trainer
__loss__ = ['Cross_Entropy','TopkCrossEntropy','SoftCrossEntropy','TopkSoftCrossEntropy','DynamicTopkCrossEntropy','DynamicTopkSoftCrossEntropy']
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
