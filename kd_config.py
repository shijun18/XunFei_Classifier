 
__all__ = ['resnet18','resnet34', 'resnet50','resnest18','resnest50','efficientnet-b5'\
            'vgg16_bn','mod_vgg16','bilinearnet18','finenet50','directnet50','densenet121','densenet169','simple_net', 'tiny_net','vgg16','res2net50','res2net18','res2next50', \
            'res2next18','se_resnet18', 'se_resnet10', ]


data_config = {
    'Adver_Material':'./converter/csv_file/adver_material.csv',
    'Crop_Growth':'./converter/csv_file/crop_growth.csv',
    'Photo_Guide':'./converter/csv_file/photo_guide_merge.csv',
    'Leve_Disease':'./converter/csv_file/leve_disease.csv',
    'Temp_Freq':'./converter/csv_file/temp_freq.csv'
}

num_classes = {
    'Adver_Material':137,
    'Crop_Growth':4,
    'Photo_Guide':8,
    'Leve_Disease':3,
    'Temp_Freq':24
}

TASK = 'Temp_Freq'
NET_NAME = 'resnet18'
VERSION = 'v1.0'

TEA_NET_NAME = 'efficientnet-b5'
TEACHER_VERSION = 'v6.0-pretrained'

DEVICE = '0'
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
CKPT_PATH = './kd_ckpt/{}/{}/fold{}'.format(TASK,VERSION,str(CURRENT_FOLD))
TEACHER_CKPT_PATH = './ckpt/{}/{}/fold{}'.format(TASK,TEACHER_VERSION,str(CURRENT_FOLD))

WEIGHT_PATH = get_weight_path(CKPT_PATH)
TEACHER_WEIGHT_PATH  = get_weight_path(TEACHER_CKPT_PATH)
TEACHER_WEIGHT_PATH_LIST = get_weight_list('./ckpt/{}/{}/'.format(TASK,TEACHER_VERSION))
print(WEIGHT_PATH)

if PRE_TRAINED:
    WEIGHT_PATH_LIST = get_weight_list('./kd_ckpt/{}/{}/'.format(TASK,VERSION))
else:
    WEIGHT_PATH_LIST = None



MEAN = {
    'Adver_Material':[0.485, 0.456, 0.406], 
    'Crop_Growth':None,
    'Photo_Guide':(0.450),
    'Leve_Disease':(0.496,0.527,0.387),
    'Temp_Freq':None
}

STD = {
    'Adver_Material':[0.229, 0.224, 0.225],
    'Crop_Growth':None,
    'Photo_Guide':(0.224),
    'Leve_Disease':(0.230,0.216,0.237),
    'Temp_Freq':None
    
}

MILESTONES = {
    'Adver_Material':[30,60,90],
    'Crop_Growth':[30,60,90],
    'Photo_Guide':[30,60,90],
    'Leve_Disease':[30,60,90],
    'Temp_Freq':[30,60,90]
}

EPOCH = {
    'Adver_Material':150,
    'Crop_Growth':120,
    'Photo_Guide':200, #120
    'Leve_Disease':120,
    'Temp_Freq':120
}

TRANSFORM = {
    'Adver_Material':[2,6,7,8,9],#[6,7,8,2,9]
    'Crop_Growth':[6,7,8,13,9],
    'Photo_Guide':[18,2,6,9,19],#[18,2,6,9,19]
    'Leve_Disease':[2,6,7,8,9,10,19],
    'Temp_Freq':[2,4,9,19] #[2,3,4,9,19]
}

SHAPE = {
    'Adver_Material':(512, 512),
    'Crop_Growth':(256, 256),
    'Photo_Guide':(256, 256),
    'Leve_Disease':(512, 512),
    'Temp_Freq':(512,512)
}


CHANNEL = {
    'Adver_Material':3,
    'Crop_Growth':3,
    'Photo_Guide':1,
    'Leve_Disease':3,
    'Temp_Freq':3
}

# Arguments when trainer initial
INIT_TRAINER = {
    'net_name':NET_NAME,
    'teacher_net_name':TEA_NET_NAME,
    'lr': 1e-3 if not PRE_TRAINED else 5e-5, #1e-3
    'n_epoch': EPOCH[TASK],
    'channels': CHANNEL[TASK],
    'num_classes': NUM_CLASSES,
    'input_shape': SHAPE[TASK],
    'crop': 0,
    'batch_size': 24,
    'num_workers': 2,
    'device': DEVICE,
    'pre_trained': PRE_TRAINED,
    'weight_path': WEIGHT_PATH,
    'teacher_weight_path': TEACHER_WEIGHT_PATH,
    'weight_decay': 0.0001, #0.0001
    'momentum': 0.9,
    'mean': MEAN[TASK],
    'std': STD[TASK],
    'gamma': 0.1,
    'milestones': MILESTONES[TASK],
    'use_fp16':True,
    'transform':TRANSFORM[TASK],
    'drop_rate': 0.5, #0.5
    'external_pretrained':True if 'pretrained' in VERSION else False,#False
    'alpha':0.95,
    'temperature':6
}

# no_crop     

# mean:0.131
# std:0.209

#crop
# mean:0.189
# std:0.185


# Arguments when perform the trainer
__loss__ = ['KD_Loss']
LOSS_FUN = 'KD_Loss'
# Arguments when perform the trainer
SETUP_TRAINER = {
    'output_dir': './kd_ckpt/{}/{}'.format(TASK,VERSION),
    'log_dir': './kd_log/{}/{}'.format(TASK,VERSION),
    'optimizer': 'Adam',
    'loss_fun': LOSS_FUN,
    'class_weight': None,
    'lr_scheduler': 'MultiStepLR' #'MultiStepLR'
}
