import os
import argparse
from trainer import My_Classifier
import pandas as pd
from data_utils.csv_reader import csv_reader_single
from config import INIT_TRAINER, SETUP_TRAINER,TASK,NUM_CLASSES
from config import VERSION, CURRENT_FOLD, FOLD_NUM, WEIGHT_PATH_LIST, TTA_TIMES, CSV_PATH

import time
import numpy as np
import random

KEY = {
    'Adver_Material':['image_id','category_id'],
    'Crop_Growth':['image_id','category_id'],
    'Photo_Guide':['image','label'],
    'Leve_Disease':['image_id','category_id'],
    'Temp_Freq':['image_id','category_id'],
    'Farmer_Work':['image_id','category_id']
}

ADD_FACTOR = {
    'Adver_Material':0,
    'Crop_Growth':0,
    'Photo_Guide':1,
    'Leve_Disease':1,
    'Temp_Freq':0,
    'Farmer_Work':0
}

def get_cross_validation(path_list, fold_num, current_fold):

    _len_ = len(path_list) // fold_num

    train_id = []
    validation_id = []
    end_index = current_fold * _len_
    start_index = end_index - _len_
    if current_fold == fold_num:
        validation_id.extend(path_list[start_index:])
        train_id.extend(path_list[:start_index])
    else:
        validation_id.extend(path_list[start_index:end_index])
        train_id.extend(path_list[:start_index])
        train_id.extend(path_list[end_index:])

    print("Train set length:", len(train_id),
          "Val set length:", len(validation_id))
    return train_id, validation_id


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', default='train-cross', choices=["train-cross", "inf-cross", "train", "inf"],
                        help='choose the mode', type=str)
    parser.add_argument('-s', '--save', default='no', choices=['no', 'n', 'yes', 'y'],
                        help='save the forward middle features or not', type=str)
    parser.add_argument('-p', '--path', default='/staff/shijun/torch_projects/XunFei_Classifier/dataset',
                        help='the directory path of input image', type=str)
    args = parser.parse_args()
    
    label_dict = {}
    # Set data path & classifier
    
    pre_csv_path = CSV_PATH
    pre_label_dict = csv_reader_single(pre_csv_path, key_col='id', value_col='label')
    label_dict.update(pre_label_dict)

    
    if args.mode != 'train-cross' and args.mode != 'inf-cross':
        classifier = My_Classifier(**INIT_TRAINER)
        print(get_parameter_number(classifier.net))

    # Training with cross validation
    ###############################################
    if args.mode == 'train-cross':
        path_list = list(label_dict.keys())
        print("dataset length is %d"%len(path_list))

        loss_list = []
        acc_list = []

        for current_fold in range(1, FOLD_NUM+1):
            print("=== Training Fold ", current_fold, " ===")
            classifier = My_Classifier(**INIT_TRAINER)

            if current_fold == 0:
                print(get_parameter_number(classifier.net))

            train_path, val_path = get_cross_validation(
                path_list, FOLD_NUM, current_fold)
            # train_path, val_path = get_cross_val_by_class(
            #     path_list, FOLD_NUM, current_fold) # split by class
            SETUP_TRAINER['train_path'] = train_path
            SETUP_TRAINER['val_path'] = val_path
            SETUP_TRAINER['label_dict'] = label_dict
            SETUP_TRAINER['cur_fold'] = current_fold

            start_time = time.time()
            val_loss, val_acc = classifier.trainer(**SETUP_TRAINER)
            loss_list.append(val_loss)
            acc_list.append(val_acc)

            print('run time:%.4f' % (time.time()-start_time))

        print("Average loss is %f, average acc is %f" %
              (np.mean(loss_list), np.mean(acc_list)))
    ###############################################

    # Training
    ###############################################
    elif args.mode == 'train':
        path_list = list(label_dict.keys())
        print("dataset length is %d"%len(path_list))

        train_path, val_path = get_cross_validation(
            path_list, FOLD_NUM, CURRENT_FOLD)
        SETUP_TRAINER['train_path'] = train_path
        SETUP_TRAINER['val_path'] = val_path
        SETUP_TRAINER['label_dict'] = label_dict
        SETUP_TRAINER['cur_fold'] = CURRENT_FOLD

        start_time = time.time()
        classifier.trainer(**SETUP_TRAINER)

        print('run time:%.4f' % (time.time()-start_time))
    ###############################################

    # Inference
    ###############################################
    elif 'inf' in args.mode:
        add_factor = ADD_FACTOR[TASK]
        save_dir = './analysis/result/{}/{}'.format(TASK,VERSION)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if args.mode == 'inf':
            test_id = os.listdir(args.path)
            # test_id.sort(key=lambda x:eval(x.split('.')[0].split('_')[-1]))
            test_id.sort(key=lambda x:x.split('.')[0])
            test_path = [os.path.join(args.path, case)
                        for case in test_id]
            save_path = os.path.join(save_dir,f'submission_fold{CURRENT_FOLD}.csv')

            start_time = time.time()

            result = classifier.inference(test_path)
            print('run time:%.4f' % (time.time()-start_time))

            info = {}
            info[KEY[TASK][0]] = [os.path.basename(case) for case in test_path]
            info[KEY[TASK][1]] = [int(case) + add_factor for case in result['pred']]
            for i in range(NUM_CLASSES):
                info[f'prob_{str(i+1)}'] = np.array(result['prob'])[:,i].tolist()

            # info['prob'] = result['prob']
            csv_file = pd.DataFrame(info)
            csv_file.to_csv(save_path, index=False)
        ###############################################

        # Inference with cross validation
        ###############################################
        elif args.mode == 'inf-cross':
            test_id = os.listdir(args.path)
            # test_id.sort(key=lambda x:eval(x.split('.')[0].split('_')[-1]))
            test_id.sort(key=lambda x:x.split('.')[0])
            test_path = [os.path.join(args.path, case)
                        for case in test_id]
            save_path_vote = os.path.join(save_dir,'submission_vote.csv')
            save_path = os.path.join(save_dir,'submission_ave.csv')

            result = {
                'pred': [],
                'vote_pred': [],
                'prob': []
            }

            all_prob_output = []
            all_vote_output = []

            start_time = time.time()
            for i, weight_path in enumerate(WEIGHT_PATH_LIST):
                print("Inference %d fold..." % (i+1))
                print("weight: %s"%weight_path)
                INIT_TRAINER['weight_path'] = weight_path
                classifier = My_Classifier(**INIT_TRAINER)

                prob_output, vote_output = classifier.inference_tta(
                    test_path, TTA_TIMES)
                all_prob_output.append(prob_output)
                all_vote_output.append(vote_output)

            avg_output = np.mean(all_prob_output, axis=0)
            result['prob'].extend(avg_output.tolist())

            result['pred'].extend(np.argmax(avg_output, 1).tolist())
            vote_array = np.asarray(all_vote_output).astype(int)
            result['vote_pred'].extend([max(list(vote_array[:,i]),key=list(vote_array[:,i]).count) for i in range(vote_array.shape[1])])

            print('run time:%.4f' % (time.time()-start_time))

            info = {}
            info[KEY[TASK][0]] = [os.path.basename(case) for case in test_path]
            info[KEY[TASK][1]] = [int(case) + add_factor for case in result['pred']]
            for i in range(NUM_CLASSES):
                info[f'prob_{str(i+1)}'] = np.array(result['prob'])[:,i].tolist()
            # info['prob'] = result['prob']
            csv_file = pd.DataFrame(info)
            csv_file.to_csv(save_path, index=False)

            info = {}
            info[KEY[TASK][0]] = [os.path.basename(case) for case in test_path]
            info[KEY[TASK][1]] = [int(case) + add_factor for case in result['vote_pred']]
            for i in range(NUM_CLASSES):
                info[f'prob_{str(i+1)}'] = np.array(result['prob'])[:,i].tolist()
            # info['prob'] = result['prob']
            csv_file = pd.DataFrame(info)
            csv_file.to_csv(save_path_vote, index=False)
        ###############################################
