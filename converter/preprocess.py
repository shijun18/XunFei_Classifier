import os
from PIL import Image
import numpy as np


def preprocess(input_path,save_path):
    save_path = f'{save_path}/{os.path.basename(input_path)}'
    img = Image.open(input_path).convert('RGB')
    # print(img.size)

    w,h = img.size
    if w <= h: 
        if w < h//3:
            paste_time = 2*(h//3) // w
            new_image = Image.new(size=(paste_time*w,h),mode='RGB')
            for i in range(paste_time):
                new_image.paste(img,(i*w,0))
            img = new_image
    else:
        if h < w//3:
            paste_time = 3*(w//2) // h
            new_image = Image.new(size=(w,paste_time*h),mode='RGB')
            for i in range(paste_time):
                new_image.paste(img,(0,i*h))
            img = new_image

    img.save(save_path)


def make_data(input_path,save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for entry in os.scandir(input_path):
        print(entry.path)
        if entry.is_dir():
            tmp_save_path = os.path.join(save_path,entry.name)
            make_data(entry.path,tmp_save_path)
        else:
            preprocess(entry.path,save_path)



if __name__ == '__main__':
    input_path = '../dataset/Adver_Material/pre_train'
    save_path = '../dataset/Adver_Material/train'
    make_data(input_path,save_path)

    input_path = '../dataset/Adver_Material/pre_test'
    save_path = '../dataset/Adver_Material/test'
    make_data(input_path,save_path)