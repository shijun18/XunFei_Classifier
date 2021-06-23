import os
import numpy as np
from PIL import Image


def cal_mean_std(data_path):
    r_image = []
    g_image = []
    b_image = []
    count = 0
    for entry in os.scandir(data_path): 
        for item in os.scandir(entry.path):
            img = Image.open(item.path).convert('RGB')
            # print(np.array(img).shape) 
            r_img = (np.array(img)[...,0].astype(np.float32)/255.0).flatten()
            g_img = (np.array(img)[...,1].astype(np.float32)/255.0).flatten()
            b_img = (np.array(img)[...,2].astype(np.float32)/255.0).flatten()
            r_image.extend(r_img)
            g_image.extend(g_img)
            b_image.extend(b_img)
            count += 1
    print('data len: %d' % count)
    print('R mean:%.3f' % np.mean(r_image))
    print('G mean:%.3f' % np.mean(g_image))
    print('B mean:%.3f' % np.mean(b_image))
    print('R std:%.3f' % np.std(r_image))
    print('G std:%.3f' % np.std(g_image))
    print('B std:%.3f' % np.std(b_image))

  
if __name__ == '__main__':
    data_path = '../dataset/Adver_Material/train/'
    cal_mean_std(data_path)
