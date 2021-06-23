import os,h5py
import numpy as np
from PIL import Image

def save_as_hdf5(data,save_path,key=None):
  '''
  Numpy array save as hdf5.

  Args:
  - data: numpy array
  - save_path: string, destination path
  - key: string, key value for reading
  '''
  hdf5_file = h5py.File(save_path, 'a')
  hdf5_file.create_dataset(key, data=data)
  hdf5_file.close()


def hdf5_reader(data_path,key=None):
  '''
  Hdf5 file reader, return numpy array.
  '''
  hdf5_file = h5py.File(data_path,'r')
  image = np.asarray(hdf5_file[key],dtype=np.float32)
  hdf5_file.close()

  return image


def cal_mean_std(data_path):
    image = []
    count = 0
    for entry in os.scandir(data_path): 
        for subdir in os.scandir(entry.path): 
            for item in os.scandir(subdir.path):
                img = Image.open(item.path).convert('L')
                img = (np.array(img).astype(np.float32)/255.0).flatten()
                image.extend(img)
                count += 1
    print('data len: %d' % count)
    print('mean:%.3f' % np.mean(image))
    print('std:%.3f' % np.std(image))

  
if __name__ == '__main__':
    data_path = '../dataset/pre_data/train/'
    cal_mean_std(data_path)

    data_path = '../dataset/pre_crop_data/train/'
    cal_mean_std(data_path)

