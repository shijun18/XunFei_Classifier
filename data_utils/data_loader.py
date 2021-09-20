import sys
sys.path.append('..')
from PIL import Image
from torch.utils.data import Dataset
import torch
import random

class DataGenerator(Dataset):
  '''
  Custom Dataset class for data loader.
  Args：
  - path_list: list of file path
  - label_dict: dict, file path as key, label as value
  - transform: the data augmentation methods
  '''
  def __init__(self, path_list, label_dict=None, channels=1, transform=None, repeat_factor=1.0):

      self.path_list = path_list
      self.label_dict = label_dict
      self.transform = transform
      self.channels = channels  
      self.repeat_factor = repeat_factor

  def __len__(self):
      return int(len(self.path_list)*self.repeat_factor)


  def __getitem__(self,index):
      # Get image and label
      # image: D,H,W
      # label: integer, 0,1,..
      index = index % len(self.path_list)
      if self.channels == 3:
          image = Image.open(self.path_list[index]).convert('RGB')
      else:
          image = Image.open(self.path_list[index]).convert('L')
      if self.transform is not None:
          image = self.transform(image)

      if self.label_dict is not None:
          label = self.label_dict[self.path_list[index]]    
          # Transform
          sample = {'image':image, 'label':int(label)} #01/23 -> >1
      else:
          sample = {'image':image}
      
      return sample



def split_image(image):
    """
    image: PIL Image
    """
    w,h = image.size
    split_list = []
    spilt_w,split_h = w//3,h//3

    for i in range(3):
        w_p = i*spilt_w
        for j in range(3):
            h_p = j*split_h
            split_list.append(image.crop((w_p,h_p,w_p+spilt_w,h_p+split_h)))
    split_list.append(image)

    return split_list

class SplitDataGenerator(Dataset):


  '''
  Custom Dataset class for data loader.
  Args：
  - path_list: list of file path
  - label_dict: dict, file path as key, label as value
  - transform: the data augmentation methods
  '''
  def __init__(self, path_list, label_dict=None, channels=1, transform=None):

    self.path_list = path_list
    self.label_dict = label_dict
    self.transform = transform
    self.channels = channels


  def __len__(self):
    return len(self.path_list)


  def __getitem__(self,index):
      # Get image and label
      # image: D,H,W
      # label: integer, 0,1,..

      assert self.channels == 1
      image = Image.open(self.path_list[index]).convert('L')

      image_list = split_image(image)
      if self.transform is not None:
          new_image = []
          for img in image_list:
              new_image.append(self.transform(img))

      image = torch.cat(new_image,0)

      if self.label_dict is not None:
          label = self.label_dict[self.path_list[index]]    
          # Transform
          sample = {'image':image, 'label':int(label)}
      else:
          sample = {'image':image}
        
      return sample


class BalanceDataGenerator(Dataset):
  '''
  Custom Dataset class for data loader.
  Args：
  - path_list: list of file path
  - label_dict: dict, file path as key, label as value
  - transform: the data augmentation methods
  '''
  def __init__(self, path_list, label_dict=None, channels=1, transform=None,factor=0.5):

      self.path_list = path_list
      self.label_dict = label_dict
      self.transform = transform
      self.channels = channels
      self.factor = factor
      

  def __len__(self):
      assert isinstance(self.path_list[0],list)
      assert len(self.path_list) == 2
      return sum([len(case) for case in self.path_list])


  def __getitem__(self,index):
      # Get image and label
      # image: D,H,W
      # label: integer, 0,1,..
      item_path = random.choice(self.path_list[int(random.random() < self.factor)])
      if self.channels == 3:
          image = Image.open(item_path).convert('RGB')
      else:
          image = Image.open(item_path).convert('L')
          # image = Image.open(self.path_list[index]).convert('RGB').split()[1]
      if self.transform is not None:
          image = self.transform(image)

      if self.label_dict is not None:
          label = self.label_dict[item_path]    
          # Transform
          sample = {'image':image, 'label':int(label)}
      else:
          sample = {'image':image}
      
      return sample