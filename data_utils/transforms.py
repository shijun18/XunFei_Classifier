import numpy as np
from skimage.util import random_noise

from PIL import Image, ImageFilter
import torchvision.transforms.functional as TF
import random

class RandomRotate(object):

    def __init__(self, angels):
        self.angels = angels

    def __call__(self, image):

        angle = random.choice(self.angels)
        image = TF.rotate(image, angle)
        return image

class DeNoise(object):

    def __init__(self, kernel_size):
        self.kernel_size = kernel_size

    def __call__(self, image):
        image = image.filter(ImageFilter.MedianFilter(self.kernel_size))
        return image


class AddNoise(object):

    def __call__(self, image):
        if image.mode == 'RGB':
            image = random_noise(np.array(image)[...,0],mode='s&p') 
            image = Image.fromarray((image*255).astype(np.uint8)).convert('RGB')
        else:
            image = random_noise(np.array(image),mode='s&p') 
            image = Image.fromarray((image*255).astype(np.uint8)).convert('L')
        return image


class SquarePad(object):
    
    def __call__(self, image):
        H = W = max(image.size)
        image_array = np.asarray(image)
        h,w = image_array.shape[:2]
        off_h = (H-h) // 2 # >= 0
        off_w = (W-w) // 2 # >= 0
        
        if h != H or w != W:
            if image.mode == 'RGB':
                image_array = np.pad(image_array,((off_h,off_h),(off_w,off_w),(0,0)),'constant')
            else:
                image_array = np.pad(image_array,((off_h,off_h),(off_w,off_w)),'constant')
            new_img = Image.fromarray(image_array)
        else:
            new_img = image

        return new_img


class Trimming(object):
    
    def __call__(self, image):
        w,h = image.size
        factor = random.choice(range(1,5))*0.1
        if w >= 5*h:
            crop = (factor*w,0,(0.2+factor)*w,h)
        elif h >= 5*w:
            crop = (0,factor*h,w,(0.2+factor)*h)
        else:
            crop = (0,0,w,h)
        new_img = image.crop(crop)

        return new_img

        