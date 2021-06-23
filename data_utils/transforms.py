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
        image = random_noise(np.array(image),mode='s&p') 
        image = Image.fromarray(image*255)
        return image