import os
import random

import numpy as np
from PIL import Image

import torch
import torchvision.transforms as transforms
import torch.utils.data as data
 
DEFAULT_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Use transforms.RandomHorizontalFlip

class RandomTranslate(object):
    """Randomly translate the image in a sample (uses wraparound)."""

    def __init__(self, maxHShift : float, maxVShift : float):
        """
        Args:
            maxHShift (float): Maximum horizontal shift (as a fraction of the full image width). 
            maxVShift (float): Maximum vertical shift (as a fraction of the full image height).
        """

        self.maxHShift = maxHShift
        self.maxVShift = maxVShift

    def __call__(self, image):

        image_arr = np.array(image)

        maxShiftPixelsH = int(image_arr.shape[1] * self.maxHShift)
        maxShiftPixelsV = int(image_arr.shape[0] * self.maxVShift)

        image = np.roll(image_arr, random.randint(-maxShiftPixelsH, maxShiftPixelsH), axis=1)
        image = np.roll(image_arr, random.randint(-maxShiftPixelsV, maxShiftPixelsV), axis=0)

        return Image.fromarray(image_arr)    

class SketchDataset(data.Dataset):
    """Simple class for storing a sketch dataset."""

    def __init__(self, root_dir : str, use_augmentation : bool, size_override = None, device = DEFAULT_DEVICE):
        """
        Arguments:
            
            root_dir (string): 


        """

        """
        Parameters
        -----------
        root_dir : str
            Directory with all the images (must be in png format).
        use_augmentation : bool
            Whether to use include data augmentation in the dataset. Currently supported 
            augmentations are horizontal flips and translations.
        size_override : int
            If not None, restricts the dataset to the first 'size_override' entries.
            if there are fewer than 'size_override' images in the provided directory, 
            all images are used. If 'size_override' is negative, it is ignored.
        device : device
            The device onto which the dataset should be loaded. By default, the dataset is 
            stored on the GPU if cuda is available, and on the CPU otherwise.
        """

        if root_dir.endswith('/') or root_dir.endswith('\\'):
            root_dir = root_dir[:-1]
            
        self.root_dir = root_dir
        self.size_override = size_override
        self.device = device
        if use_augmentation:
            self.transform = transforms.Compose([RandomTranslate(0.50, 0.50), transforms.RandomHorizontalFlip(0.5), transforms.ToTensor(), transforms.Grayscale(1)])
        else:
            self.transform = transforms.Compose([transforms.ToTensor(), transforms.Grayscale(1)])

        self.count = 0
        self.names = []
        print("Searching " + self.root_dir + " for files...")
        for path in os.scandir(self.root_dir):
            filestring = path.path[len(self.root_dir) + 1:]
            if path.is_file() and filestring.endswith('.png'):
                self.count += 1
                self.names.append(filestring)

        if self.size_override is not None and self.size_override > -1:
            self.count = min(self.size_override, self.count)

    def __len__(self):
        return self.count
        
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()[0]
        
        img_name = os.path.join(self.root_dir, self.names[index])

        image = self.transform(Image.open(img_name).convert('RGB')).to(self.device)

        return image
    
