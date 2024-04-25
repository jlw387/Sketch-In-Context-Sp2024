import os
import random
import pickle

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torchvision.transforms as transforms
import torch.utils.data as data
from tqdm import tqdm
 
DEFAULT_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_image(img_path, device = DEFAULT_DEVICE) -> torch.Tensor:
    transform = transforms.Compose([transforms.ToTensor(), transforms.Grayscale(1)])
    
    return transform(Image.open(img_path).convert('RGB')).to(device)

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

        pickled_data_path = root_dir + "/" + "stored_data.pkl"

        if os.path.isfile(pickled_data_path):
            with open(pickled_data_path, 'rb') as f:
                loaded_dict = pickle.load(f)
                self.count = loaded_dict["count"]
                self.images = loaded_dict["images"]

        else:
            self.count = 0
            images_lst = []
            print("Searching " + self.root_dir + " for files...")
            for path in tqdm(os.scandir(self.root_dir)):
                filestring = path.path[len(self.root_dir) + 1:]
                if path.is_file() and filestring.endswith('.png'):
                    images_lst.append(self.transform(Image.open(path.path).convert('RGB')).to(self.device))
                    self.count += 1
                    if self.size_override is not None and self.count >= self.size_override:
                        break

            if self.size_override is not None and self.size_override > -1:
                self.count = min(self.size_override, self.count)

            print("Formatting Images...")
            self.images = torch.zeros((len(images_lst), 1, images_lst[0].shape[-2], images_lst[0].shape[-1]), device=self.device)
            idx = 0
            for image in tqdm(images_lst):
                self.images[idx, :] = image
                idx += 1

            loaded_dict = {
                    "count" : self.count,
                    "images" : self.images, 
                }
                
            with open(pickled_data_path, 'wb') as f:
                pickle.dump(loaded_dict, f)
            
            print("Created Pickled Dataset!")

    def __len__(self):
        return self.count
        
    def __getitem__(self, index):

        return self.images[index]
    



class SketchPointDataset(data.Dataset):
    """Simple class for storing a sketch dataset."""

    def __init__(self, root_dir : str, size_override : int = None, num_points = 10000, device = DEFAULT_DEVICE):
        """
        Parameters
        -----------
        root_dir : str
            Directory with all the images (must be in png format).
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
        self.device = device
        self.size_override = size_override
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Grayscale(1)])

        pickled_data_path = root_dir + "/" + "stored_data.pkl"

        if os.path.isfile(pickled_data_path):
            with open(pickled_data_path, 'rb') as f:
                loaded_dict = pickle.load(f)
                self.count = loaded_dict["count"]
                self.images = loaded_dict["images"]
                self.image_indices = loaded_dict["image_indices"]
                self.points = loaded_dict["points"]
                self.sds = loaded_dict["sds"]

        else:
            self.count = 0

            images_lst = []
            points_lst = []
            use_override = self.size_override is not None and self.size_override > -1

            print("Searching " + self.root_dir + " for files...")
            for path in tqdm(os.scandir(self.root_dir)):          
                img_path = path.path + "\sketch.png"
                images_lst.append(self.transform(Image.open(img_path).convert('RGB')).to(self.device))

                points_path = path.path + "\sample_points.csv"
                points_frame = pd.read_csv(points_path, header=None)

                surface_start = (points_frame.index[(points_frame == "Surface Points").any(axis=1)]).array[0]
                random_start = (points_frame.index[(points_frame == "Random Points").any(axis=1)]).array[0]

                grid_frame = points_frame.iloc[1:surface_start,:]
                surface_frame = points_frame.iloc[surface_start + 1:random_start,:]
                random_frame = points_frame.iloc[random_start + 1:,:]  

            # surface_subset = surface_frame.sample(n=4000, replace=False)

                points = torch.Tensor(pd.concat([grid_frame, surface_frame, random_frame]).astype('float64').values).to(self.device)
                if points.shape[0] < num_points:
                    print(self.count, path.path)
                    print("  ", points.shape[0])
                    print("  ", grid_frame.shape[0])
                    print("  ", surface_frame.shape[0])
                    print("  ", random_frame.shape[0])
                points = points[0:num_points]
                points_lst.append(points)
                
                # print(filestring)
                self.count += 1

                if use_override and self.count >= self.size_override:
                    break

            print("Formatting Images...")
            self.images = torch.zeros((len(images_lst), 1, images_lst[0].shape[-2], images_lst[0].shape[-1]), device=self.device)
            idx = 0
            for image in tqdm(images_lst):
                self.images[idx, :] = image
                idx += 1

            print("Formatting Points...")
            self.image_indices = torch.zeros(len(points_lst) * num_points, device=self.device).to(torch.int32)
            self.points = torch.zeros((len(points_lst) * num_points, 3), device=self.device)
            self.sds = torch.zeros(len(points_lst) * num_points, device=self.device)
            idx = 0
            for point_tensor in tqdm(points_lst):
                self.image_indices[idx * num_points : (idx + 1) * num_points] = idx
                self.points[idx * num_points : (idx + 1) * num_points,:] = point_tensor[:,0:3]
                self.sds[idx * num_points : (idx + 1) * num_points] = point_tensor[:,3]
                idx += 1
            
            loaded_dict = {
                "count" : self.count,
                "images" : self.images,
                "image_indices" : self.image_indices, 
                "points" : self.points,
                "sds" : self.sds, 
            }
            
            with open(pickled_data_path, 'wb') as f:
                pickle.dump(loaded_dict, f)
            
            print("Created Pickled Dataset!")
            # print(self.names)

    def __len__(self):
        return self.count
        
    def __getitem__(self, index):
        # print(index)
        # print(self.image_indices.shape)
        # print(self.image_indices[index])
        return self.images[self.image_indices[index]], self.points[index], self.sds[index]



    