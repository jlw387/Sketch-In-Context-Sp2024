import os
from datetime import datetime

from PIL import Image
import numpy as np

import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision.io import ImageReadMode, read_image

from matplotlib import pyplot as plt

import MegaNet.network as network

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Written by ChatGPT
def find_most_recent_model(directory_path):
    # Get list of folders in the directory
    folders = os.listdir(directory_path)
    
    # Filter out non-folders
    folders = [f for f in folders if os.path.isdir(os.path.join(directory_path, f))]
    
    if not folders:
        return None  # No folders found
    
    # Function to extract timestamp from folder name
    def extract_timestamp(folder_name):
        return datetime.strptime(folder_name, "%Y_%m_%d_%H_%M_%S")
    
    # Sort folders by timestamp
    sorted_folders = sorted(folders, key=extract_timestamp, reverse=True)
    
    # Return path to most recent folder
    return os.path.join(directory_path, sorted_folders[0])

def eval_sketch_pretrainer(img_filename : str, network_param_filename : str):

    img_name_relevant = img_filename[img_filename.rfind("/") + 1:-4]

    try:
        img_tensor = (read_image(img_filename, ImageReadMode.GRAY).float() / 255).unsqueeze(0).to(DEVICE)
    except:
        print("Could not find image at \'" + img_filename + "\'")
        return
    
    vsp_network = network.load_VSP(network_param_filename)
    if vsp_network is None:
        return
    
    reprojection, mean, logVar = vsp_network(img_tensor)

    print(reprojection)
    print(vsp_network.old_loss_fn(img_tensor, reprojection, mean, logVar))

    reprojection_img = reprojection.detach().cpu().numpy().reshape(reprojection.shape[-2:])

    plt.imshow(reprojection_img, cmap='Greys_r')
    plt.show()

    save_image(img_tensor, "TestResults/" + img_name_relevant + "_Original.png")
    save_image(reprojection, "TestResults/" + img_name_relevant + "_Reprojected.png")

if __name__ == "__main__":
    img_file = input("Enter Image File Name:")
    if img_file == "":
        img_file = "Sketches/Renders2_Test/Render1.png"
    
    param_file = input("Enter Network Directory:")
    if param_file == "":
        param_file = find_most_recent_model("Models/CVAE")

    eval_sketch_pretrainer(img_file, param_file)