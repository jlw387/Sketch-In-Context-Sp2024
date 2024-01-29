from PIL import Image
import numpy as np

import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision.io import ImageReadMode, read_image

import MegaNet.network as network

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

img_transform = transforms.Lambda(lambda image: torch.from_numpy(np.array(image).astype(np.float32)).unsqueeze(0))

def eval_sketch_pretrainer(img_filename, network_param_filename):
    try:
        img_tensor = (read_image(img_filename, ImageReadMode.GRAY).float() / 255).unsqueeze(0).to(DEVICE)
    except:
        print("Could not find image at \'" + img_filename + "\'")
        return
    
    vsp_network = network.load_VSP(network_param_filename)
    if vsp_network is None:
        return
    
    reprojection, mean, logVar = vsp_network(img_tensor)

    print(vsp_network.loss_function(img_tensor, reprojection, mean, logVar))

    save_image(reprojection, img_filename[:-4] + "_Reprojected.png")

if __name__ == "__main__":
    img_file = input("Enter Image File Name:")
    param_file = input("Enter Network Directory:")
    eval_sketch_pretrainer(img_file, param_file)