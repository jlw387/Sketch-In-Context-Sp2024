import time
import datetime
import os

import numpy as np

import pickle

import torch
from torch.utils.data import DataLoader

from matplotlib import pyplot as plt

import dataset as dataset
from MegaNet.network import SketchToSDF, VariationalSketchPretrainer

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE = torch.device("cuda")

def gpu_tensor_to_np_image(t):
    np_t = t.detach().cpu().numpy()
    return np_t.reshape(np_t.shape[-2:])

def reprojection(weight_number, test_number, use_augmentation = True):
    
    network_depth = 4
    filter_size = 3
    final_grid_size = 16

    sketch_dir_test = "Sketches/Renders2_Test"
    dir_str = "Ablations/CVAE/" + weight_number

    latent_dimensions_keyphrase = "Latent Dimensions: "
    # learning_rate_keyphrase = "Learning Rate: " 

    weight_path = dir_str + "/ModelWeights/FinalWeights.pt"
    test_dataset = dataset.SketchDataset(sketch_dir_test, use_augmentation, 512, device=DEVICE)

    with open(dir_str + "/RunDescription.txt") as run_description_file:
        run_description = run_description_file.readlines()
        for line in run_description:

            # Get Latent Dimensions
            if line.startswith(latent_dimensions_keyphrase):
                latent_dims = int(line[len(latent_dimensions_keyphrase):])

            # # Get Learning Rate
            # if line.startswith(learning_rate_keyphrase):
            #     learning_rate = float(line[len(learning_rate_keyphrase):])



    # # Initialize network
    network = VariationalSketchPretrainer(depth=network_depth, 
                                          latent_dims=latent_dims,
                                          filter_size=filter_size,
                                          final_grid_size=final_grid_size,device=DEVICE)
    
    # optimizer = torch.optim.Adam(network.parameters(), learning_rate)

    checkpoint = torch.load(weight_path) 
    network.load_state_dict(checkpoint)
    # optimizer.load_state_dict(checkpoint['opt'])

    test = test_dataset[test_number]
    reprojections, _, _, _ = network.forward(test)

    fig = plt.imshow(gpu_tensor_to_np_image(reprojections[0,:]), cmap='Greys_r')
    plt.axis('off')
    plt.show()


weight_number = "2024_02_27_07_10_28"
test_number = 20

reprojection(weight_number, test_number)