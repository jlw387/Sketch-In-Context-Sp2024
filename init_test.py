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

def get_now():
    return datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

def gpu_tensor_to_np_image(t):
    np_t = t.detach().cpu().numpy()
    return np_t.reshape(np_t.shape[-2:])

def init_test(network_depth, latent_dims, filter_size, final_grid_size, sketch_dir_train, sketch_dir_test, batch_size_param, use_augmentation = True):
    """Trains a Variational Sketch Pretrainer network."""

    # Load training and test datasets
    train_dataset = dataset.SketchDataset(sketch_dir_train, use_augmentation, device=DEVICE)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_param, shuffle=True)

    test_dataset = dataset.SketchDataset(sketch_dir_test, use_augmentation, 256, device=DEVICE)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size_param, shuffle=False)

    # Initialize network
    network = VariationalSketchPretrainer(depth=network_depth, 
                                          latent_dims=latent_dims,
                                          filter_size=filter_size,
                                          final_grid_size=final_grid_size,device=DEVICE)                                        

    starting_test_loss = 0
    test_img_count = 0

    # Get starting test loss
    for idx, image_batch in enumerate(test_dataloader):
        # Get batch size (this could be less than the specified batch size
        # if the total image count is not divisible by the batch size)
        batch_size = image_batch.shape[0]

        # Add current batch size to total image count
        test_img_count += batch_size

        # Forward propagate image batch
        reprojections, z, mu, logVar = network(image_batch)

        # Compute batch loss
        batch_loss = network.bce_loss_fn(image_batch, reprojections, mu, logVar)

        # Add batch loss to overall epoch training loss
        starting_test_loss += batch_loss.item()

    starting_test_elbo = -starting_test_loss / test_img_count
    print("Starting Loss:", starting_test_elbo)
    # print("Mu:", mu)
    # print("Mu[0,:]:", mu[0,:])
    # print("LogVar:", logVar)
    # print("LogVar[0,:]:", logVar[0,:])


    fig = plt.imshow(gpu_tensor_to_np_image(reprojections[0,:]), cmap='Greys_r')
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    # NETWORK PARAMETERS

    network_depth = 4
    # latent_dims = 1024
    filter_size = 3
    final_grid_size = 16

    
    # RUN PARAMETERS
     
    sketch_dir_train = "Sketches/Renders"
    sketch_dir_test = "Sketches/Renders2_Test"

    batch_size_param = 256
    # learning_rate = 0.0001
    total_epochs = 400
    use_augmentation = True

    count = 0
    init_test(network_depth, 512, filter_size, final_grid_size, sketch_dir_train, sketch_dir_test, batch_size_param)