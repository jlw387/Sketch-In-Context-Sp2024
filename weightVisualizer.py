import numpy as np

import pickle

import torch
from torch.utils.data import DataLoader

from matplotlib import pyplot as plt

import dataset as dataset
from MegaNet.network import SketchToSDF, VariationalSketchPretrainer

network_depth = 4
filter_size = 3
final_grid_size = 16

weight_number = "2024_02_27_07_10_28"

sketch_dir_test = "Sketches/Renders2_Test"
dir_str = "Ablations/CVAE2/" + weight_number

latent_dimensions_keyphrase = "Latent Dimensions: "
# learning_rate_keyphrase = "Learning Rate: " 

weight_path = dir_str + "/ModelWeights/FinalWeights.pt"
test_dataset = dataset.SketchDataset(sketch_dir_test, False, 512, device=torch.device('cpu'))

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
                                        final_grid_size=final_grid_size,device=torch.device('cpu'))

# optimizer = torch.optim.Adam(network.parameters(), learning_rate)

checkpoint = torch.load(weight_path) 
network.load_state_dict(checkpoint)
# optimizer.load_state_dict(checkpoint['opt'])

weights = network.convLayers._modules['3'].weight.data.numpy()
sub_weights = weights[0,:32,:,:].reshape(32,3,3)

# Create a grid of subplots
fig, axs = plt.subplots(4, 8, figsize=(12, 6))

# Set the background color of the figure to grey
fig.patch.set_facecolor((0.2,0.2,0.2))

# Loop through each subplot and plot a 2D slice
for i in range(4):
    for j in range(8):
        filter_2D_grid = sub_weights[i * 8 + j,:,:].reshape(3,3)
        axs[i, j].imshow(filter_2D_grid, cmap='Greys_r', vmin=-1, vmax=1)  # You can change the colormap as needed
        axs[i, j].set_title('Filter {}'.format(i * 8 + j + 1), color='white')
        axs[i, j].axis('off')  # Turn off axis labels

plt.show()