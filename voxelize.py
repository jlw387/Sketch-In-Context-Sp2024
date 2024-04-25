"""
Used for visualizing SDFs with matplotlib's voxel grid plotting capabilities.

If the default plot is closed, the user is allowed to set a different signed distance
as the threshold for the surface. By default, this is 0, but this added functionality
allows the user to see the isosurface slightly beyond or slightly within the true surface.
Mostly useful for debugging.

Model path/architecture loading is adjusted each time the file is used, so the early lines
are changed every time a new model is loaded.

Could DEFINITELY use some numpy array optimization for generating the voxel grid in the first place.
"""

from math import inf
import numpy as np
import torch

from MegaNet.network import SketchToSDF, load_S2SDF
from dataset import load_image

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

# network_timestamp = "2022_06_23_18_43_40"
network_timestamp = "2024_03_20_13_29_59"

image_path = "./SDFDatasets/Gen_1710532434_Train/Model_0/sketch.png"
# code_path = "../SDFGen/BigDataset/Gen_1661051586/Model_0/code2022_06_22_11_22_48_1024.txt"

image_tensor = load_image(image_path, device=torch.device('cuda'))

# Initialize network
model = load_S2SDF(network_dir="Models/SDF/" + network_timestamp, weights_string="FinalWeights", device=torch.device('cuda'))

cube_size = 32
cube_center = 0,0,0
cube_range = 1

increment = 2/(cube_size - 1)

image_input = torch.zeros((cube_size**2, 1, image_tensor.shape[-2], image_tensor.shape[-1]))
image_input[:,:,:,:] = image_tensor
print("Image Input:", image_input.shape)
image_input = image_input.to(torch.device('cuda'))

points_input = torch.zeros((cube_size**3, 3))

for i in tqdm(range(cube_size)):
    i_val = (-1 + increment*i)*cube_range + cube_center[0]
    for j in range(cube_size):
        j_val = (-1 + increment*j)*cube_range + cube_center[1]
        for k in range(cube_size):
            k_val = (-1 + increment*k)*cube_range + cube_center[2]
            points_input[i*cube_size**2 + j*cube_size + k,:] = torch.Tensor([i_val, j_val, k_val])

print("Point Input:", points_input.shape)

val = torch.zeros((cube_size**3, 1))

print("\nEvaluating Model:")
for i in tqdm(range(cube_size)):
    result = model(image_input, points_input[i*(cube_size**2) : (i + 1)*cube_size**2,:]).detach()
    val[i*(cube_size**2) : (i + 1)*cube_size**2,:] = result

print("Finished Loading!")

print("Average Signed Distance:", val.mean())

threshold = 0

while(True):
    print("Gettting Voxels...")
    voxels = val.numpy().reshape((cube_size, cube_size, cube_size)) <= threshold

    print("Plotting Voxels...")
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.voxels(voxels)

    print("Showing plot...")
    plt.show()

    a = input("Enter new threshold or 'q' to quit: ")

    try:
        threshold = float(a)
    except ValueError:
        print("Quitting...")
        break
