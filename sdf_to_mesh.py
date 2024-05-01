import numpy as np
import os
import sys
from tqdm import tqdm
from skimage import measure
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision import transforms


from MegaNet.network import *
import dataset as dataset
from obj_export import obj_exporter
from torch.utils.data import DataLoader

DEVICE = torch.device("cuda")

# test_number = 0
def pad_to_square(image):
    """
    Pad the input PIL image to form a square based on the longest edge.

    Args:
        image (PIL.Image): Input image.

    Returns:
        PIL.Image: Padded image.
    """
    width, height = image.size
    max_size = max(width, height)
    pad_width = (max_size - width) // 2
    pad_height = (max_size - height) // 2
    padding = (pad_width, pad_height, pad_width, pad_height)
    return transforms.functional.pad(image, padding, fill=255)

def sdf_to_mesh(img_path, export_path):
    weight_number = "2024_04_12_10_45_04"
    # weight_number = "2024_03_26_03_58_24"
    network_depth = 4
    filter_size = 3
    final_grid_size = 16
    # batch_size_param = 450

    # time_stamp = "Gen_1710532434"
    # sketch_points_dir = f"SDFDatasets\{time_stamp}_Train"
    dir_str = "Models/SDF/" + weight_number
    # model_dir = sketch_points_dir + f"\Model_{test_number}"



    # def extract_camera_params(camera_params_file):
    #     with open(camera_params_file, 'r') as f:
    #         lines = f.readlines()
    #         camera_position_str = lines[0].strip().split(': ')[1].strip('()').split(',')  # Extract position string and remove parentheses
    #         camera_position = [float(x) for x in camera_position_str]
    #         camera_scale_factor = float(lines[1].strip().split(':')[1])
    #     return camera_position, camera_scale_factor

    # model_dir = os.path.join(sketch_points_dir, f"Model_{test_number}")
    # camera_params_file = os.path.join(model_dir, 'camera_params.txt')
    # print(model_dir)
    # print(camera_params_file)
    # open(camera_params_file, 'r')

    # # Check if camera_params.txt exists for the current model
    # if os.path.exists(camera_params_file):
    #     # Extract camera parameters
    #     camera_position, camera_scale_factor = extract_camera_params(camera_params_file)
        
    #     # Print or do something with the camera parameters
    #     print(f"Model: {model_dir}")
    #     print(f"Camera Position: {camera_position}")
    #     print(f"Camera Scale:  {camera_scale_factor}")
    #     print("\n")

    latent_dimensions_keyphrase = "Latent Dimensions: "
    # learning_rate_keyphrase = "Learning Rate: " 

    weight_path = dir_str + "/ModelWeights/FinalWeights.pt"
    # test_dataset = dataset.SketchPointDataset(sketch_points_dir, 1500, num_points=9500, device=DEVICE)
    # test_dataloader = DataLoader(test_dataset, batch_size=batch_size_param, shuffle=False)
    # print(test_dataset[0*9500])
    # print(test_dataset[1*9500])
    # flattened_tensor = test_dataset[0*9500][0].flatten()
    # # Write tensor shape to file
    # with open("tensor_data.txt", "w") as file:
    #     file.write(f"{test_dataset[0*9500][0].shape}\n")

    # # Write flattened tensor data to file
    # with open("tensor_data.txt", "a") as file:
    #     for value in flattened_tensor:
    #         file.write(f"{value}\n")
    # flattened_tensor = test_dataset[1*9500][0].flatten()
    # with open("tensor_data1.txt", "w") as file:
    #     file.write(f"{test_dataset[1*9500][0].shape}\n")

    # # Write flattened tensor data to file
    # with open("tensor_data1.txt", "a") as file:
    #     for value in flattened_tensor:
    #         file.write(f"{value}\n")

    # def compare_text_files(file1_path, file2_path):
    #     # Read the contents of both files
    #     with open(file1_path, 'r') as file1:
    #         file1_content = file1.read()

    #     with open(file2_path, 'r') as file2:
    #         file2_content = file2.read()

    #     # Compare the contents
    #     if file1_content == file2_content:
    #         print("The files are the same.")
    #     else:
    #         print("The files are different.")

    # # Example usage
    # file1_path = "tensor_data.txt"
    # file2_path = "tensor_data1.txt"
    # compare_text_files(file1_path, file2_path)

    with open(dir_str + "/RunDescription.txt") as run_description_file:
        run_description = run_description_file.readlines()
        for line in run_description:

            # Get Latent Dimensions
            if line.startswith(latent_dimensions_keyphrase):
                latent_dims = int(line[len(latent_dimensions_keyphrase):])

    # # Initialize network
    network = SketchToSDF(depth=network_depth, 
                        latent_dims=latent_dims,
                        filter_size=filter_size,
                        final_grid_size=final_grid_size,device=DEVICE)

    checkpoint = torch.load(weight_path) 
    network.load_state_dict(checkpoint)


    grid_size = 64
    cube_center = 0,0,0
    cube_range = 1

    increment = 2/(grid_size)

    # test = test_dataset[test_number*9500][0]
    # Open an image file

    transform = transforms.Compose([transforms.ToTensor(), transforms.Grayscale(1), transforms.Resize((256, 256))])
    tensor_image = transform(pad_to_square(Image.open(img_path).convert('RGB'))).cuda()

    to_pil = transforms.ToPILImage()
    image = to_pil(tensor_image)
    image.save("see.png")
    
    img_tensor = tensor_image.unsqueeze(1)
    print(type(img_tensor))
    print(img_tensor.shape)
    # img_tensor = test.unsqueeze(1)
    # print(img_tensor.shape)
 

    x_grid, y_grid, z_grid = torch.meshgrid(torch.arange(grid_size + 1), torch.arange(grid_size + 1), torch.arange(grid_size + 1))
    point_tensor = 2 * (torch.stack((x_grid, y_grid, z_grid), dim=-1)/grid_size) - 1
    # print(point_tensor)

        
    with torch.no_grad():
        z = network.forward_encoder_section(img_tensor)

    with torch.no_grad():
        # print(z.shape)
        # print(torch.Tensor(point_tensor).shape)
        result = network.forward_sdf_section(z, torch.Tensor(point_tensor)).cpu().unsqueeze(-1).numpy()

    evals = np.reshape(result,(grid_size+1, grid_size+1, grid_size+1))

    # camera_position = np.array([x, y, z])     #here to extract camera params
    # camera_orientation = np.eye(3)  

    # def preprocess_data(data, camera_pos, camera_orientation):
    #     translated_data = data - camera_pos
    #     rotated_data = np.transpose(translated_data, axes=(2, 1, 0))  # Adjust axes for scikit-image
    #     return rotated_data

    # preprocessed_data = preprocess_data(evals, camera_position, camera_orientation)

    verts, faces, normals, values = measure.marching_cubes(evals, 0)

    # print(verts.max())
    # print(verts.min())
    # print(faces)

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot_trisurf(verts[:, 0], verts[:,1], faces, verts[:, 2],
    #                 cmap='Spectral', lw=1)
    plt.show()

    # END TODO

    # export_path = input("Enter Export Path: ")
    obj_exporter(export_path, 2*verts/grid_size - 1, faces)


# Check if arguments are provided
if len(sys.argv) < 3:
    print("Usage: python your_script.py <value>")
    sys.exit(1)

# Get the value passed as argument
img_path = sys.argv[1]
export_path = sys.argv[2]
print(img_path)
print(export_path)

# Now you can use the value in your script
sdf_to_mesh(img_path, export_path)
# Do whatever you want with the value here