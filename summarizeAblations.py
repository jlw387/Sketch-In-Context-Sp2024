import os
import numpy as np
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from PIL import Image

dir_str = "Ablations/CVAE/"

run_folders = os.listdir(dir_str)

latent_dimensions_keyphrase = "Latent Dimensions: "
learning_rate_keyphrase = "Learning Rate: "
max_elbo_keyphrase = "Max Test ELBO: "

overall_max_elbo_index = -1
overall_max_elbo = -float('inf')

latent_dims = []
learning_rates = []

counter = 0
run_descriptions = []
training_elbos = []
test_elbos = []

bad_elbos = {}

#color_list = list(colors.BASE_COLORS.values())
color_list = [(1.0,0.0,0.0),
              (1.0,0.5,0.0),
              (0.95,0.95,0.0),
              (0.25,1.0,0.25),
              (0.25,0.75,0.75),
              (0.5,0.65,1.0)]

for folder in run_folders:

    # Process Run Descriptions
    with open(dir_str + folder + "/RunDescription.txt") as run_description_file:
        run_description = run_description_file.readlines()
        run_descriptions.append(run_description)
        for line in run_description:

            # Get Latent Dimensions
            if line.startswith(latent_dimensions_keyphrase):
                latent_dims.append(int(line[len(latent_dimensions_keyphrase):]))

            # Get Learning Rate
            if line.startswith(learning_rate_keyphrase):
                learning_rates.append(float(line[len(learning_rate_keyphrase):]))

            # Get Max Elbo
            if line.startswith(max_elbo_keyphrase):
                current_max_elbo = float(line[len(max_elbo_keyphrase):])
                
                # Check for Best ELBO
                if current_max_elbo > overall_max_elbo:
                    overall_max_elbo = current_max_elbo
                    overall_max_elbo_index = counter

                # Check for Bad ELBO
                if current_max_elbo < -25000:
                    bad_elbos[counter] = current_max_elbo

    # Process Elbo Data
    training_elbos.append(np.loadtxt(dir_str + folder + "/TrainingElbos.txt"))
    test_elbos.append(np.loadtxt(dir_str + folder + "/TestElbos.txt"))

    counter += 1

# Print best result
print("Best ELBO:", overall_max_elbo)
print("\tBest ELBO Index:", overall_max_elbo_index)
print("\tLatent Dimensions:", latent_dims[overall_max_elbo_index])  
print("\tLearning Rate:", learning_rates[overall_max_elbo_index]) 

# Plot training/test of best run
# plt.plot(training_elbos[overall_max_elbo_index])
# plt.plot(test_elbos[overall_max_elbo_index])
# plt.title("Best Run")
# plt.show()

# print("Plot Color Order:\n===================")
# for c_index in range(6):
#     print(" ", color_list[c_index])

# print("\n")

# Plot all training runs
# counter = 0
# for te in training_elbos:
#     plt.plot(te, color=color_list[counter // 4])
#     counter += 1

# plt.title("Training Elbos vs. Epoch")
# plt.show()


# Plot all test runs
# counter = 0
# for te in test_elbos:
#     plt.plot(te, color=color_list[counter // 4])
#     counter += 1

# plt.title("Test Elbos vs. Epoch")
# plt.show()

# Plot all training runs by batch size
# for i in range(6):
#     for j in range(4):
#         plt.plot(training_elbos[i*4 + j])

#     plt.title("Training Run " + str(i + 1))
#     plt.show()

# Plot all test runs by batch size
# for i in range(0,6):
#     for j in range(4):
#         plt.plot(test_elbos[i*4 + j])

#     plt.title("Test Run " + str(i + 1))
#     plt.show()

# Plot all test runs by learning rate
# for i in range(4):
#     for j in range(6):
#         plt.plot(test_elbos[i+j*4], label="LD-" + str(latent_dims[i + j*4]))
    
#     plt.title("LR " + str(learning_rates[i]))
#     plt.legend()
#     plt.show()


# Print Failed Run Folder Names:
print("Failed Test Runs:\n======================")
for k in bad_elbos.keys():
    print(" ", run_folders[k])
    plt.imshow(Image.open(dir_str + run_folders[k] + "/Reprojections/PreTrain.jpg"))
    plt.show()
        