import time
import datetime
import os

import numpy as np

import pickle

import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from matplotlib import pyplot as plt

import dataset as dataset
from MegaNet.network import SketchToSDF, VariationalSketchPretrainer

# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE = torch.device("cuda")

# Set the environment variable
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

scaler = GradScaler()

def get_now():
    return datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

def gpu_tensor_to_np_image(t):
    np_t = t.detach().cpu().numpy()
    return np_t.reshape(np_t.shape[-2:])

def pre_train(network_depth, latent_dims, filter_size, final_grid_size, sketch_dir_train, sketch_dir_test, batch_size_param, learning_rate, total_epochs, run_description, use_augmentation = True):
    """Trains a Variational Sketch Pretrainer network."""

    # # NETWORK PARAMETERS

    # network_depth = 4
    # latent_dims = 1024
    # filter_size = 3
    # final_grid_size = 16

    
    # # RUN PARAMETERS
     
    # sketch_dir_train = "Sketches/Renders"
    # sketch_dir_test = "Sketches/Renders2_Test"

    # batch_size_param = 256
    # learning_rate = 0.0001
    # total_epochs = 20
    # use_augmentation = True


    # DATA COLLECTION PARAMETERS

    # The number of epochs between save attempts. If set to 'n', then every
    # 'n-th' epoch is a "save-epoch", where the weights are saved if certain 
    # conditions are met
    epoch_precision = 10  

    # If true, only keeps the weights of the highest performing "save-epoch"
    # along with the starting weights and final weights
    only_save_highest_performance_with_endpoints = True 

    # If true, does not save weights unless the current performance is higher
    # than the performance of the previously saved weights. This is ignored if
    # 'highest_performance_with_endpoints_only' is set to True. 
    only_save_higher_performance = True 

    # Get current time for timestamping this training run
    now = datetime.datetime.now()
    run_timestamp = get_now()
    
    if not os.path.isdir("Models/"):
        os.mkdir("Models/")

    if not os.path.isdir("Models/CVAE"):
        os.mkdir("Models/CVAE")

    # Set up output path/folder
    model_dir_name = "Models/CVAE/" + run_timestamp + "/"
    os.mkdir(model_dir_name)

    # Load training and test datasets
    train_dataset = dataset.SketchDataset(sketch_dir_train, use_augmentation, device=DEVICE)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_param, shuffle=True)

    test_dataset = dataset.SketchDataset(sketch_dir_test, use_augmentation, 512, device=DEVICE)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size_param, shuffle=False)

    # Initialize network
    network = VariationalSketchPretrainer(depth=network_depth, 
                                          latent_dims=latent_dims,
                                          filter_size=filter_size,
                                          final_grid_size=final_grid_size,device=DEVICE)
                                          
    # Initialize optimizer
    optimizer = torch.optim.Adam(network.parameters(), learning_rate)

    # Print layers to confirm architecture
    print("\n==================================")
    print("Network Architecture:")
    print("==================================\n")
    network.print_layers()

    # Save network parameters to file
    

    # Save starting weights
    os.mkdir(model_dir_name + "/ModelWeights/")
    torch.save(network.state_dict(), model_dir_name + "/ModelWeights/StartWeights.pt")

    # Create directory for saving reprojections
    os.mkdir(model_dir_name + "/Reprojections")

    # Get user description of run
    # run_description = input("Enter Run Description: ")

    # Initialize training loop parameters
    max_train_elbo = -1e20
    max_test_elbo = -1e20
    max_saved_test_elbo = -1e20
    max_saved_epoch = -1
    prev_max_saved_epoch = -1
    training_elbos = []
    test_elbos = []

    run_ended_early = False

    save_non_optimal_weights = not only_save_highest_performance_with_endpoints and not only_save_higher_performance

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
    test_elbos.append(starting_test_elbo)
    print("Starting Loss:", starting_test_elbo)

    fig = plt.imshow(gpu_tensor_to_np_image(reprojections[0,:]), cmap='Greys_r')
    plt.axis('off')
    plt.savefig(model_dir_name + '/Reprojections/PreTrain.jpg')

    # Get starting time for training loop
    print("Start Training!")
    start_time = time.time()

    try:
        # Overall training loop
        for epoch in range(total_epochs):
            epoch_start_time = time.time()

            # Initialize overall loss variable
            overall_training_loss = 0
            train_img_count = 0

            # Reset gradient to zero
            optimizer.zero_grad()

            # Per epoch training loop
            for idx, image_batch in enumerate(train_dataloader):
                # with autocast():

                # Get batch size (this could be less than the specified batch size
                # if the total image count is not divisible by the batch size)
                batch_size = image_batch.shape[0]

                # Add current batch size to total image count
                train_img_count += batch_size

                # Forward propagate image batch
                reprojections, z, mu, logVar = network(image_batch)

                # Compute batch loss
                batch_loss = network.bce_loss_fn(image_batch, reprojections, mu, logVar) 
                total_batch_loss = batch_loss.item()
                batch_loss = batch_loss / batch_size

                # Check for invalid loss
                if batch_loss < 0:
                    print("IMPOSSIBLE LOSS FOUND!\n")

                # Add batch loss to overall epoch training loss
                overall_training_loss += total_batch_loss

                # Backpropagate loss
                batch_loss.backward()
                optimizer.step()
                # scaler.scale(overall_training_loss).backward()
                # scaler.step(optimizer)
                # scaler.update()

            overall_test_loss = 0
            test_img_count = 0

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
                overall_test_loss += batch_loss.item()

            print("Epoch", epoch, "\n=============================")
            print("\tDuration:", time.time() - epoch_start_time, "s")

            avg_train_elbo = -overall_training_loss / train_img_count
            avg_test_elbo = -overall_test_loss / test_img_count

            training_elbos.append(avg_train_elbo)
            test_elbos.append(avg_test_elbo)

            print("\tAverage Training Loss:", avg_train_elbo)
            print("\tAverage Test Loss:", avg_test_elbo)

            # Update max elbo variables
            max_train_elbo = max(max_train_elbo, avg_train_elbo)
            max_test_elbo = max(max_test_elbo, avg_test_elbo)

            # Try to save the weights
            # 
            # First, check if this is a 'save-epoch'
            if (epoch + 1) % epoch_precision == 0:
                # Next, check if the test ELBO is higher than the last saved ELBO value
                if save_non_optimal_weights or avg_test_elbo > max_saved_test_elbo:
                    # Save current model weights
                    torch.save(network.state_dict(), model_dir_name + "/ModelWeights/Epoch" + str(epoch) + ".pt")
                    
                    # Update saved epoch parameters
                    prev_max_saved_epoch = max_saved_epoch
                    max_saved_epoch = epoch

                    # Delete old weights if 'only_save_highest_performance_with_endpoints' is set to 'True'
                    if only_save_highest_performance_with_endpoints and prev_max_saved_epoch != -1:
                        os.remove(model_dir_name + "/ModelWeights/Epoch" + str(prev_max_saved_epoch) + ".pt")

            fig.set_data(gpu_tensor_to_np_image(reprojections[0,:]))
            plt.savefig(model_dir_name + '/Reprojections/Epoch_' + str(epoch) + '.jpg')

        # Get ending time
        finish_time = time.time()

        # Save final weights
        torch.save(network.state_dict(), model_dir_name + "/ModelWeights/FinalWeights.pt")

    except Exception as e:
        print("Run Ended Early Due to Error!")
        print(e)
        run_ended_early = True

    except KeyboardInterrupt as ki:
        print("Run Ended Early Due to Error!")
        run_ended_early = True

    # Save network parameters
    with open(model_dir_name + 'network_parameters.pkl', 'wb') as f:
        pickle.dump(network.get_network_param_dict(), f)

    # Save loss information
    np.savetxt(model_dir_name + "TrainingElbos.txt", np.array(training_elbos))
    np.savetxt(model_dir_name + "TestElbos.txt", np.array(test_elbos))

    # Write out run information
    with open(model_dir_name + "RunDescription.txt", 'w') as f:
        f.write("Run Date: " + now.strftime("%m-%d-%Y") + "\n")
        f.write("Run Time: " + now.strftime("%H:%M:%S") + "\n\n")
        if run_ended_early:
            f.write("RUN TERMINATED EARLY!!\n\n")
        f.write("Directory: " + model_dir_name + "\n\n")
        f.write("Training Images Directory:" + sketch_dir_train + "\n")
        f.write("Number of Training Images: " + str(len(train_dataset)) + "\n\n")
        f.write("Test Images Directory:" + sketch_dir_test + "\n")
        f.write("Number of Test Images: " + str(len(test_dataset)) + "\n\n")
        f.write("Latent Dimensions: " + str(network.latent_dims) + "\n")
        f.write("Number of epochs: " + str(epoch + (0 if run_ended_early else 1)) + "\n")
        f.write("Batch Size: " + str(batch_size_param) + "\n")
        f.write("Learning Rate: " + str(learning_rate) + "\n")
        f.write("\nNetwork Architecture:\n")
        f.write(network.to_str() + "\n\n")
        if not run_ended_early:
            f.write("Computation Time: " + str(finish_time - start_time) + "\n")
        f.write("Max Training ELBO: " + str(max_train_elbo) + "\n")
        f.write("Max Test ELBO: " + str(max_test_elbo) + "\n")
        f.write("Use Augmentations: " + str(use_augmentation))
        # if use_augmentation:
        #     f.write("\n  Augmentation HFlip: " + str(augmentations[0]))
        #     f.write("\n  Augmentation Trans: " + str(augmentations[1]))
        #     f.write("\n  Augmentation Scale: " + str(augmentations[2]))
        #     f.write("\n  Augmentation Rotate: " + str(augmentations[3]))
        f.write("\nRun Description: " + run_description + "\n")
        f.close()

    
def train(network_depth, latent_dims, filter_size, final_grid_size, sketch_points_dir_train, sketch_points_dir_test, weight_time_stamp, batch_size_param, learning_rate, total_epochs, load_weight, frozen, run_description):
    # DATA COLLECTION PARAMETERS
    
    # The number of epochs between save attempts. If set to 'n', then every
    # 'n-th' epoch is a "save-epoch", where the weights are saved if certain 
    # conditions are met
    
    epoch_precision = 10  

    # If true, only keeps the weights of the highest performing "save-epoch"
    # along with the starting weights and final weights
    only_save_highest_performance_with_endpoints = True 

    # If true, does not save weights unless the current performance is higher
    # than the performance of the previously saved weights. This is ignored if
    # 'highest_performance_with_endpoints_only' is set to True. 
    only_save_higher_performance = True 

    # Get current time for timestamping this training run
    now = datetime.datetime.now()
    run_timestamp = get_now()
    
    # Load training and test datasets
    train_dataset = dataset.SketchPointDataset(sketch_points_dir_train, 13500, num_points=9500, device=DEVICE)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_param, shuffle=True)

    test_dataset = dataset.SketchPointDataset(sketch_points_dir_test, 1500, num_points=9500, device=DEVICE)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size_param, shuffle=False)
    
    # for idx, batch in enumerate(train_dataloader):
    #     print(idx)
    #     print(len(batch))
    #     image_batch, points_batch = batch
    #     print(batch[0].shape)
    #     print(batch[1].shape)
    #     break

    # print("end enum")

    dataset_allocated_memory = torch.cuda.memory_allocated(DEVICE) / 1024 / 1024
    print("Dataset Memory Usage:", dataset_allocated_memory, "MB")

    if not os.path.isdir("Models/"):
        os.mkdir("Models/")

    if not os.path.isdir("Models/SDF"):
        os.mkdir("Models/SDF")

    # Set up output path/folder
    model_dir_name = "Models/SDF/" + run_timestamp + "/"
    os.mkdir(model_dir_name)

    # Initialize network
    network = SketchToSDF(depth=network_depth, 
                            latent_dims=latent_dims,
                            filter_size=filter_size,
                            final_grid_size=final_grid_size,device=DEVICE)
    
    if load_weight:
        # load pretrained weights
        dir_str = "Ablations/CVAE2/" + weight_time_stamp
        weight_path = dir_str + "/ModelWeights/FinalWeights.pt"

        pretrained_weights = torch.load(weight_path) 

        for key in pretrained_weights.keys():
            if key in network.state_dict().keys():
                # print(key)
                # print(pretrained_weights[key].shape)
                # print(network.state_dict()[key].shape)
                network.state_dict()[key].copy_(pretrained_weights[key])
        if frozen:
            # Freeze encoding section
            network.set_encoder_frozen(True) 

    # Initialize optimizer
    optimizer = torch.optim.Adam(network.parameters(), learning_rate)

    for param_group in optimizer.param_groups:
        print("LR:", param_group['lr'])

    network_allocated_memory = torch.cuda.memory_allocated(DEVICE) / 1024 / 1024 - dataset_allocated_memory

    print("Network Device:", next(network.parameters()).device)

    
    print("Network Memory Usage:", network_allocated_memory, "MB")
    print("Total Memory Usage:", torch.cuda.memory_allocated(DEVICE) / 1024 / 1024, "MB")

    # Print layers to confirm architecture
    print("\n==================================")
    print("Network Architecture:")
    print("==================================\n")
    network.print_layers()

    # # Save network parameters to file

    # Save starting weights
    os.mkdir(model_dir_name + "/ModelWeights/")
    torch.save(network.state_dict(), model_dir_name + "/ModelWeights/StartWeights.pt")

    # Get user description of run
    # run_description = input("Enter Run Description: ")

    # Initialize training loop parameters
    min_train_loss = 1.0e20
    min_test_loss = 1.0e20
    min_saved_test_loss = 1.0e20
    min_saved_epoch = -1
    prev_min_saved_epoch = -1
    training_losses = []
    test_losses = []

    run_ended_early = False

    save_non_optimal_weights = not only_save_highest_performance_with_endpoints and not only_save_higher_performance

    starting_train_loss = 0
    train_img_count = 0
    starting_test_loss = 0
    test_img_count = 0

    with torch.no_grad():
        # Get starting train loss
        for idx, (image_batch, points_batch, sds_batch) in enumerate(train_dataloader):
            with autocast():
                # Get batch size (this could be less than the specified batch size
                # if the total image count is not divisible by the batch size)
                batch_size = image_batch.shape[0]

                # Add current batch size to total image count
                train_img_count += batch_size

                # Forward propagate image batch
                preds = network(image_batch, points_batch)

                # Compute batch loss
                batch_loss = network.compute_loss(preds, sds_batch)

                # Add batch loss to overall epoch training loss
                starting_train_loss += batch_loss.item()

        starting_train_loss = starting_train_loss / train_img_count
        training_losses.append(starting_train_loss)
        print("Starting Train Loss:", starting_train_loss)

        # Get starting test loss
        for idx, (image_batch, points_batch, sds_batch) in enumerate(test_dataloader):
            with autocast():
                # Get batch size (this could be less than the specified batch size
                # if the total image count is not divisible by the batch size)
                batch_size = image_batch.shape[0]

                # Add current batch size to total image count
                test_img_count += batch_size

                # Forward propagate image batch
                preds = network(image_batch, points_batch)

                # Compute batch loss
                batch_loss = network.compute_loss(preds, sds_batch)

                # Add batch loss to overall epoch training loss
                starting_test_loss += batch_loss.item()

        starting_test_loss = starting_test_loss / test_img_count
        test_losses.append(starting_test_loss)
        print("Starting Test Loss:", starting_test_loss)

    # Get starting time for training loop
    print("Start Training!")
    start_time = time.time()

    try:
        # Overall training loop
        for epoch in range(total_epochs):
            epoch_start_time = time.time()

            # Initialize overall loss variable
            overall_training_loss = 0
            train_img_count = 0

            # Per epoch training loop
            for idx, (image_batch, points_batch, sds_batch) in enumerate(train_dataloader):
                # Reset gradient to zero
                optimizer.zero_grad()
                with autocast():
                    if torch.cuda.is_available():
                        # Get batch size (this could be less than the specified batch size
                        # if the total image count is not divisible by the batch size)
                        batch_size = image_batch.shape[0]

                        # Add current batch size to total image count
                        train_img_count += batch_size

                        # Forward propagate image batch
                        preds = network(image_batch, points_batch)

                        # Compute batch loss
                        batch_loss = network.compute_loss(preds, sds_batch)
                        total_batch_loss = batch_loss.item()
                        batch_loss = batch_loss / batch_size

                        # Check for invalid loss
                        if batch_loss < 0:
                            print("IMPOSSIBLE LOSS FOUND!\n")

                        # Add batch loss to overall epoch training loss
                        overall_training_loss += total_batch_loss

                        # Backpropagate loss
                        # batch_loss.backward()
                        # optimizer.step()
                        scaler.scale(batch_loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

            overall_test_loss = 0
            test_img_count = 0

            for idx, (image_batch, points_batch, sds_batch) in enumerate(test_dataloader):
                with autocast():
                    if torch.cuda.is_available():
                        # Get batch size (this could be less than the specified batch size
                        # if the total image count is not divisible by the batch size)
                        batch_size = image_batch.shape[0]

                        # Add current batch size to total image count
                        test_img_count += batch_size

                        # Forward propagate image batch
                        preds = network(image_batch, points_batch)

                        # Compute batch loss
                        batch_loss = network.compute_loss(preds, sds_batch)

                        # Add batch loss to overall epoch training loss
                        overall_test_loss += batch_loss.item()


            print("Epoch", epoch, "\n=============================")
            print("\tDuration:", time.time() - epoch_start_time, "s")

            avg_train_loss = overall_training_loss / train_img_count
            avg_test_loss = overall_test_loss / test_img_count

            training_losses.append(avg_train_loss)
            test_losses.append(avg_test_loss)

            print("\tAverage Training Loss:", avg_train_loss)
            print("\tAverage Test Loss:", avg_test_loss)
            print("\tMemory Allocated:", torch.cuda.memory_allocated() / 1024 / 1024, "MB")

            # Update min loss variables
            min_train_loss = min(min_train_loss, avg_train_loss)
            min_test_loss = min(min_test_loss, avg_test_loss)

            # Try to save the weights
            # 
            # First, check if this is a 'save-epoch'
            if (epoch + 1) % epoch_precision == 0:
                # Next, check if the test ELBO is higher than the last saved ELBO value
                if save_non_optimal_weights or avg_test_loss < min_saved_test_loss:
                    # Save current model weights
                    torch.save(network.state_dict(), model_dir_name + "/ModelWeights/Epoch" + str(epoch) + ".pt")
                    
                    # Update saved epoch parameters
                    prev_min_saved_epoch = min_saved_epoch
                    min_saved_epoch = epoch
                    min_saved_test_loss = avg_test_loss

                    # Delete old weights if 'only_save_highest_performance_with_endpoints' is set to 'True'
                    if only_save_highest_performance_with_endpoints and prev_min_saved_epoch != -1:
                        os.remove(model_dir_name + "/ModelWeights/Epoch" + str(prev_min_saved_epoch) + ".pt")


        # Get ending time
        finish_time = time.time()

        # Save final weights
        torch.save(network.state_dict(), model_dir_name + "/ModelWeights/FinalWeights.pt")

    except Exception as e:
        print("Run Ended Early Due to Error!")
        print(e)
        run_ended_early = True

    except KeyboardInterrupt as ki:
        print("Run Ended Early Due to CTRL + C!")
        run_ended_early = True

    # Save network parameters
    with open(model_dir_name + 'network_parameters.pkl', 'wb') as f:
        pickle.dump(network.get_network_param_dict(), f)

    # Save loss information
    np.savetxt(model_dir_name + "TrainingLosses.txt", np.array(training_losses))
    np.savetxt(model_dir_name + "TestLosses.txt", np.array(test_losses))

    # Write out run information
    with open(model_dir_name + "RunDescription.txt", 'w') as f:
        f.write("Run Date: " + now.strftime("%m-%d-%Y") + "\n")
        f.write("Run Time: " + now.strftime("%H:%M:%S") + "\n\n")
        if run_ended_early:
            f.write("RUN TERMINATED EARLY!!\n\n")
        f.write("Directory: " + model_dir_name + "\n\n")
        f.write("Number of Training Images: " + str(len(train_dataset)) + "\n")
        f.write("Number of Test Images: " + str(len(test_dataset)) + "\n\n")
        f.write("Latent Dimensions: " + str(network.latent_dims) + "\n")
        f.write("Number of epochs: " + str(epoch + (0 if run_ended_early else 1)) + "\n")
        f.write("Batch Size: " + str(batch_size_param) + "\n")
        f.write("Learning Rate: " + str(learning_rate) + "\n")
        f.write("\nNetwork Architecture:\n")
        f.write(network.to_str() + "\n\n")
        if not run_ended_early:
            f.write("Computation Time: " + str(finish_time - start_time) + "\n")
        f.write("Loss Type: " + network.get_loss_name() + "\n")
        f.write("Starting Training Loss:" + str(starting_train_loss) + "\n")
        f.write("Starting Test Loss:" + str(starting_test_loss) + "\n")
        f.write("Min Training Loss: " + str(min_train_loss) + "\n")
        f.write("Min Test Loss: " + str(min_test_loss) + "\n")
        f.write("Load Weight: " + str(load_weight) + "\n")
        f.write("Frozen: " + str(frozen) + "\n")
        f.write("\nRun Description: " + run_description + "\n")
        f.close()




if __name__ == "__main__":
    torch.cuda.init()
    torch.cuda.empty_cache()

    # NETWORK PARAMETERS

    network_depth = 4
    latent_dims = 1024
    filter_size = 3
    final_grid_size = 16

    
    # RUN PARAMETERS

    sketch_dir_train = ""
    sketch_dir_test = ""

    batch_size_param = 450
    learning_rate = 5e-5
    total_epochs = 800

    time_stamp = "Gen_1710532434"
    # time_stamp = "Gen_1712888438"
    sketch_points_dir_train = f"SDFDatasets\{time_stamp}_Train"   #need change
    sketch_points_dir_test = f"SDFDatasets\{time_stamp}_Test"    #need change

    # batch_size_param = 450
    # learning_rate = 5e-5
    # total_epochs = 400

    weight_time_stamp = "2024_02_27_17_58_34"

    # run_description = "Train from scratch with single convolutional layer (Cube Dataset)."

    # count = 0
    # for learning_rate in [0.000025, 0.00005, 0.000075]:
    #     for latent_dims in [512, 1024, 1536, 2048]:
    #         run_description = f"{count}th Run"
    #         pre_train(network_depth, latent_dims, filter_size, final_grid_size, sketch_dir_train, sketch_dir_test, batch_size_param, learning_rate, total_epochs, run_description)
    #         count += 1

    # sketch_dir_train = "Sketches/Renders"
    # sketch_dir_test = "Sketches/Renders2_Test"

    # pre_train(network_depth, latent_dims, filter_size, final_grid_size, sketch_dir_train, sketch_dir_test, batch_size_param, learning_rate, total_epochs, run_description)

    # train(network_depth, latent_dims, filter_size, final_grid_size, sketch_points_dir_train, sketch_points_dir_test, weight_time_stamp, batch_size_param, learning_rate, total_epochs, False, False, run_description)

    # count = 0
    # for load_weight, frozen in [(True, True), (True, False),(False, False)]:
    #         for learning_rate in [1e-4, 5e-5, 2.5e-5]:
    #             for i in range(3):
    #                 run_description = f"{i}th run in the {count}th group"
    #                 train(network_depth, latent_dims, filter_size, final_grid_size, sketch_points_dir_train, sketch_points_dir_test, weight_time_stamp, batch_size_param, learning_rate, total_epochs, load_weight, frozen, run_description)
    #             count += 1

    count = 0
    for latent_dims in [128, 256, 512, 1024, 1536]:
        for i in range(3):
            run_description = f"{i+1}th run in the {latent_dims} latent dim group"
            train(network_depth, latent_dims, filter_size, final_grid_size, sketch_points_dir_train, sketch_points_dir_test, weight_time_stamp, batch_size_param, learning_rate, total_epochs, False, False, run_description)
        count += 1