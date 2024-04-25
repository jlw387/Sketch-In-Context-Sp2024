import numpy as np
import matplotlib.pyplot as plt


iterations_data = np.load("IterationBehavior.npy")

img_dim = int(np.sqrt(iterations_data.shape[-1]))

iterations_data = iterations_data.reshape((iterations_data.shape[0], img_dim, img_dim))

stop_index = np.where(np.abs(iterations_data).sum(axis=2).sum(axis=1) == 0)[0][0]
print(stop_index)

while True:
    threshold_str = input("Enter threshold (q to quit): ")

    if threshold_str == 'q':
        break
    try:
        threshold = float(threshold_str)

        img = plt.imshow(iterations_data[0], vmin=-threshold, vmax=threshold)
        plt.colorbar()
        plt.title("SD Evaluations")
        plt.pause(0.005)

        for iter in range(stop_index):
            img.set_data(iterations_data[iter,:])
            plt.title("SD Evaluations Iteration " + str(iter))
            plt.pause(0.005)

    except Exception as e:
        print("Couldn't run visualizer due to error:")
        print(e)

