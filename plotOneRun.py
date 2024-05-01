import numpy as np
from matplotlib import pyplot as plt

model_type = "SDF"

timestamp = "2024_04_29_15_31_32"
title = "Hybrid Network Performance (Zero Spatial Channels, Jitter=6)"

plt.xlabel("Epoch")
plt.ylabel("ELBO" if model_type == "CVAE" else "Mean Squared Tanh Error")

train_txt_label = "TrainingElbos.txt" if model_type == "CVAE" else "TrainingLosses.txt"
test_txt_label = "TestElbos.txt" if model_type == "CVAE" else "TestLosses.txt"


try:
    training_losses = np.loadtxt("./Models/" + model_type + "/" + timestamp + "/" + train_txt_label)
except Exception as e:
    training_losses = np.loadtxt("./Models/" + model_type + "/" + timestamp + "/TrainingMSEs.txt")

try:
    test_losses = np.loadtxt("./Models/" + model_type + "/" + timestamp + "/" + test_txt_label)
except Exception as e:
    test_losses = np.loadtxt("./Models/" + model_type + "/" + timestamp + "/TestMSEs.txt")

plt.plot(training_losses, label=timestamp + "_Train")
plt.plot(test_losses, label=timestamp + "_Test")

if model_type == "SDF":
    ones_baseline = np.full_like(training_losses, 0.2701)
    zeros_baseline = np.full_like(training_losses, 0.1559)
    # neg_ones_baseline = np.full_like(training_losses, 1.2017)

    #plt.plot(ones_baseline, '--', label="Baseline - Ones")
    #plt.plot(zeros_baseline, '--', label="Baseline - Zeros")
    # plt.plot(neg_ones_baseline, '--', label="Baseline - Negative Ones")

plt.title(title)
plt.xlim((0,800))
plt.ylim((0,0.0165))
plt.legend()
plt.show()

