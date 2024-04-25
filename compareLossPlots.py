import numpy as np
from matplotlib import pyplot as plt

model_type = "SDF"

# frozen_runs = [
#     # "2024_03_25_17_50_46",
#     # "2024_03_25_18_08_29",
#     # "2024_03_25_18_26_12",
#     # "2024_03_25_18_43_51",
#     # "2024_03_25_19_01_26",
#     # "2024_03_25_19_19_04",
#     # "2024_03_25_19_36_50",
#     # "2024_03_25_19_54_32",
#     # "2024_03_25_20_12_18"
#     ]

# unfrozen_runs = [
#     # "2024_03_25_20_29_59",
#     # "2024_03_25_21_04_38",
#     # "2024_03_25_21_39_06",
#     # "2024_03_25_22_13_32",
#     # "2024_03_25_22_47_57",
#     # "2024_03_25_23_22_24",
#     # "2024_03_25_23_56_51",
#     # "2024_03_26_00_31_20",
#     # "2024_03_26_01_05_53"
#     ]

scratch_runs = [
    # "2024_03_26_01_40_32",
    # "2024_03_26_02_15_03",
    # "2024_03_26_02_49_26",
    # "2024_03_26_03_23_55",
    # "2024_03_26_03_58_24",
    # "2024_03_26_04_32_55",
    # "2024_03_26_05_07_29",
    # "2024_03_26_05_42_18",
    # "2024_03_26_06_16_54"
]

# curriculum_learning_runs = [
#     "2024_04_12_10_45_04"
# ]

lr_5e5_runs = [
    "2024_03_26_03_23_55", 
    "2024_03_26_03_58_24",
    "2024_03_26_04_32_55", 
    "2024_03_25_23_22_24",
    "2024_03_25_22_47_57",
    "2024_03_25_22_13_32",
    "2024_03_25_18_43_51",
    "2024_03_25_19_01_26",
    "2024_03_25_19_19_04"
]

plt.xlabel("Epoch")
plt.ylabel("ELBO" if model_type == "CVAE" else "Mean Squared Tanh Error")

train_txt_label = "TrainingElbos.txt" if model_type == "CVAE" else "TrainingLosses.txt"
test_txt_label = "TestElbos.txt" if model_type == "CVAE" else "TestLosses.txt"

for r in lr_5e5_runs:
    try:
        training_losses = np.loadtxt("./Models/" + model_type + "/" + r + "/" + train_txt_label)
    except Exception as e:
        training_losses = np.loadtxt("./Models/" + model_type + "/" + r + "/TrainingMSEs.txt")
    
    try:
        test_losses = np.loadtxt("./Models/" + model_type + "/" + r + "/" + test_txt_label)
    except Exception as e:
        test_losses = np.loadtxt("./Models/" + model_type + "/" + r + "/TestMSEs.txt")

    plt.plot(training_losses, label=r + "_Train")
    plt.plot(test_losses, label=r + "_Test")

if model_type == "SDF":
    ones_baseline = np.full_like(training_losses, 0.2701)
    zeros_baseline = np.full_like(training_losses, 0.1559)
    # neg_ones_baseline = np.full_like(training_losses, 1.2017)

    plt.plot(ones_baseline, '--', label="Baseline - Ones")
    plt.plot(zeros_baseline, '--', label="Baseline - Zeros")
    # plt.plot(neg_ones_baseline, '--', label="Baseline - Negative Ones")

plt.title("SDF Trained from single cube LR =5e-5")
plt.legend()
plt.show()

