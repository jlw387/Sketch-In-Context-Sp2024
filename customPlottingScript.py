import numpy as np
from matplotlib import pyplot as plt

model_type = "SDF"

# runs = [
#     "2024_02_27_17_58_34",
#     "2024_03_20_02_01_24"
#     ]

# runs = [
#     ["2024_03_25_17_50_46",
#     "2024_03_25_18_08_29",
#     "2024_03_25_18_26_12"],
#     ["2024_03_25_18_43_51",
#     "2024_03_25_19_01_26",
#     "2024_03_25_19_19_04"],
#     ["2024_03_25_19_36_50",
#     "2024_03_25_19_54_32",
#     "2024_03_25_20_12_18"]
#     ]

# runs = [
#     ["2024_03_25_20_29_59",
#     "2024_03_25_21_04_38",
#     "2024_03_25_21_39_06"],
#     ["2024_03_25_22_13_32",
#     "2024_03_25_22_47_57",
#     "2024_03_25_23_22_24"],
#     ["2024_03_25_23_56_51",
#     "2024_03_26_00_31_20",
#     "2024_03_26_01_05_53"]
#     ]

# runs = [
#     ["2024_03_26_01_40_32",
#     "2024_03_26_02_15_03",
#     "2024_03_26_02_49_26"],
#     ["2024_03_26_03_23_55",
#     "2024_03_26_03_58_24",
#     "2024_03_26_04_32_55"],
#     ["2024_03_26_05_07_29",
#     "2024_03_26_05_42_18",
#     "2024_03_26_06_16_54"]
# ]

runs = [
    ["2024_03_25_18_43_51",
    "2024_03_25_19_01_26",
    "2024_03_25_19_19_04"],
    ["2024_03_25_22_13_32",
    "2024_03_25_22_47_57",
    "2024_03_25_23_22_24"],
    ["2024_03_26_03_23_55",
    "2024_03_26_03_58_24",
    "2024_03_26_04_32_55"]
]

plt.xlabel("Epoch")
plt.ylabel("ELBO" if model_type == "CVAE" else "Mean Squared Tanh Error")

train_txt_label = "TrainingElbos.txt" if model_type == "CVAE" else "TrainingLosses.txt"
test_txt_label = "TestElbos.txt" if model_type == "CVAE" else "TestLosses.txt"

types = ["Frozen", "Unfrozen", "Scratch"]

lr_count = 0
for lr in runs:
    total_training_losses = None
    total_test_losses = None
    run_count = 0
    for r in lr:
        training_losses = np.loadtxt("./Models/" + model_type + "/" + r + "/" + train_txt_label)
        if total_training_losses is None:
            total_training_losses = np.zeros(training_losses.shape)
        total_training_losses += training_losses
    
        test_losses = np.loadtxt("./Models/" + model_type + "/" + r + "/" + test_txt_label)
        if total_test_losses is None:
            total_test_losses = np.zeros(test_losses.shape)
        total_test_losses += test_losses
        run_count += 1 

    average_training_losses = total_training_losses / run_count
    average_test_losses = total_test_losses / run_count

    plt.plot(average_training_losses, label=types[lr_count] + "_Train")
    plt.plot(average_test_losses, label=types[lr_count] + "_Test")

    lr_count += 1

if model_type == "SDF":
    ones_baseline = np.full_like(training_losses, 0.2701)
    zeros_baseline = np.full_like(training_losses, 0.1559)
    # neg_ones_baseline = np.full_like(training_losses, 1.2017)

    plt.plot(ones_baseline, '--', label="Baseline - Ones")
    plt.plot(zeros_baseline, '--', label="Baseline - Zeros")
    # plt.plot(neg_ones_baseline, '--', label="Baseline - Negative Ones")

plt.title("Average SDF Network Performance (Learning Rate = 5e-5)")
plt.legend()
plt.show()

