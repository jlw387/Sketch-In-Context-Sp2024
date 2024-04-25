import numpy as np

import torch
from torch.utils.data import DataLoader

from MegaNet.network import SketchToSDF
import dataset

time_stamp = "Gen_1709917339"
sketch_points_dir_test = f"SDFDatasets\{time_stamp}\Train"

test_dataset = dataset.SketchPointDataset(sketch_points_dir_test, 1, device=torch.device('cpu'))
test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False)

network = SketchToSDF(loss="mse_tanh", device=torch.device('cpu'))

num_examples = 0
ones_loss = 0
zeros_loss = 0
neg_ones_loss = 0

for idx, (image_batch, points_batch, sds_batch) in enumerate(test_dataloader):
    batch_size = image_batch.shape[0]
    
    baseline_ones = torch.ones_like(sds_batch)
    baseline_zeros = torch.zeros_like(sds_batch)
    baseline_neg_ones = -torch.ones_like(sds_batch)

    ones_loss += network.compute_loss(baseline_ones, sds_batch)
    zeros_loss += network.compute_loss(baseline_zeros, sds_batch)
    neg_ones_loss += network.compute_loss(baseline_neg_ones, sds_batch)

    num_examples += batch_size
    


print("Ones baseline: ", ones_loss / num_examples)
print("Zeros baseline: ", zeros_loss / num_examples)
print("Negative ones baseline: ", neg_ones_loss / num_examples)