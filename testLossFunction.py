from PIL import Image

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

def run_loss_tests(input, target):
    print("MSE Loss is:", F.mse_loss(input, target, reduction='sum'))
    # print("CE Loss is:", F.cross_entropy(input, target, reduction='sum'))
    # print("MML Loss is:", F.multilabel_margin_loss(input, target, reduction='sum'))
    print("MSML Loss is:", F.multilabel_soft_margin_loss(input, target, reduction='sum'))

print("TESTING SINGLE IMAGE\n===================")

white_path = "./Testing/white.png"
sketch_path = "./Testing/sketch1.png"

white_image = Image.open(white_path).convert("RGB")
sketch_image = Image.open(sketch_path).convert("RGB")

white_image.load()
sketch_image.load()

white_pixels = list(white_image.getdata())
print("White Mode:", white_pixels[:10])

white_tensor = transforms.Grayscale(1)(transforms.ToTensor()(white_image))
sketch_tensor = transforms.Grayscale(1)(transforms.ToTensor()(sketch_image))

diff_tensor = white_tensor - sketch_tensor

print("White Tensor:", white_tensor.size())
print("Sketch Tensor:", sketch_tensor.size())

print("")

print("White Min:", white_tensor.min())
print("White Max:", white_tensor.max())
print("White Total Mag:", torch.sum(white_tensor))

print("Sketch Min:", sketch_tensor.min())
print("Sketch Max:", sketch_tensor.max())
print("Sketch Total Mag:", torch.sum(sketch_tensor))

print("Diff Min:", diff_tensor.min())
print("Diff Max:", diff_tensor.max())
print("Diff Total Mag^2:", torch.sum(diff_tensor * diff_tensor))

print("")

run_loss_tests(white_tensor, sketch_tensor)

white_tensor = white_tensor.unsqueeze(0)
sketch_tensor = sketch_tensor.unsqueeze(0)

print("\n")

print("White Tensor:", white_tensor.size())
print("Sketch Tensor:", sketch_tensor.size())

print("")

run_loss_tests(white_tensor, sketch_tensor)



print("\n\nTESTING MULTIPLE IMAGES\n===================")

sketches = ["sketch1.png", "sketch2.png"]
num_sketches = len(sketches)

sketch_paths = []
for sketch in sketches:
    sketch_paths.append("./Testing/" + sketch)

sketch_images = []
for sketch_path in sketch_paths:
    sketch_images.append(Image.open(sketch_path))


sketch_tensor = torch.zeros((num_sketches, 1, sketch_images[0].size[1], sketch_images[0].size[0]))
counter = 0
for sketch_image in sketch_images:
    sketch_t = transforms.Grayscale(1)(transforms.ToTensor()(sketch_image))
    sketch_tensor[counter, 0, :, :] = sketch_t
    counter += 1

white_tensor = white_tensor.repeat_interleave(num_sketches,dim=0)

print("White Tensor:", white_tensor.size())
print("Sketch Tensor:", sketch_tensor.size())

print("")

run_loss_tests(white_tensor, sketch_tensor)