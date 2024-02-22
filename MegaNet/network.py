import math
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

DEFAULT_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Requires i >= 0
def next_highest_power_of_two(i : int):
    if i == 0:
        return 0
    return 1 << (i - 1).bit_length()


def check_input_validity(depth, latent_dims, img_channels, starting_filters, filter_size, 
                         final_channels, final_grid_size):
    if not (type(depth) is int and depth >= 1):
        raise ValueError("Depth must be an integer that is at least 1!")
    
    if not (type(latent_dims) is int and latent_dims >= 1):
        raise ValueError("The number of latent dimensions must be an integer that is at least 1!")
    
    if not (type(img_channels) is int and img_channels >= 1):
        raise ValueError("The number of image channels must be an integer that is at least 1!")

    if final_channels is not None and not (type(final_channels) is int and final_channels >= 1):
        raise ValueError("The number of channels of the final convolutional layer must be either " 
                            + "None or an integer that is at least 1!")
    
    if not (type(starting_filters) is int and starting_filters >= 1):
        raise ValueError("The number of starting filters must be an integer that is at least 1!")
    
    if not (type(filter_size) is int and filter_size >= 1):
        raise ValueError("The filter size must be an integer that is at least 1!")
    
    if not (type(final_grid_size) is int and math.log2(final_grid_size).is_integer()):
            raise ValueError("The final grid sizes must be powers of 2!")


# Lightly modified from this pytorch tutorial:
# https://github.com/Jackson-Kang/Pytorch-VAE-tutorial/tree/master
def reparameterization(mean : Tensor, sd : Tensor, device : torch.device = DEFAULT_DEVICE):
    """
    Samples a Multivariate Gaussian Distribution (MGD) with the specified mean and variance

    mean : torch.Tensor
        The mean of the MGD to be sampled.

    sd : torch.Tensor
        The standard deviation of the MGD to be sampled.

    device : device
        The device to which the result should be saved. By default, the data is saved
        to the GPU if cuda is available, and on the CPU otherwise.

    This reparameterization is used to enable the Variational Autoencoder to backpropogate 
    loss to the mean/variance while still allowing for "non-deterministic" sampling of the
    distribution.        
    """
    epsilon = torch.randn_like(sd).to(device)        # sampling epsilon        
    z = mean + sd * epsilon                          # reparameterization trick
    return z


def preprocess_input(input : Tensor, min_img_size : int, fill_value : float = 0.0) -> Tensor:
    """
    Pre-processes an input tensor with constant-value padding so that it can be properly 
    processed by a network.

    model : VariationalSketchPretrainer or SketchToSDF
        The network for which the input will be pre-processed

    input : torch.Tensor 
        The input image or batch of images to preprocess

    fill_value : float
        The fill value for the padding
    
    If the image is smaller than the minimum image size needed for the network, it is 
    padded up to this size. If the input image has an odd size in height/
    width, it is padded slightly more on the bottom/right than on the top/left.
    """
    size = input.size()

    # Compute target image size
    target_height = max(min_img_size, size[-2]) 
    target_width = max(min_img_size, size[-1])

    # Compute padding size
    pad_height = target_height - size[-2]
    pad_width = target_width - size[-1]

    top = bottom = pad_height // 2
    if pad_height % 2 != 0:
        bottom += 1

    left = right = pad_width // 2
    if pad_width % 2 != 0:
        right += 1

    return torch.nn.functional.pad(input, pad=(left, right, top, bottom), value=fill_value)

def load_VSP(network_dir, weights_string="FinalWeights", device=DEFAULT_DEVICE):
    if network_dir[-1] != "/" and network_dir[-1] != "\\":
        network_dir += "/"
    try:
        with open(network_dir + "network_parameters.pkl", 'rb') as f:
            loaded_dict = pickle.load(f)
            network = VariationalSketchPretrainer(depth=loaded_dict["depth"], 
                                           latent_dims=loaded_dict["latent_dims"],
                                           img_channels=loaded_dict["img_channels"],
                                           starting_filters=loaded_dict["starting_filters"],
                                           filter_size=loaded_dict["filter_size"],
                                           final_channels=loaded_dict["final_channels"],
                                           final_grid_size=loaded_dict["final_grid_size"],
                                           device=device)
    
    except Exception as e:
        print("Failed to load network parameters at \'" + network_dir + "\'")
        print(e)
        return None
    
    weights_path = network_dir + "ModelWeights/" + weights_string + ".pt"

    try:
        network.load_state_dict(torch.load(weights_path))
        return network
    except Exception as e:
        print("Failed to load network weights at \'" + weights_path + "\'")
        print(e)
        return None 

class VariationalSketchPretrainer(nn.Module):
    def __init__(self, depth=4, latent_dims=256, img_channels=1, starting_filters=32, filter_size=5, 
                 final_channels : int = None, final_grid_size=16, device=torch.device('cuda')):
        """
        Parameters
        -----------
        depth : int
            The number of convolutional layers of the network (defaults to 4).
        latent_dims : int
            The number of dimensions of the latent space created from 
            the result of the convolutional block (defaults to 256).
        img_channels : int 
            The number of channels of the input image (defaults to 1).
        starting_filters : int
            The number of filters for the first convolutional layer (defaults to 32).
            All subsequent convolutional layers double this number (unless final_channels
            is specified, see below)
        filter_size : int
            The size of the kernel for each filter (defaults to 3).
        final_channels : int
            The number of channels for the final convolutional layer, 
            if set to None, the number of channels is twice that of
            the previous layer (defaults to None).
        final_grid_size : int
            The size of the grid used for max_pooling after the final convolutional layer.
            Must be a power of 2.
        device : device
            The device on which the network should be stored. By default, the network is stored
            on the GPU if cuda is available, and on the CPU otherwise.
        """
        super(VariationalSketchPretrainer, self).__init__()

        check_input_validity(depth, latent_dims, img_channels, starting_filters, filter_size, 
                             final_channels, final_grid_size)

        # Copy inputs to instance variables
        self.depth = depth
        self.latent_dims = latent_dims
        self.img_channels = img_channels
        self.starting_filters = starting_filters
        self.final_channels = final_channels
        self.final_grid_size = final_grid_size
        self.filter_size = filter_size
        self.device = device

        # Compute minimum image size
        self.min_img_size = (2**depth) * final_grid_size
        
        # Set gaussian likelihood scale variable
        # self.scale = nn.Parameter(torch.tensor([0.0], device=DEFAULT_DEVICE))
        
        # Set up encoding convolutional layers
        self.convLayers = nn.ModuleList()
        channels_in = img_channels
        channels_out = starting_filters
        for layer in range(depth):
            if layer == depth - 1:
                if final_channels is not None:
                    channels_out = final_channels
                else:
                    self.final_channels = channels_out
            self.convLayers.append(nn.Conv2d(in_channels=channels_in, 
                                             out_channels=channels_out, 
                                             kernel_size=filter_size, 
                                             stride=(2,2), padding=(1,1), 
                                             device=self.device))
            if layer != depth - 1:
                channels_in = channels_out
                channels_out *= 2

        # Set up latent space conversion
        self.poolToLatentDistibution = nn.Linear(final_grid_size * final_grid_size * channels_out, latent_dims*2, device=self.device)

        # Set up latent space back-conversion
        self.latentSampleToPool = nn.Linear(latent_dims, final_grid_size * final_grid_size * channels_out, device=self.device)

        # Set up decoding convolutional layers
        self.deconvLayers = nn.ModuleList()
        channels_in = self.convLayers[-1].out_channels
        channels_out = self.convLayers[-1].in_channels
        for layer in range(depth):
            if layer == depth - 1:
                channels_out = img_channels
            self.deconvLayers.append(nn.ConvTranspose2d(in_channels=channels_in, 
                                                        out_channels=channels_out, 
                                                        kernel_size=filter_size, 
                                                        stride=(2,2), padding=(1,1), 
                                                        output_padding=(1,1), device=self.device))
            channels_in = channels_out
            channels_out = int(channels_out/2)
        

    def forward(self, x : torch.Tensor):
        x = preprocess_input(x, self.min_img_size, 0)

        # Compute pooling size ratios for later
        poolRatioV = x.size()[-1] // self.min_img_size
        poolRatioH = x.size()[-2] // self.min_img_size

        # Forward propagate
        for layer in self.convLayers:
            x = F.relu(layer(x))
    
        # Compute unpooling parameters for later
        unPoolDims = x.size()
        unPoolStride = (poolRatioV, poolRatioH)
        unPoolKernelSize = (unPoolDims[-2] - (self.final_grid_size-1) * poolRatioV, 
                            unPoolDims[-1] - (self.final_grid_size-1) * poolRatioH) 

        # Apply max pooling to reduce dimensionality to a known fixed size
        # x, indices = torch.nn.functional.adaptive_max_pool2d_with_indices(x, (self.final_grid_size, self.final_grid_size), True)

        # Save current tensor size for reshaping later
        decodeReshapeSize = x.size()

        # Flatten the tensor in the last three dimensions to use with the linear layer
        x = x.flatten(start_dim=-3)

        # Forward propagate to get the latent distribution parameters
        x = self.poolToLatentDistibution(x)

        # Split the result into the mean and log variance
        mean, log_var = torch.split(x, x.shape[-1]//2, dim=-1)

        # Sample the latent distribution for a latent vector
        z = reparameterization(mean, torch.exp(0.5*log_var), self.device)

        # Decoder half begins here

        # Forward propagate
        x = self.latentSampleToPool(z)
        
        # Reshape tensor to regain the last three dimensions (tensor format is now '... x C x H x W')
        x = x.reshape(decodeReshapeSize)
    
        # Apply max_unpooling to partially reverse the pooling done in the encoder
        # x = torch.nn.functional.max_unpool2d(x, indices, stride=unPoolStride, kernel_size=unPoolKernelSize, padding=0, output_size=unPoolDims)

        # Forward propagate
        for layer in self.deconvLayers:
            x = F.relu(layer(x))

        # Return the final tensor and the estimated latent distribution parameters
        return torch.sigmoid(x), z, mean, log_var
    
    
    # Taken from this PyTorch tutorial
    # https://www.kaggle.com/code/maunish/training-vae-on-imagenet-pytorch
    """
    def kl_loss(self,z,mean,std):
        p = torch.distributions.Normal(torch.zeros_like(mean, device=DEFAULT_DEVICE),
                                       torch.ones_like(std, device=DEFAULT_DEVICE))
        q = torch.distributions.Normal(mean,torch.exp(std/2))

        log_pz = p.log_prob(z)
        log_qzx = q.log_prob(z)

        kl_loss = (log_qzx - log_pz)
        kl_loss = kl_loss.sum(-1)
        return kl_loss

    # Modified from this PyTorch tutorial
    # https://www.kaggle.com/code/maunish/training-vae-on-imagenet-pytorch
    def gaussian_likelihood(self, inputs : Tensor, outputs : Tensor, scale : Tensor):
        dist = torch.distributions.Normal(outputs,torch.exp(scale))
        log_pxz = dist.log_prob(inputs)
        return log_pxz.sum(dim=(1,2,3))

    # Modified from this PyTorch tutorial
    # https://www.kaggle.com/code/maunish/training-vae-on-imagenet-pytorch
    def loss_fn(self,inputs,outputs,z,mean,logVar):
        std = torch.exp(logVar/2)
        kl_loss = self.kl_loss(z,mean,std)
        rec_loss = self.gaussian_likelihood(inputs,outputs,self.scale)

        return torch.mean(kl_loss - rec_loss)
    """

    def bce_loss_fn(self, image_batch, reprojections, mu, logVar):
        reconstruction_loss = nn.functional.binary_cross_entropy(reprojections, image_batch, reduction='sum')
        kl_divergence_loss = -0.5 * torch.sum(1 + logVar - mu.pow(2) - logVar.exp())
        return (reconstruction_loss + kl_divergence_loss)

    def get_network_param_dict(self):
        # Dictionary of parameters for saving/loading purposes
        return {
            "depth":self.depth,
            "latent_dims":self.latent_dims,
            "img_channels": self.img_channels,
            "starting_filters":self.starting_filters,
            "filter_size":self.filter_size,
            "final_channels":self.final_channels,
            "final_grid_size":self.final_grid_size
        }

    def to_str(self):
        s = ""
        
        for layer in self.convLayers:
            s += str(layer) + "\n"

        s += "PoolToSize " + str(self.final_grid_size) + "x" + \
                             str(self.final_grid_size) + "x" + \
                             str(self.final_channels) + "\n"
        
        s += str(self.poolToLatentDistibution) + "\n"
        s += str("Reparameterization Step\n")
        s += str(self.latentSampleToPool) + "\n"

        s += "UnPoolFromSize " + str(self.final_grid_size) + "x" \
                               + str(self.final_grid_size) + "x" \
                               + str(self.final_channels) + "\n"
        for layer in self.deconvLayers:
            s += str(layer) + "\n"

        return s
        

    def print_layers(self):
        print("Convolution Block:\n============================")
        for layer in self.convLayers:
            print(layer)

        print("\nAdaptive Pooling Layer: " + str(self.final_grid_size) + "x" \
                                           + str(self.final_grid_size) + "x" \
                                           + str(self.final_channels) + "\n")

        print(self.poolToLatentDistibution, "\n")
        
        print("Reparameterization Step\n")

        print(self.latentSampleToPool, "\n")
        
        print("Unpooling Layer: " + str(self.final_grid_size) + "x" \
                                  + str(self.final_grid_size) + "x" \
                                  + str(self.final_channels) + "\n")

        print("Deconvolution Block:\n============================")
        for layer in self.deconvLayers:
            print(layer)
        


class SketchToSDF(nn.Module):
    def __init__(self, depth=4, latent_dims=256, img_channels=1, starting_filters=32, 
                 filter_size=3, final_channels:int = None, final_grid_size=16, device=DEFAULT_DEVICE):
        """
        Parameters
        -----------
        depth : int
            The number of convolutional layers of the network (defaults to 4).
        latent_dims : int
            The number of dimensions of the latent space created from 
            the result of the convolutional block (defaults to 256).
        img_channels : int 
            The number of channels of the input image (defaults to 1).
        starting_filters : int
            The number of filters for the first convolutional layer (defaults to 32).
            All subsequent convolutional layers double this number (unless final_channels
            is specified, see below)
        filter_size : int
            The size of the kernel for each filter (defaults to 3).
        final_channels : int
            The number of channels for the final convolutional layer, 
            if set to None, the number of channels is twice that of
            the previous layer (defaults to None).
        final_grid_size : int
            The size of the grid used for max_pooling after the final convolutional layer.
            Must be a power of 2.
        """
        super(SketchToSDF, self).__init__()


        check_input_validity(depth, latent_dims, img_channels, starting_filters, filter_size, 
                             final_channels, final_grid_size)

        # Set instance variables
        self.depth = depth
        self.latent_dims = latent_dims
        self.img_channels = img_channels
        self.starting_filters = starting_filters
        self.final_channels = final_channels
        self.final_grid_size = final_grid_size
        self.filter_size = filter_size
        self.device = device

        # Compute minimum image size
        self.min_img_size = (2**depth) * final_grid_size

        # Set up encoding convolutional layers
        self.convLayers = nn.ModuleList()
        channels_in = img_channels
        channels_out = starting_filters
        for layer in range(depth):
            if layer == depth - 1 and final_channels is not None:
                channels_out = final_channels
            self.convLayers.append(nn.Conv2d(in_channels=channels_in, 
                                             out_channels=channels_out, 
                                             kernel_size=filter_size, 
                                             stride=(2,2), padding=(1,1), 
                                             device=device))
            if layer != depth - 1:
                channels_in = channels_out
                channels_out *= 2

        # Set up latent space conversion
        self.poolToLatentDistibution = nn.Linear(final_grid_size * final_grid_size * channels_out, latent_dims*2, device=device)
    
    def forward(self, x):
        return




        