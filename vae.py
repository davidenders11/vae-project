import torch
import torch.nn as nn
import torch.nn.functional as F
from . import encoder
from . import decoder

# Images from github are of size 178x218, we need to change hidden_dims to work for 64x64

class Basic_VAE(nn.module):

    def __init__(self, in_dim, hidden_dims, latent_dim):
        super().__init__()
        self.encoder = encoder(in_dim, hidden_dims, latent_dim)

        self.decoder = decoder(latent_dim, hidden_dims)
    
    def forward(self, input_img):
        mu, log_var = self.encoder(input_img)
        reconstructed_img = self.decoder(mu, log_var)
        return reconstructed_img

    # The loss must include a measure of the difference between the input image and reconstructed image
    # as well as a measure of the difference between the normals we get from mu and log_var and the standard
    # normal by using kl divergence
    def loss_function(self, reconstructed_img, input_img, mu, log_var, kld_weight=2):

        img_loss = F.mse_loss(reconstructed_img, input_img)
        kld_loss = 1 # gotta figure out how to compute this
        kld_loss *= kld_weight

        return img_loss + kld_loss

hidden_dims = [16, 32, 64, 128]
vae = Basic_VAE(64, )