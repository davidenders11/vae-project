import torch
import torch.nn as nn
from encoder import Encoder
from decoder import Decoder

# Images from github are of size 178x218, we need to change hidden_dims to work for 64x64


class Basic_VAE(nn.Module):
    def __init__(self, in_dim, hidden_dims, latent_dim):
        super().__init__()
        self.encoder = Encoder(in_dim, hidden_dims, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dims)

    def forward(self, input_img):
        mu, log_var = self.encoder(input_img)
        reconstructed_img = self.decoder(mu, log_var)
        return reconstructed_img

    # The loss must include a measure of the difference between the input image and reconstructed image
    # as well as a measure of the difference between the normals we get from mu and log_var and the standard
    # normal by using kl divergence
