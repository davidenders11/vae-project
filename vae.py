import torch
import torch.nn as nn
from encoder import Encoder
from decoder import Decoder

# Images from github are of size 178x218, we need to change hidden_dims to work for 64x64


class Basic_VAE(nn.Module):
    def __init__(self, in_dim, hidden_dims, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(in_dim, hidden_dims, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dims)

    def forward(self, input_img):
        mu, log_var = self.encoder(input_img)
        reconstructed_img = self.decoder(mu, log_var)
        return [reconstructed_img, mu, log_var]

    def sample_img(self):
        zeros = torch.zeros((self.latent_dim, 1))
        ones = torch.ones((self.latent_dim, 1))

        normal_input = torch.normal(zeros, ones)
        normal_input = normal_input.view(1, self.latent_dim)
        print(normal_input.shape)
        sampled_img = self.decoder.decode(normal_input)
        return sampled_img
    # The loss must include a measure of the difference between the input image and reconstructed image
    # as well as a measure of the difference between the normals we get from mu and log_var and the standard
    # normal by using kl divergence
