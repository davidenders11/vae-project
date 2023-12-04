import torch
import torch.nn as nn
import torch.nn.functional as F
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
    def loss_function(self, reconstructed_img, input_img, mu, log_var, kld_weight=2):
        img_loss = F.mse_loss(reconstructed_img, input_img)
        # article on calculating kl divergence between 2 gaussians:
        # https://medium.com/@outerrencedl/variational-autoencoder-and-a-bit-kl-divergence-with-pytorch-ce04fd55d0d7 
        kld_loss = torch.mean(torch.sum(-log_var +  (log_var.exp()**2-mu**2)/2 - 1/2))
        kld_loss *= kld_weight

        return img_loss + kld_loss
