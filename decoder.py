import torch
import torch.nn as nn
from encoder import Encoder


class Decoder(nn.Module):

    """
    Assumes that encoder_vector is of size latent_dim*2 for mu and log_var
    """

    def __init__(self, latent_dim, hidden_dims) -> None:
        super().__init__()

        # initialize class variables
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.decode_input = []

        # temp container for constructing layers
        modules = []

        # hyperparameters
        self.hidden_dim_mult = 4  # This number must be a square
        kernel_size = 3
        stride = 2
        padding = 1
        out_padding = 1

        # construct the first layer of the decoder network
        self.latent_dim = latent_dim
        fc_1 = nn.Linear(self.latent_dim, self.hidden_dims[-1] * self.hidden_dim_mult)
        self.decode_input.append(fc_1)
        self.decode_input == nn.Sequential(*self.decode_input)

        # hidden dims shared with encoder so need to be reversed for decoder
        hidden_dims_reversed = list(self.hidden_dims)
        hidden_dims_reversed.reverse()
        print(hidden_dims_reversed)

        # Construct decoder network to up-sample data
        for i in range(len(hidden_dims_reversed) - 1):
            layer = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=hidden_dims_reversed[i],
                    out_channels=hidden_dims_reversed[i + 1],
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    output_padding=out_padding,
                ),
                nn.BatchNorm2d(hidden_dims_reversed[i + 1]),
            )
            modules.append(layer)

        last_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_dims_reversed[-1],
                out_channels=3,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.Tanh(),
        )

        modules.append(last_layer)

        self.decoder = nn.Sequential(*modules)

    def decode(self, input):
        res = self.decode_input(input)
        res = res.view(
            self.hidden_dims[-1],
            int(self.hidden_dim_mult ** (0.5)),
            int(self.hidden_dim_mult ** (0.5)),
        )
        res = self.decoder(res)
        return res

    # Then we must make a function to sample from our encoder vector
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        sample_term = torch.randn_like(log_var)
        return mu + std * sample_term
        # Sample values from the gaussians characterized by mu and var
        pass

    # The forward function must then reparamterize (sample) from the encoded vector
    # passed in and then pass those values into the upsampling network
    def forward(self, mu, log_var):
        res = self.reparameterize(mu, log_var)
        res = self.decode(res)
        return res
