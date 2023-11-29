import torch.nn as nn
from encoder import Encoder


class Decoder(nn.Module):

    """
    Assumes that encoder_vector is of size latent_dim*2 for mu and var
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
        latent_dim_mult = 4
        kernel_size = 3
        stride = 2
        padding = 1
        out_padding = 1

        # initialize ____ with _____ (not sure what this does lol)
        fc_1 = nn.Linear(self.latent_dim, self.latent_dim * latent_dim_mult)
        fc_2 = nn.Linear(self.latent_dim * latent_dim_mult, self.hidden_dims[-1])
        self.decode_input.append(fc_1)
        self.decode_input.append(fc_2)

        # hidden dims shared with encoder so need to be reversed for decoder
        hidden_dims_reversed = reversed(self.hidden_dims)

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

        # Now we have to get the data in our previous format 64x64x3 TODO: should dimensions be 2D?
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
        # I don't think the output of decode_input will be able to fit into the input to decoder
        # so we will have to figure out the dims and  reshape it
        # shouldn't fc_2 output_dims instead be equal to hidden_dims[-1]? that way it would fit
        res = self.decoder(res)
        return res

    # Then we must make a function to sample from our encoder vector
    def reparameterize(self, mu, var):
        standard = nn.exp(0.5 * var)
        epsilon = nn.randn_like(standard)
        return epsilon * standard + mu

    def forward(self, input):
        # I think to get in_dim we have to flatten the 64x64x3 to 64x192 or something like that - in_dim should be 2d i believe
        # hidden_dims and latent_dim should be a hyperparameter chosen in the parent class VariationalAutoEncoder
        encoder = Encoder(input.shape, self.hidden_dims, self.latent_dim)
        mu, var = encoder.encode(input)
        sample = self.reparameterize(mu, var)
        res = self.decode(sample)
        return res
