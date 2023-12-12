import torch
import torch.nn as nn


class Encoder(nn.Module):

    """
    Latent dim is the number of means there are. Note that number of means = number of variance

    Keep hidden_dims as a list so in the vae model we can pass the same hidden_dims into both the encoder and decoder

    However, this module should output a vector thats the size of latent_dim
    """

    def __init__(self, in_dim, hidden_dims, latent_dim) -> None:
        super().__init__()
        # hyperparameters
        kernel_size = 5
        stride = 2
        padding = 1
        # This number must equal HxW of the final output convolution from the encoded layer, for example, ours is 128x3x3 where 3x3 is HxW and 9 = 3*3
        conv_to_fc_mult = 9
        
        # temp container for layer construction
        modules = []

        # build encoder layers with dimensions in hidden_dims
        for i in range(len(hidden_dims)):
            out_dim = hidden_dims[i]
            layer = nn.Sequential(
                # Conv2d takes a 4d tensor as input, NxCxHxW, where N is the batch size,
                # C is the number of channels, and H and W are the height and width of the data
                nn.Conv2d(
                    in_channels=in_dim,
                    out_channels=out_dim,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                ),
                nn.BatchNorm2d(out_dim),
                nn.LeakyReLU(),
            )
            modules.append(layer)
            in_dim = out_dim

        self.encoder = nn.Sequential(*modules)

        # We might need to find a way to force the gaussians to have a mean of 0
        # and a std of 1

        # create layer for finding mu vector
        self.fc_mu = nn.Sequential(
            nn.Linear(hidden_dims[-1] * conv_to_fc_mult, latent_dim)
        )

        # create layer for finding sigma vector
        self.fc_var = nn.Sequential(
            nn.Linear(hidden_dims[-1] * conv_to_fc_mult, latent_dim)
        )

    def encode(self, input_img):
        # Transforms the input_img into a latent distribution
        # Returns a tuple of mu vector and sigma vector
        encoded = self.encoder(input_img)
        # print("encoded.shape:", encoded.shape)
        encoded = torch.flatten(encoded, start_dim=1)
        # print("encoded.shape after flatten:", encoded.shape)
        mu = self.fc_mu(encoded)
        var = self.fc_var(encoded)
        return (mu, var)

    def forward(self, input_img):
        return self.encode(input_img)
