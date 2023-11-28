import torch.nn as nn

class Encoder(nn.Module):

    """
    Latent dim is the number of means there are. Note that number of means = number of variance

    Keep hidden_dims as a list so in the vae model we can pass the same hidden_dims into both the encoder and decoder

    However, this module should output a vector thats the size of latent_dim
    """
    def __init__(self, in_dim, hidden_dims, latent_dim) -> None:
        super().__init__()
        kernel_size=5
        stride=2
        padding=1

        conv_to_fc_mult = 3
        latent_dim_mult = 3

        modules = []

        for i in range(len(hidden_dims)):
            out_dim = hidden_dims[i]
            layer = nn.Sequential(
                nn.Conv2d(in_channels=in_dim,
                          out_channels=out_dim,
                          kernel_size=kernel_size, 
                          stride=stride, 
                          padding=padding),
                nn.BatchNorm2d(out_dim),
                nn.LeakyReLU()
            )
            modules.append(layer)
            in_dim = out_dim
    
        self.encoder = nn.Sequential(*modules)

        # We might need to find a way to force the gaussians to have a mean of 0
        # and a std of 1
        self.fc_mu = nn.Sequential(
            nn.Linear(hidden_dims[-1]*conv_to_fc_mult, latent_dim)
        )

        self.fc_var = nn.Sequential(
            nn.Linear(hidden_dims[-1]*conv_to_fc_mult, latent_dim)
        )

    def encode(self, input):
        # Transforms the input into a latent distribution
        # Returns a tuple of mu vector and sigma vector

        encoded = self.encoder(input)
        mu = self.fc_mu(encoded)
        sigma = self.fc_var(encoded)

        return (mu,sigma)
    
        
