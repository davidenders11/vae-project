import torch.nn as nn

class Decoder(nn.Module):

    """
    Assumes that encoder_vector is of size latent_dim*2 for mu and var
    """
    def __init__(self, latent_dim, hidden_dims) -> None:
        super().__init__()

        # First we must make the network to upsample from our encoding using nn.ConvTranspose2D
            # The encoding will be in a vector of size latent_dim X 1, so we should pass it through a linear layer then put it in the of hidden_dims[-1]

        # 
        decode_input = []
        modules = []

        latent_dim_mult = 4
        kernel_size = 3
        stride = 2
        padding = 1
        out_padding=1

        self.latent_dim = latent_dim
        self.fc_1 = nn.Linear(self.latent_dim, self.latent_dim*latent_dim_mult)
        self.fc_2 = nn.Linear(self.latent_dim*latent_dim_mult, self.latent_dim*latent_dim_mult)

        decode_input.append(self.fc_1)
        decode_input.append(self.fc_2)

        hidden_dims_reversed = reversed(hidden_dims)

        # Up sample data 
        for i in range(len(hidden_dims_reversed)-1):
            layer = nn.Sequential(
                nn.ConvTranspose2d(in_channels=hidden_dims_reversed[i],
                                   out_channels=hidden_dims_reversed[i+1],
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=padding,
                                   output_padding=out_padding),
                nn.BatchNorm2d(hidden_dims_reversed[i+1])
            )
            modules.append(layer)

        # Now we have to get the data in our previous format 64x64x3
        last_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_dims_reversed[-1],
                out_channels=3,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding),
            nn.Tanh()
        )

        modules.append(last_layer)
        
        self.decoder = nn.Sequential(*modules)
        
    def decode(self, input):
        res = self.decode_input(input)
        # I don't think the output of decode_input will be able to fit into the input to decoder
        # so we will have to figure out the dims and  reshape it
        res = self.decoder(res)
        return res

    # Then we must make a function to sample from our encoder vector
    def reparameterize(self, mu, var):
        # Sample values from the gaussians characterized by mu and var
        pass

    # def forward(self, input):
        

    # The forward function must then reparamterize (sample) from the encoded vector passed in and then pass those values into the upsampling network 