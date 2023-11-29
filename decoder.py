import torch
import torch.nn as nn
import encoder


class Decoder(nn.Module):

    """
    Assumes that encoder_vector is of size latent_dim*2 for mu and log_var
    """

    def __init__(self, latent_dim, hidden_dims) -> None:
        super().__init__()

        self.hidden_dims = hidden_dims

        # The output from encoding will be in a vector of size latent_dim X 1
        # Pass it through a linear layer and output dimension hidden_dims[-1]
        self.decode_input = []
        modules = []

        # hyperparameters

        # This number must be a square
        self.hidden_dim_mult = 4

        kernel_size = 3
        stride = 2
        padding = 1
        out_padding = 1

        # TODO: I think fc_1 and fc_2 maybe don't have to be class variables since they get passed into
        # decode_input which is already one? (I just made it one)
        self.latent_dim = latent_dim
        self.fc_1 = nn.Linear(self.latent_dim, self.hidden_dims[-1] * self.hidden_dim_mult)
    
        self.decode_input.append(self.fc_1)

        hidden_dims_reversed = reversed(hidden_dims)

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

        # Now we have to get the data in our previous format 64x64x3
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
        res = res.view(self.hidden_dims[-1], int(self.hidden_dim_mult**(0.5)), int(self.hidden_dim_mult**(0.5)))
        # I don't think the output of decode_input will be able to fit into the input to decoder
        # so we will have to figure out the dims and  reshape it
        res = self.decoder(res)
        return res

    # Then we must make a function to sample from our encoder vector
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var) # explaining why we use log variance https://chat.openai.com/share/4a6594e7-11b9-44b8-bee2-62f8227e4fa4 
        sample_term = torch.randn_like(log_var)
        return mu + std * sample_term
        # Sample values from the gaussians characterized by mu and var
        pass

    def forward(self, mu, log_var):
        res = self.reparameterize(mu, log_var)
        res = self.decode(res)
        return res
        

    #     encoderObject = encoder.Encoder(in_dim, hidden_dims, latent_dim)
    #     mu, var = encoderObject.encode(input)
    #     sample = self.reparameterize(mu, var)
    #     res = self.decode(sample)
    #     return res

    # The forward function must then reparamterize (sample) from the encoded vector passed in and then pass those values into the upsampling network
