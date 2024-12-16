"""
GPT3.5 prompt:
can you make a torch Conv2D layer that downsizes an image?

create a UNet implementation using convolution layer based downsizing and upsizing. Make the activation a parameter. It also needs skip connections, and optional batch norm

replace the MaxPool with a convolutional layer. And add a test example

"""
import torch
import torch.nn as nn

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, activation=nn.ELU(), batch_norm=False, groups=1, kernel_size=3):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, groups=groups, padding='same')
        self.activation = activation
        self.batch_norm = batch_norm
        if batch_norm:
            self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        x = self.conv(x)
        if self.batch_norm:
            x = self.bn(x)
        x = self.activation(x)
        return x
    

class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, activation=nn.ReLU(), batch_norm=False):
        super(Encoder, self).__init__()
        self.conv = nn.Sequential(
            Conv(in_channels, out_channels, activation, batch_norm),
            Conv(out_channels, out_channels, activation, batch_norm)
        )
        self.pool = nn.Conv2d(out_channels, out_channels, kernel_size=2, stride=2)
        
    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.pool(x1)
        return x1,x2
    


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channel=True, activation=nn.ReLU(), batch_norm=True):
        super(Decoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.skip_channel = skip_channel

        # https://distill.pub/2016/deconv-checkerboard/ reducing checkerboard artifacts
        #self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2)
        self.conv = nn.Sequential(
            Conv(in_channels*2 if skip_channel else in_channels, out_channels, activation, batch_norm),
            Conv(out_channels, out_channels, activation, batch_norm)
        )
        
    def forward(self, x1, x2=None):
        x1_upscaled = torch.nn.functional.interpolate(x1, scale_factor=2, mode='nearest')
        if x2 is not None:
            assert self.skip_channel
            x = torch.cat([x2, x1_upscaled], dim=1)
        else:
            assert not self.skip_channel
            x = x1_upscaled
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, input_features, layer_features, output_features, activation=nn.ReLU(), batch_norm=True):
        """
        Resize depth is the difference between decoder and encoder depth. resize_depth>0 means the output is smaller than the input
        """
        super(UNet, self).__init__()
        self.inc = Conv(input_features, layer_features[0], activation, batch_norm, kernel_size=1)
        self.depth = len(layer_features)

        encoders = [Encoder(layer_features[i], layer_features[i+1], activation, batch_norm) for i in range(self.depth-1)]
        decoders = [Decoder(layer_features[-i-1], layer_features[-i-2], self.depth-1-i, activation, batch_norm) for i in range(self.depth-1)]

        self.encoders = nn.ModuleList(encoders)
        self.decoders = nn.ModuleList(decoders)

        self.outc = nn.Conv2d(layer_features[0], output_features, kernel_size=1)
        
    def forward(self, x):
        x = self.inc(x)
        outputs = []
        for i in range(len(self.encoders)):
            x1,x = self.encoders[i](x)
            outputs.append(x1)
        for i in range(len(self.decoders)):
            if self.decoders[i].skip_channel:
                x = self.decoders[i](x, outputs[-i-1])
            else:
                x = self.decoders[i](x)
        x = self.outc(x)
        return x

if __name__ == "__main__":
    # Test example
    batch_size = 2
    in_channels = 3
    out_channels = 1
    height = 32
    width = 32
    resize_depth = 2
    # Create a random input tensor
    x = torch.randn(batch_size, in_channels, height, width)

    # Instantiate the UNet model
    model = UNet(in_channels, [16, 20, -16], out_channels)

    # Pass the input tensor through the model
    out = model(x)

    # make sure input and output shapes match
    assert out.shape == (batch_size, out_channels, height, width)

    # Print the output tensor shape
    print("Output shape:", out.shape)
