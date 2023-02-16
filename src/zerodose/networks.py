"""Networks for ZeroDose."""
import torch
import torch.nn as nn


# Create a function to iterate and print 100 numbers with full docstring


class ConvBlock(nn.Module):
    """Convolutional block."""

    def __init__(self, ch_i, ch_o, dropout=0):
        """Initialize the block."""
        self.sequence = []
        super().__init__()

        for _i in range(2):
            self.sequence += [nn.Conv3d(ch_i, ch_o, padding=1, kernel_size=3)]
            self.sequence += [nn.BatchNorm3d(ch_o)]
            self.sequence += [nn.ReLU(True)]
            if dropout:
                self.sequence += [nn.Dropout(dropout)]

            ch_i = ch_o
        self.model = nn.Sequential(*self.sequence)

    def forward(self, x):
        """Forward pass of the block."""
        return self.model(x)


class UpConv(nn.Module):
    """Upsampling convolutional block."""

    def __init__(self, ch_i, ch_o, output_pad=0):
        """Initialize the block."""
        self.sequence = []
        super().__init__()

        self.sequence += [
            nn.ConvTranspose3d(
                ch_i, ch_o, stride=2, padding=1, kernel_size=3, output_padding=1
            )
        ]
        self.sequence += [nn.BatchNorm3d(ch_o)]
        self.sequence += [nn.ReLU(True)]

        self.model = nn.Sequential(*self.sequence)

    def forward(self, x, skip):
        """Forward pass of the block."""
        a = self.model(x)

        return torch.cat((skip, a), axis=1)


class DownConv(nn.Module):
    """Downsampling convolutional block."""

    def __init__(self, ch_i, ch_o):
        """Initialize the block."""
        self.sequence = []
        super().__init__()
        self.sequence += [nn.Conv3d(ch_i, ch_o, stride=2, padding=1, kernel_size=3)]
        self.sequence += [nn.BatchNorm3d(ch_o)]
        self.sequence += [nn.LeakyReLU(0.2, True)]
        self.model = nn.Sequential(*self.sequence)

    def forward(self, x):
        """Forward pass of the block."""
        return self.model(x)


class UNet3D(nn.Module):
    """3D U-Net model for PET reconstruction."""

    def __init__(self, refiner=False, do=False):
        """Initialize the model."""
        super().__init__()

        # Going Down
        if refiner:
            self.conv_d1 = ConvBlock(2, 32, dropout=0.1 * do)  # 32, 196, 196
        else:
            self.conv_d1 = ConvBlock(1, 32, dropout=0.1 * do)  # 32, 196, 196

        self.down_1 = DownConv(32, 128)  # 16, 48, 48

        self.conv_d2 = ConvBlock(128, 64, dropout=0.1 * do)  # 16, 48, 48
        self.down_2 = DownConv(64, 256)  # 8, 24, 24

        self.conv_d3 = ConvBlock(256, 128, dropout=0.2 * do)  # 8, 24, 24
        self.down_3 = DownConv(128, 512)  # 4, 12, 12

        self.conv_d4 = ConvBlock(512, 256, dropout=0.2 * do)  # 4, 12, 12
        self.down_4 = DownConv(256, 512)  # 2, 6, 6

        # Bottom
        self.conv_b = ConvBlock(512, 512, dropout=0.3 * do)  # 2, 6, 6

        # Going up
        self.up_1 = UpConv(512, 256)  # 4, 12, 12
        self.conv_u1 = ConvBlock(512, 256, dropout=0.3 * do)  # 4, 12, 12

        self.up_2 = UpConv(256, 128)  # 8, 24, 24
        self.conv_u2 = ConvBlock(256, 128, dropout=0.2 * do)  # 8, 24, 24

        self.up_3 = UpConv(128, 64)  # 16, 48, 48
        self.conv_u3 = ConvBlock(128, 64, dropout=0.2 * do)  # 16, 48, 48

        self.up_4 = UpConv(64, 32)  # 32, 196, 196
        self.conv_u4 = ConvBlock(64, 32, dropout=0.1 * do)  # 32, 196, 196

        self.last = nn.Conv3d(32, 1, padding=1, kernel_size=3)  # 32, 196, 196
        self.last_sigmoid = nn.Sigmoid()  # 32, 196, 196

    def forward(self, x):
        """Forward pass of the model."""
        # Going down
        c1 = self.conv_d1(x)
        x = self.down_1(c1)

        c2 = self.conv_d2(x)
        x = self.down_2(c2)

        c3 = self.conv_d3(x)
        x = self.down_3(c3)

        c4 = self.conv_d4(x)
        x = self.down_4(c4)

        # Bottom
        x = self.conv_b(x)

        # Going up
        x = self.conv_u1(self.up_1(x, c4))
        x = self.conv_u2(self.up_2(x, c3))
        x = self.conv_u3(self.up_3(x, c2))
        x = self.conv_u4(self.up_4(x, c1))

        x = self.last(x)
        return x


class DummyGenerator(nn.Module):
    """Dummy generator for testing purposes."""

    def __init__(self, *args, **kwargs):
        """Initialize the generator."""
        super().__init__()
        self.w = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        """Forward pass of the generator."""
        return self.w * x