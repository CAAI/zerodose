"""Networks for ZeroDose."""
from typing import List

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Convolutional block."""

    def __init__(self, ch_i: int, ch_o: int, dropout: float = 0) -> None:
        """Initialize the block."""
        self.sequence: List[nn.Module] = []
        super().__init__()

        for _i in range(2):
            self.sequence += [nn.Conv3d(ch_i, ch_o, padding=1, kernel_size=3)]
            self.sequence += [nn.BatchNorm3d(ch_o)]
            self.sequence += [nn.ReLU(True)]
            if dropout:
                self.sequence += [nn.Dropout(dropout)]

            ch_i = ch_o
        self.model = nn.Sequential(*self.sequence)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the block."""
        return self.model(x)


class UpConv(nn.Module):
    """Upsampling convolutional block."""

    def __init__(self, ch_i: int, ch_o: int) -> None:
        """Initialize the block."""
        self.sequence: List[nn.Module] = []
        super().__init__()

        self.sequence += [
            nn.ConvTranspose3d(
                ch_i, ch_o, stride=2, padding=1, kernel_size=3, output_padding=1
            )
        ]
        self.sequence += [nn.BatchNorm3d(ch_o)]
        self.sequence += [nn.ReLU(True)]

        self.model = nn.Sequential(*self.sequence)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Forward pass of the block."""
        a: torch.Tensor = self.model(x)

        return torch.cat((skip, a), dim=1)


class DownConv(nn.Module):
    """Downsampling convolutional block."""

    def __init__(self, ch_i: int, ch_o: int) -> None:
        """Initialize the block."""
        self.sequence: List[nn.Module] = []
        super().__init__()
        self.sequence += [nn.Conv3d(ch_i, ch_o, stride=2, padding=1, kernel_size=3)]
        self.sequence += [nn.BatchNorm3d(ch_o)]
        self.sequence += [nn.LeakyReLU(0.2, True)]
        self.model = nn.Sequential(*self.sequence)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the block."""
        return self.model(x)


class UNet3D(nn.Module):
    """3D U-Net model for PET reconstruction."""

    def __init__(self, use_dropout: bool = False) -> None:
        """Initialize the model."""
        super().__init__()

        self.conv_d1 = ConvBlock(1, 32, dropout=0.1 * use_dropout)  # 32, 196, 196

        self.down_1 = DownConv(32, 128)  # 16, 48, 48

        self.conv_d2 = ConvBlock(128, 64, dropout=0.1 * use_dropout)  # 16, 48, 48
        self.down_2 = DownConv(64, 256)  # 8, 24, 24

        self.conv_d3 = ConvBlock(256, 128, dropout=0.2 * use_dropout)  # 8, 24, 24
        self.down_3 = DownConv(128, 512)  # 4, 12, 12

        self.conv_d4 = ConvBlock(512, 256, dropout=0.2 * use_dropout)  # 4, 12, 12
        self.down_4 = DownConv(256, 512)  # 2, 6, 6

        # Bottom
        self.conv_b = ConvBlock(512, 512, dropout=0.3 * use_dropout)  # 2, 6, 6

        # Going up
        self.up_1 = UpConv(512, 256)  # 4, 12, 12
        self.conv_u1 = ConvBlock(512, 256, dropout=0.3 * use_dropout)  # 4, 12, 12

        self.up_2 = UpConv(256, 128)  # 8, 24, 24
        self.conv_u2 = ConvBlock(256, 128, dropout=0.2 * use_dropout)  # 8, 24, 24

        self.up_3 = UpConv(128, 64)  # 16, 48, 48
        self.conv_u3 = ConvBlock(128, 64, dropout=0.2 * use_dropout)  # 16, 48, 48

        self.up_4 = UpConv(64, 32)  # 32, 196, 196
        self.conv_u4 = ConvBlock(64, 32, dropout=0.1 * use_dropout)  # 32, 196, 196

        self.last = nn.Conv3d(32, 1, padding=1, kernel_size=3)  # 32, 196, 196
        self.last_sigmoid = nn.Sigmoid()  # 32, 196, 196

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the generator."""
        super().__init__()
        self.w = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the generator."""
        return self.w * x
