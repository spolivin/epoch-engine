"""Module containing building blocks of models and functions to build neural networks."""

# Author: Sergey Polivin <s.polivin@gmail.com>
# License: MIT License

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """Builds a series of convolutional blocks."""

    def __init__(self, encoder_channels: tuple[int]) -> None:
        """Initializes a class instance.

        Args:
            encoder_channels (tuple[int]): Tuple of convolution channels.
        """
        super().__init__()
        # Creating a series of encoder blocks in accordance with channels in `encoder_channels`
        encoder_blocks = [
            self._make_encoder_block(in_channel, out_channel)
            for in_channel, out_channel in zip(encoder_channels, encoder_channels[1:])
        ]
        # Sequentially connecting the generated convolution blocks
        self.encoder_blocks = nn.Sequential(*encoder_blocks)

    def _make_encoder_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        """Creates a convolution block with ReLU activation and MaxPooling.

        Args:
            in_channels (int): Number of input channels for convolution.
            out_channels (int): Number of output channels for convolution.

        Returns:
            nn.Sequential: Convolution block of sequentially connected layers.
        """
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Makes a forward pass.

        Args:
            x (torch.Tensor): Input tensor for Encoder.

        Returns:
            torch.Tensor: Output tensor of Encoder.
        """
        return self.encoder_blocks(x)


class Decoder(nn.Module):
    """Builds a series of feedforward blocks."""

    def __init__(self, decoder_features: tuple[int], num_labels: int) -> None:
        """Initializes a class instance.

        Args:
            decoder_features (tuple[int]): Tuple of features in the Decoder.
            num_labels (int): Number of output labels in the last layer.
        """
        super().__init__()
        # Creating a series of decoder blocks in accordance with features in `decoder_features`
        decoder_blocks = [
            self._make_decoder_block(in_feature, out_feature)
            for in_feature, out_feature in zip(decoder_features, decoder_features[1:])
        ]
        # Sequentially connecting the generated feedforward blocks
        self.decoder_blocks = nn.Sequential(*decoder_blocks)
        # Creating the last layer of the Decoder
        self.last = nn.Linear(decoder_features[-1], num_labels)

    def _make_decoder_block(self, in_features: int, out_features: int) -> nn.Sequential:
        """Creates a fully connected linear layer with Sigmoid activation.

        Args:
            in_features (int): Number of input features of the Decoder.
            out_features (int): Number of output features of the Decoder.

        Returns:
            nn.Sequential: Decoder block of sequentially connected layers.
        """
        return nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Makes a forward pass.

        Args:
            x (torch.Tensor): Input tensor for Decoder.

        Returns:
            torch.Tensor: Output tensor of Decoder.
        """
        x = self.decoder_blocks(x)
        x = self.last(x)

        return x


class EDNet(nn.Module):
    """Joins Encoder with Decoder to form a CNN."""

    def __init__(
        self,
        in_channels: int,
        encoder_channels: tuple[int],
        decoder_features: tuple[int],
        num_labels: int,
    ) -> None:
        """Initializes a class instance.

        Args:
            in_channels (int): Number of input channels for the image.
            encoder_channels (tuple[int]): Tuple of channels for the Encoder.
            decoder_features (tuple[int]): Tuple of channels for the Decoder.
            num_labels (int): Number of ouput labels.
        """
        super().__init__()
        # Setting up Encoder block
        self.encoder_channels = [in_channels, *encoder_channels]
        self.encoder = Encoder(self.encoder_channels)
        # Setting up Decoder block
        self.decoder_features = decoder_features
        self.decoder = Decoder(self.decoder_features, num_labels)
        # Flatten layer
        self.flatten = nn.Flatten()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Makes a forward pass.

        Args:
            x (torch.Tensor): Input tensor for a CNN.

        Returns:
            torch.Tensor: Output tensor of a CNN.
        """
        x = self.encoder(x)
        x = self.flatten(x)
        x = self.decoder(x)

        return x


class BasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        """Builds a block of ResNet layer with 2 inner blocks (CONV + BN + SKIP CONNECTION).

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            stride (int, optional): Value of stride. Defaults to 1.
        """
        super().__init__()
        # BLOCK 1 (CONV + BN) #################################################
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        # BLOCK 2 (CONV + BN) #################################################
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # SKIP CONNECTION #################################################
        self.shortcut = nn.Sequential()
        # Specifying condition for applying residual connection
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Makes a forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        # Output for BLOCK 1
        out = F.relu(self.bn1(self.conv1(x)))
        # Output for BLOCK 2
        out = self.bn2(self.conv2(out))
        # Adding up SKIP CONNECTION if condition met
        out += self.shortcut(x)
        # Output of RESNET BLOCK
        return F.relu(out)


class ResNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        block: BasicBlock,
        num_blocks: list[int],
        num_classes: int = 10,
    ) -> None:
        """Builds a ResNet model with 3 layers.

        Args:
            in_channels (int): Number of input channels.
            block (BasicBlock): Block of ResNet layer.
            num_blocks (list[int]): Number of blocks inside Block of ResNet layer.
            num_classes (int, optional): Number of output classes. Defaults to 10.
        """
        super().__init__()
        # Specifying a number of initial input channels for ResNet block
        self.in_block_channels = 16

        # LAYER 0 (CONV + BN) #################################################################################
        self.conv1 = nn.Conv2d(
            in_channels,
            self.in_block_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(self.in_block_channels)

        # LAYERS 1, 2, 3 (`num_blocks` blocks with 2 CONV/BN BLOCKS for 1 layer) ##############################
        self.layer1 = self._make_resnet_layer(
            block=block, out_channels=16, num_blocks=num_blocks[0], stride=1
        )
        self.layer2 = self._make_resnet_layer(
            block=block, out_channels=32, num_blocks=num_blocks[1], stride=2
        )
        self.layer3 = self._make_resnet_layer(
            block=block, out_channels=64, num_blocks=num_blocks[2], stride=2
        )

        # GLOBAL POOLING LAYER ################################################################################
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # FC LAYER ############################################################################################
        self.fc = nn.Linear(64, num_classes)

    def _make_resnet_layer(
        self,
        block: BasicBlock,
        out_channels: int,
        num_blocks: int,
        stride: int,
    ) -> nn.Sequential:
        """Creates a ResNet layer by stacking up BasicBlock-s.

        Args:
            block (BasicBlock): Block inside ResNet layer.
            out_channels (int): Number of output channels.
            num_blocks (int): Number of blocks inside ResNet layer.
            stride (int): Value of stride.

        Returns:
            nn.Sequential: Layer of ResNet with blocks added inside.
        """
        # Specifying strides to be used for each BasicBlock
        strides = [stride] + [1] * (num_blocks - 1)
        # Stacking up ResNet blocks
        layers = []
        for stride in strides:
            layers.append(block(self.in_block_channels, out_channels, stride))
            self.in_block_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Makes a forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = torch.flatten(out, 1)
        return self.fc(out)
