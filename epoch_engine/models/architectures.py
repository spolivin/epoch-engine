"""CNN architectures for image classification: EDNet (encoder-decoder) and ResNet."""

# Author: Sergey Polivin <s.polivin@gmail.com>
# License: MIT License

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """Sequential convolutional feature extractor.

    Stacks Conv2d -> ReLU -> MaxPool2d blocks derived from consecutive pairs in
    ``encoder_channels``, progressively downsampling the spatial dimensions.
    """

    def __init__(self, encoder_channels: tuple[int]) -> None:
        """
        Args:
            encoder_channels (tuple[int]): Channel sizes for each stage,
                e.g. ``(3, 32, 64)`` creates two conv blocks: 3 -> 32 and 32 -> 64.
        """
        super().__init__()
        # Creating a series of encoder blocks in accordance with channels in `encoder_channels`
        encoder_blocks = [
            self._make_encoder_block(in_channel, out_channel)
            for in_channel, out_channel in zip(
                encoder_channels, encoder_channels[1:]
            )
        ]
        # Sequentially connecting the generated convolution blocks
        self.encoder_blocks = nn.Sequential(*encoder_blocks)

    def _make_encoder_block(
        self, in_channels: int, out_channels: int
    ) -> nn.Sequential:
        """Builds a single Conv2d (3X3, pad=1) -> ReLU -> MaxPool2d (3X3) block.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.

        Returns:
            nn.Sequential: The assembled conv block.
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
        """Passes ``x`` through all encoder blocks in sequence.

        Args:
            x (torch.Tensor): Input feature map ``(B, C, H, W)``.

        Returns:
            torch.Tensor: Downsampled feature map.
        """
        return self.encoder_blocks(x)


class Decoder(nn.Module):
    """Fully-connected classification head.

    Stacks Linear -> Sigmoid blocks from consecutive pairs in
    ``decoder_features``, followed by a final linear layer that maps to
    ``num_labels`` outputs (logits).
    """

    def __init__(self, decoder_features: tuple[int], num_labels: int) -> None:
        """
        Args:
            decoder_features (tuple[int]): Feature sizes for each hidden
                stage, e.g. ``(512, 256)`` creates one Linear→Sigmoid block.
            num_labels (int): Number of output classes/logits.
        """
        super().__init__()
        # Creating a series of decoder blocks in accordance with features in `decoder_features`
        decoder_blocks = [
            self._make_decoder_block(in_feature, out_feature)
            for in_feature, out_feature in zip(
                decoder_features, decoder_features[1:]
            )
        ]
        # Sequentially connecting the generated feedforward blocks
        self.decoder_blocks = nn.Sequential(*decoder_blocks)
        # Creating the last layer of the Decoder
        self.last = nn.Linear(decoder_features[-1], num_labels)

    def _make_decoder_block(
        self, in_features: int, out_features: int
    ) -> nn.Sequential:
        """Builds a single Linear -> Sigmoid hidden block.

        Args:
            in_features (int): Input dimensionality.
            out_features (int): Output dimensionality.

        Returns:
            nn.Sequential: The assembled linear block.
        """
        return nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Passes ``x`` through hidden blocks then the final linear layer.

        Args:
            x (torch.Tensor): Flattened feature vector ``(B, in_features)``.

        Returns:
            torch.Tensor: Class logits ``(B, num_labels)``.
        """
        x = self.decoder_blocks(x)
        x = self.last(x)

        return x


class EDNet(nn.Module):
    """Encoder-decoder CNN for image classification.

    Combines an :class:`Encoder` (convolutional feature extraction with
    spatial downsampling) with a :class:`Decoder` (fully-connected
    classification head) via a flatten operation.
    """

    def __init__(
        self,
        in_channels: int,
        encoder_channels: tuple[int],
        decoder_features: tuple[int],
        num_labels: int,
    ) -> None:
        """
        Args:
            in_channels (int): Number of channels in the input image
                (e.g. 1 for grayscale, 3 for RGB).
            encoder_channels (tuple[int]): Channel sizes for the encoder
                stages (excluding ``in_channels`` which is prepended).
            decoder_features (tuple[int]): Feature sizes for the decoder
                hidden stages. The first value must match the flattened
                encoder output size.
            num_labels (int): Number of output classes.
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
        """Encodes, flattens, then decodes the input.

        Args:
            x (torch.Tensor): Input image tensor ``(B, C, H, W)``.

        Returns:
            torch.Tensor: Class logits ``(B, num_labels)``.
        """
        x = self.encoder(x)
        x = self.flatten(x)
        x = self.decoder(x)

        return x


class BasicBlock(nn.Module):
    """Residual building block for ResNet.

    Contains two 3X3 Conv2d -> BN sub-layers with a skip connection. When the
    spatial size or channel count changes (``stride != 1`` or
    ``in_channels != out_channels``), the shortcut uses a 1X1 convolution
    with BN to match dimensions.
    """

    def __init__(
        self, in_channels: int, out_channels: int, stride: int = 1
    ) -> None:
        """
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            stride (int, optional): Stride for the first conv and the
                shortcut projection. Defaults to 1.
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
        """Applies two conv -> BN sub-layers with a residual shortcut and ReLU.

        Args:
            x (torch.Tensor): Input feature map ``(B, in_channels, H, W)``.

        Returns:
            torch.Tensor: Output feature map ``(B, out_channels, H', W')``.
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
    """Three-stage ResNet for image classification.

    Architecture: initial Conv2d -> BN -> ReLU stem, three residual stages with
    channel widths 16/32/64, adaptive average pooling, and a fully-connected
    output layer.
    """

    def __init__(
        self,
        in_channels: int,
        num_blocks: list[int],
        block: BasicBlock = BasicBlock,
        num_classes: int = 10,
    ) -> None:
        """
        Args:
            in_channels (int): Number of channels in the input image.
            num_blocks (list[int]): Number of :class:`BasicBlock` blocks per
                stage, e.g. ``[2, 2, 2]`` for a 6-layer residual body.
            block (type[BasicBlock], optional): Block class used to build each
                stage. Defaults to :class:`BasicBlock`.
            num_classes (int, optional): Number of output classes. Defaults
                to 10.
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
        """Stacks ``num_blocks`` BasicBlocks into a single residual stage.

        The first block uses ``stride`` for downsampling; subsequent blocks
        use stride 1. Updates ``self.in_block_channels`` in place.

        Args:
            block (type[BasicBlock]): Block class to instantiate.
            out_channels (int): Output channels for every block in the stage.
            num_blocks (int): Number of blocks in the stage.
            stride (int): Stride applied to the first block.

        Returns:
            nn.Sequential: The assembled residual stage.
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
        """Runs the full ResNet pipeline: stem -> stages -> pool -> classifier.

        Args:
            x (torch.Tensor): Input image tensor ``(B, in_channels, H, W)``.

        Returns:
            torch.Tensor: Class logits ``(B, num_classes)``.
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = torch.flatten(out, 1)
        return self.fc(out)
