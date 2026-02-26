"""Tests for epoch_engine.models module."""

import torch
import torch.nn as nn

from epoch_engine.models import BasicBlock, EDNet, ResNet
from epoch_engine.models.architectures import Decoder, Encoder

# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------


class TestEncoder:
    def test_single_block_output_shape(self):
        """One Conv block reduces spatial dims via MaxPool(3): 27 → 9."""
        # (1, 16) -> one Conv+ReLU+MaxPool(3) block
        # Input: (B, 1, 27, 27) -> MaxPool(3) -> (B, 16, 9, 9)
        encoder = Encoder(encoder_channels=(1, 16))
        x = torch.randn(2, 1, 27, 27)
        out = encoder(x)
        assert out.shape == (2, 16, 9, 9)

    def test_two_blocks_output_shape(self):
        """Two Conv blocks apply MaxPool twice: 27 → 9 → 3."""
        # (1, 16, 32) -> two blocks, 27 -> 9 -> 3
        encoder = Encoder(encoder_channels=(1, 16, 32))
        x = torch.randn(2, 1, 27, 27)
        out = encoder(x)
        assert out.shape == (2, 32, 3, 3)

    def test_output_is_tensor(self):
        """Output is a torch.Tensor."""
        encoder = Encoder(encoder_channels=(3, 8))
        x = torch.randn(1, 3, 27, 27)
        out = encoder(x)
        assert isinstance(out, torch.Tensor)

    def test_no_grad_inference(self):
        """Encoder runs without errors under torch.no_grad()."""
        encoder = Encoder(encoder_channels=(1, 16))
        x = torch.randn(1, 1, 27, 27)
        with torch.no_grad():
            out = encoder(x)
        assert out.shape[1] == 16


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------


class TestDecoder:
    def test_output_shape(self):
        """Decoder projects to the correct number of classes."""
        # decoder_features=(128, 64), num_labels=10
        # Input: (B, 128) -> (B, 10)
        decoder = Decoder(decoder_features=(128, 64), num_labels=10)
        x = torch.randn(4, 128)
        out = decoder(x)
        assert out.shape == (4, 10)

    def test_single_hidden_layer(self):
        """Decoder works with a single hidden-layer config."""
        decoder = Decoder(decoder_features=(64,), num_labels=5)
        x = torch.randn(3, 64)
        out = decoder(x)
        assert out.shape == (3, 5)

    def test_last_layer_is_linear(self):
        """The final layer is a Linear module with correct out_features."""
        decoder = Decoder(decoder_features=(32, 16), num_labels=4)
        assert isinstance(decoder.last, nn.Linear)
        assert decoder.last.out_features == 4

    def test_output_is_tensor(self):
        """Output is a torch.Tensor."""
        decoder = Decoder(decoder_features=(32,), num_labels=2)
        x = torch.randn(1, 32)
        out = decoder(x)
        assert isinstance(out, torch.Tensor)


# ---------------------------------------------------------------------------
# EDNet
# ---------------------------------------------------------------------------


class TestEDNet:
    def _make_model(self):
        # Input: (B, 1, 27, 27)
        # Encoder: [1, 16] -> output (B, 16, 9, 9) -> flatten -> 1296
        # Decoder: (1296, 128) -> output (B, 10)
        return EDNet(
            in_channels=1,
            encoder_channels=(16,),
            decoder_features=(1296, 128),
            num_labels=10,
        )

    def test_output_shape(self):
        """Forward pass produces logits of shape (B, num_labels)."""
        model = self._make_model()
        x = torch.randn(4, 1, 27, 27)
        out = model(x)
        assert out.shape == (4, 10)

    def test_output_is_tensor(self):
        """Output is a torch.Tensor."""
        model = self._make_model()
        x = torch.randn(2, 1, 27, 27)
        out = model(x)
        assert isinstance(out, torch.Tensor)

    def test_encoder_channels_stored(self):
        """in_channels is prepended to encoder_channels."""
        model = self._make_model()
        # in_channels prepended to encoder_channels
        assert model.encoder_channels == [1, 16]

    def test_has_flatten_layer(self):
        """Model contains a Flatten layer."""
        model = self._make_model()
        assert isinstance(model.flatten, nn.Flatten)

    def test_batch_size_one(self):
        """Works correctly with a single-sample batch."""
        model = self._make_model()
        x = torch.randn(1, 1, 27, 27)
        out = model(x)
        assert out.shape == (1, 10)


# ---------------------------------------------------------------------------
# BasicBlock
# ---------------------------------------------------------------------------


class TestBasicBlock:
    def test_output_shape_same_channels(self):
        """Spatial dims unchanged with same in/out channels and stride=1."""
        block = BasicBlock(in_channels=16, out_channels=16, stride=1)
        x = torch.randn(2, 16, 8, 8)
        out = block(x)
        assert out.shape == (2, 16, 8, 8)

    def test_output_shape_different_channels(self):
        """Spatial dims halved and channels doubled with stride=2."""
        block = BasicBlock(in_channels=16, out_channels=32, stride=2)
        x = torch.randn(2, 16, 8, 8)
        out = block(x)
        assert out.shape == (2, 32, 4, 4)

    def test_identity_shortcut_when_same_channels_same_stride(self):
        """Identity shortcut (empty Sequential) used when no downsampling needed."""
        block = BasicBlock(in_channels=16, out_channels=16, stride=1)
        # shortcut should be empty Sequential (identity)
        assert len(list(block.shortcut.children())) == 0

    def test_projection_shortcut_when_channels_differ(self):
        """Projection shortcut added when channel count changes."""
        block = BasicBlock(in_channels=16, out_channels=32, stride=1)
        # shortcut must have layers (Conv2d + BN)
        assert len(list(block.shortcut.children())) > 0

    def test_projection_shortcut_when_stride_differs(self):
        """Projection shortcut added when stride > 1."""
        block = BasicBlock(in_channels=16, out_channels=16, stride=2)
        assert len(list(block.shortcut.children())) > 0

    def test_output_is_tensor(self):
        """Output is a torch.Tensor."""
        block = BasicBlock(in_channels=8, out_channels=8)
        x = torch.randn(1, 8, 4, 4)
        out = block(x)
        assert isinstance(out, torch.Tensor)


# ---------------------------------------------------------------------------
# ResNet
# ---------------------------------------------------------------------------


class TestResNet:
    def _make_model(self, in_channels=3, num_classes=10):
        return ResNet(
            in_channels=in_channels,
            num_blocks=[2, 2, 2],
            num_classes=num_classes,
        )

    def test_output_shape(self):
        """Forward pass produces logits of shape (B, num_classes)."""
        model = self._make_model()
        x = torch.randn(4, 3, 32, 32)
        out = model(x)
        assert out.shape == (4, 10)

    def test_custom_num_classes(self):
        """Output size matches a custom num_classes."""
        model = self._make_model(num_classes=5)
        x = torch.randn(2, 3, 32, 32)
        out = model(x)
        assert out.shape == (2, 5)

    def test_single_channel_input(self):
        """Works with single-channel (grayscale) input."""
        model = self._make_model(in_channels=1)
        x = torch.randn(2, 1, 32, 32)
        out = model(x)
        assert out.shape == (2, 10)

    def test_has_three_resnet_layers(self):
        """Model exposes layer1, layer2, and layer3 attributes."""
        model = self._make_model()
        assert hasattr(model, "layer1")
        assert hasattr(model, "layer2")
        assert hasattr(model, "layer3")

    def test_fc_layer_output_features(self):
        """Fully-connected layer outputs the correct number of features."""
        model = self._make_model(num_classes=7)
        assert model.fc.out_features == 7

    def test_output_is_tensor(self):
        """Output is a torch.Tensor."""
        model = self._make_model()
        x = torch.randn(1, 3, 32, 32)
        out = model(x)
        assert isinstance(out, torch.Tensor)

    def test_batch_size_one(self):
        """Works correctly with a single-sample batch."""
        model = self._make_model()
        x = torch.randn(1, 3, 32, 32)
        out = model(x)
        assert out.shape == (1, 10)
