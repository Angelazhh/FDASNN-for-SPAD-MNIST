"""Spiking neural network architectures used by the FDASNN experiments."""

import torch
import torch.nn as nn
from spikingjelly.activation_based import functional, layer, neuron, surrogate


class FDASNN(nn.Module):
    """Feature-decoupled SNN for multi-step 240 x 240 spike inputs."""

    def __init__(self, T: int, channels: int):
        super().__init__()
        self.T = T
        self.vth = 1.0

        # Stem: reduce the 240 x 240 input to 120 x 120 spike features.
        self.conv0 = layer.Conv2d(1, channels, kernel_size=5, stride=2, padding=2)
        self.IF0 = neuron.IFNode(
            surrogate_function=surrogate.ATan(), v_threshold=self.vth, store_v_seq=False
        )

        # First depthwise-pointwise block includes a pooled residual branch.
        self.conv11 = layer.Conv2d(
            channels, channels, kernel_size=3, stride=1, padding=1, groups=channels
        )
        self.IF11 = neuron.IFNode(
            surrogate_function=surrogate.ATan(), v_threshold=self.vth, store_v_seq=False
        )

        self.conv12 = layer.Conv2d(channels, channels // 2, kernel_size=1)
        self.IF12 = neuron.IFNode(
            surrogate_function=surrogate.ATan(), v_threshold=self.vth, store_v_seq=False
        )

        self.maxpool1 = layer.MaxPool2d(3, 3)
        self.conv13 = layer.Conv2d(
            channels // 2,
            channels // 2,
            kernel_size=5,
            stride=3,
            padding=2,
            groups=channels // 2 // 2,
        )
        self.IF13 = neuron.IFNode(
            surrogate_function=surrogate.ATan(), v_threshold=self.vth, store_v_seq=False
        )

        self.conv14 = layer.Conv2d(
            channels // 2,
            channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=channels // 2 // 2,
        )
        self.IF14 = neuron.IFNode(
            surrogate_function=surrogate.ATan(), v_threshold=self.vth, store_v_seq=False
        )

        self.conv15 = layer.Conv2d(
            channels, channels, kernel_size=3, padding=1, groups=channels
        )
        self.IF15 = neuron.IFNode(
            surrogate_function=surrogate.ATan(), v_threshold=self.vth, store_v_seq=False
        )

        self.conv16 = layer.Conv2d(channels, channels, kernel_size=1)
        self.IF16 = neuron.IFNode(
            surrogate_function=surrogate.ATan(), v_threshold=self.vth, store_v_seq=False
        )

        # The second stage doubles channels while reducing spatial resolution.
        self.conv2 = layer.Conv2d(
            channels, channels * 2, kernel_size=3, stride=2, padding=1
        )
        self.IF2 = neuron.IFNode(
            surrogate_function=surrogate.ATan(), v_threshold=self.vth, store_v_seq=False
        )

        # This block mirrors the first residual structure at lower resolution.
        self.conv31 = layer.Conv2d(
            channels * 2,
            channels * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=channels * 2,
        )
        self.IF31 = neuron.IFNode(
            surrogate_function=surrogate.ATan(), v_threshold=self.vth, store_v_seq=False
        )

        self.conv32 = layer.Conv2d(channels * 2, channels, kernel_size=1)
        self.IF32 = neuron.IFNode(
            surrogate_function=surrogate.ATan(), v_threshold=self.vth, store_v_seq=False
        )

        self.maxpool3 = layer.MaxPool2d(3, 3)
        self.conv33 = layer.Conv2d(
            channels, channels, kernel_size=5, stride=3, padding=1, groups=channels // 2
        )
        self.IF33 = neuron.IFNode(
            surrogate_function=surrogate.ATan(), v_threshold=self.vth, store_v_seq=False
        )

        self.conv34 = layer.Conv2d(
            channels,
            channels * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=channels // 2,
        )
        self.IF34 = neuron.IFNode(
            surrogate_function=surrogate.ATan(), v_threshold=self.vth, store_v_seq=False
        )

        self.conv35 = layer.Conv2d(
            channels * 2, channels * 2, kernel_size=3, padding=1, groups=channels * 2
        )
        self.IF35 = neuron.IFNode(
            surrogate_function=surrogate.ATan(), v_threshold=self.vth, store_v_seq=False
        )

        self.conv36 = layer.Conv2d(channels * 2, channels * 2, kernel_size=1)
        self.IF36 = neuron.IFNode(
            surrogate_function=surrogate.ATan(), v_threshold=self.vth, store_v_seq=False
        )

        # The head produces one class-logit vector per time step.
        self.conv4 = layer.Conv2d(channels * 2, channels // 2, kernel_size=3, padding=1)
        self.IF4 = neuron.IFNode(
            surrogate_function=surrogate.ATan(), v_threshold=self.vth, store_v_seq=False
        )

        self.Fla = layer.Flatten()

        self.fc5 = layer.Linear(channels // 2 * 6 * 6, 10)

        functional.set_step_mode(self, step_mode="m")

    def forward(self, x: torch.Tensor):
        """Return per-time-step logits for input shaped ``[T, N, C, H, W]``."""

        x = self.conv0(x)
        x = self.IF0(x)

        # Preserve a pooled pre-spike branch for the first residual addition.
        x = self.conv11(x)
        y0 = x
        y0 = self.maxpool1(y0)
        x = self.IF11(x)

        x = self.conv12(x)
        x = self.IF12(x)

        x = self.conv13(x)
        x = self.IF13(x)

        x = self.conv14(x)
        x = self.IF14(x)

        x = self.conv15(x)
        x = self.IF15(x + y0)

        x = self.conv16(x)
        x = self.IF16(x)

        x = self.conv2(x)
        x = self.IF2(x)

        # Repeat residual fusion at the lower spatial resolution.
        x = self.conv31(x)
        y = x
        y = self.maxpool3(y)
        x = self.IF31(x)

        x = self.conv32(x)
        x = self.IF32(x)

        x = self.conv33(x)
        x = self.IF33(x)

        x = self.conv34(x)
        x = self.IF34(x)

        x = self.conv35(x)
        x = self.IF35(x + y)

        x = self.conv36(x)
        x = self.IF36(x)

        x = self.conv4(x)

        x = self.IF4(x)

        x = self.Fla(x)

        x = self.fc5(x)

        return x


class TCJA(nn.Module):
    """Joint temporal-channel attention for spike feature maps."""

    def __init__(
        self,
        kernel_size_t: int = 2,
        kernel_size_c: int = 1,
        T: int = 8,
        channel: int = 128,
    ):
        super().__init__()
        self.channel = channel
        self.kernel_size_t = kernel_size_t
        self.kernel_size_c = kernel_size_c
        self.conv_t = nn.Conv1d(
            T, T, kernel_size=kernel_size_t, padding=kernel_size_t // 2, bias=False
        )
        self.conv_c = nn.Conv1d(
            channel,
            channel,
            kernel_size=kernel_size_c,
            padding=kernel_size_c // 2,
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    @staticmethod
    def _crop_to_length(x: torch.Tensor, length: int):
        if x.shape[-1] == length:
            return x
        return x[..., :length]

    def forward(self, x: torch.Tensor):

        # Global spatial statistics drive separate temporal and channel gates.
        stat = torch.mean(x.permute(1, 0, 2, 3, 4), dim=[3, 4])
        stat_c = stat.permute(0, 2, 1)
        conv_t = self._crop_to_length(self.conv_t(stat), stat.shape[-1]).permute(
            1, 0, 2
        )
        conv_c = self._crop_to_length(self.conv_c(stat_c), stat_c.shape[-1]).permute(
            2, 0, 1
        )
        attn = self.sigmoid(conv_t * conv_c)
        return x * attn[:, :, :, None, None]


class SNN_TCJA_CONV3(nn.Module):
    """Compact SNN baseline with TCJA blocks."""

    def __init__(self, T: int, channels: int):
        super().__init__()
        self.T = T
        self.vth = 1.0

        self.conv0 = layer.Conv2d(1, channels, kernel_size=5, stride=2, padding=2)
        self.IF0 = neuron.IFNode(
            surrogate_function=surrogate.ATan(), v_threshold=self.vth, store_v_seq=False
        )

        self.conv1 = layer.Conv2d(
            channels, channels, kernel_size=3, stride=1, padding=1
        )
        self.IF1 = neuron.IFNode(
            surrogate_function=surrogate.ATan(), v_threshold=self.vth, store_v_seq=False
        )

        self.tcja1 = TCJA(kernel_size_t=3, kernel_size_c=3, T=T, channel=channels)
        self.MaxPool1 = layer.MaxPool2d(3, 3)

        self.conv2 = layer.Conv2d(
            channels, channels * 2, kernel_size=3, stride=2, padding=1
        )
        self.IF2 = neuron.IFNode(
            surrogate_function=surrogate.ATan(), v_threshold=self.vth, store_v_seq=False
        )

        self.conv3 = layer.Conv2d(
            channels * 2, channels * 2, kernel_size=3, stride=1, padding=1
        )
        self.IF3 = neuron.IFNode(
            surrogate_function=surrogate.ATan(), v_threshold=self.vth, store_v_seq=False
        )

        self.tcja2 = TCJA(kernel_size_t=2, kernel_size_c=1, T=T, channel=channels * 2)
        self.MaxPool2 = layer.MaxPool2d(3, 3)

        self.conv4 = layer.Conv2d(channels * 2, channels // 2, kernel_size=3, padding=1)
        self.IF4 = neuron.IFNode(
            surrogate_function=surrogate.ATan(), v_threshold=self.vth, store_v_seq=False
        )

        self.Fla = layer.Flatten()

        self.fc5 = layer.Linear(channels // 2 * 6 * 6, 10)

        functional.set_step_mode(self, step_mode="m")

    def forward(self, x: torch.Tensor):

        x = self.conv0(x)
        x = self.IF0(x)

        x = self.conv1(x)
        x = self.IF1(x)

        x = self.tcja1(x)
        x = self.MaxPool1(x)

        x = self.conv2(x)
        x = self.IF2(x)

        x = self.conv3(x)
        x = self.IF3(x)

        x = self.tcja2(x)
        x = self.MaxPool2(x)

        x = self.conv4(x)
        x = self.IF4(x)

        x = self.Fla(x)
        x = self.fc5(x)
        return x


class SCTFA(nn.Module):
    """Spatial-channel attention for one time step."""

    def __init__(
        self,
        T: int,
        C: int,
        reduction_t: int = 4,
        reduction_c: int = 4,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.T = T
        self.C = C
        self.kernel_size = kernel_size

        rc = max(1, min(C, reduction_c))

        self.spatial_excitation = nn.Conv2d(C, 1, kernel_size=1, bias=True)
        self.channel_mlp = nn.Sequential(
            nn.Linear(C, max(1, C // rc), bias=False),
            nn.ReLU(),
            nn.Linear(max(1, C // rc), C, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor):

        assert x.dim() == 4, f"expected [N, C, H, W], got {x.shape}"
        spatial_attn = self.sigmoid(self.spatial_excitation(x))
        channel_stat = torch.mean(x, dim=[2, 3])
        channel_attn = self.sigmoid(self.channel_mlp(channel_stat))[:, :, None, None]
        return spatial_attn * channel_attn


class SCTFAIFNode(nn.Module):
    """Integrate-and-fire node with spike-triggered attention."""

    def __init__(
        self,
        T: int,
        C: int,
        v_threshold: float = 1.0,
        decay: int = 1,
        reduction_c: int = 4,
        surrogate_function=None,
    ):
        super().__init__()
        self.T = T
        self.C = C
        self.v_threshold = v_threshold
        self.decay = decay
        self.attention = SCTFA(T=T, C=C, reduction_c=reduction_c)
        self.surrogate_function = surrogate_function or surrogate.ATan()

    def forward(self, x: torch.Tensor):

        assert x.dim() == 5, f"expected [T, N, C, H, W], got {x.shape}"
        v = torch.zeros_like(x[0])
        prev_spike = torch.zeros_like(x[0])
        prev_attn = torch.ones_like(x[0])
        spikes = []

        # Previous spikes and attention modulate the next membrane update.
        for t in range(x.shape[0]):

            v = v * prev_attn * (1 - prev_spike) + x[t]
            spike = self.surrogate_function(v - self.v_threshold)
            spikes.append(spike)
            prev_attn = self.attention(spike)
            prev_spike = spike

        return torch.stack(spikes, dim=0)


class SNN_SCTFA_CONV3(nn.Module):
    """Compact SNN baseline with SCTFA neuron blocks."""

    def __init__(self, T: int, channels: int):
        super().__init__()
        self.T = T
        self.vth = 1.0

        self.conv0 = layer.Conv2d(1, channels, kernel_size=5, stride=2, padding=2)
        self.IF0 = neuron.IFNode(
            surrogate_function=surrogate.ATan(), v_threshold=self.vth, store_v_seq=False
        )

        self.conv1 = layer.Conv2d(
            channels, channels, kernel_size=3, stride=1, padding=1
        )
        self.SCTFAIF1 = SCTFAIFNode(
            T=T,
            C=channels,
            v_threshold=self.vth,
            reduction_c=4,
            surrogate_function=surrogate.ATan(),
        )
        self.MaxPool1 = layer.MaxPool2d(3, 3)

        self.conv2 = layer.Conv2d(
            channels, channels * 2, kernel_size=3, stride=2, padding=1
        )
        self.IF2 = neuron.IFNode(
            surrogate_function=surrogate.ATan(), v_threshold=self.vth, store_v_seq=False
        )

        self.conv3 = layer.Conv2d(
            channels * 2, channels * 2, kernel_size=3, stride=1, padding=1
        )
        self.SCTFAIF3 = SCTFAIFNode(
            T=T,
            C=channels * 2,
            v_threshold=self.vth,
            reduction_c=4,
            surrogate_function=surrogate.ATan(),
        )
        self.MaxPool3 = layer.MaxPool2d(3, 3)

        self.conv4 = layer.Conv2d(channels * 2, channels // 2, kernel_size=3, padding=1)
        self.IF4 = neuron.IFNode(
            surrogate_function=surrogate.ATan(), v_threshold=self.vth, store_v_seq=False
        )

        self.Fla = layer.Flatten()

        self.fc5 = layer.Linear(channels // 2 * 6 * 6, 10)

        functional.set_step_mode(self, step_mode="m")

    def forward(self, x: torch.Tensor):

        x = self.conv0(x)
        x = self.IF0(x)

        x = self.conv1(x)
        x = self.SCTFAIF1(x)
        x = self.MaxPool1(x)

        x = self.conv2(x)
        x = self.IF2(x)

        x = self.conv3(x)
        x = self.SCTFAIF3(x)
        x = self.MaxPool3(x)

        x = self.conv4(x)
        x = self.IF4(x)

        x = self.Fla(x)
        x = self.fc5(x)
        return x
