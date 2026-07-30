import torch
from torch.autograd import Function

"""Autograd helpers used by the quantized SNN layers."""


class FakeQuantize(Function):
    """Apply fake quantization with a straight-through gradient."""

    @staticmethod
    def forward(ctx, x, qparam):
        x = qparam.quantize_tensor(x)
        x = qparam.dequantize_tensor(x)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        # Use a straight-through estimator for the non-differentiable quantizer.
        return grad_output, None


def interp(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    """Linearly interpolate each row of ``x`` over shared sample points."""
    x_ = x.reshape(x.size(0), -1)
    xp = xp.unsqueeze(0)
    fp = fp.unsqueeze(0)

    m = (fp[:, 1:] - fp[:, :-1]) / (xp[:, 1:] - xp[:, :-1])
    b = fp[:, :-1] - (m.mul(xp[:, :-1]))

    # Select the line segment immediately to the left of each query value.
    indices = torch.sum(torch.ge(x_[:, :, None], xp[:, None, :]), -1) - 1
    indices = torch.clamp(indices, 0, m.shape[-1] - 1)

    line_idx = torch.linspace(0, indices.shape[0], 1, device=indices.device).to(
        torch.long
    )
    line_idx = line_idx.expand(indices.shape)
    out = m[line_idx, indices].mul(x_) + b[line_idx, indices]
    out = out.reshape(x.shape)
    return out
