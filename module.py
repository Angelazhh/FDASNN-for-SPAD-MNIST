"""Attention blocks and post-training quantization modules for FDASNN."""

import math
import numpy as np
from spikingjelly.activation_based import neuron, surrogate, layer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from function import FakeQuantize, interp


def set_seed(seed):
    """Seed PyTorch and NumPy for reproducible experiments."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
 


def calcScaleZeroPoint(min_val, max_val, num_bits=8):
    """Calculate affine quantization scale and zero point."""
    qmin = 0.
    qmax = 2. ** num_bits - 1.
    scale = (max_val - min_val) / (qmax - qmin)

    zero_point = qmax - max_val / scale

    if zero_point < qmin:
        zero_point = torch.tensor([qmin], dtype=torch.float32).to(min_val.device)
    elif zero_point > qmax:
        zero_point = torch.tensor([qmax], dtype=torch.float32).to(max_val.device)
    
    zero_point.round_()

    return scale, zero_point

def quantize_tensor(x, scale, zero_point, num_bits=8, signed=False):
    """Map a floating-point tensor to quantized integer-valued codes."""
    if signed:
        qmin = - 2. ** (num_bits - 1)
        qmax = 2. ** (num_bits - 1) - 1
    else:
        qmin = 0.
        qmax = 2. ** num_bits - 1.
 
    q_x = zero_point + x / scale
    q_x.clamp_(qmin, qmax).round_()
    
    return q_x
 
def dequantize_tensor(q_x, scale, zero_point):
    """Map quantized codes back to floating-point values."""
    return scale * (q_x - zero_point)


def search(M):
    """Approximate a floating-point multiplier with an integer and bit shift."""
    P = 7000
    n = 1
    while True:
        Mo = int(round(2 ** n * M))
        approx_result = Mo * P >> n
        result = int(round(M * P))
        error = approx_result - result

        print("n=%d, Mo=%f, approx=%d, result=%d, error=%f" % \
            (n, Mo, approx_result, result, error))

        if math.fabs(error) < 1e-9 or n >= 22:
            return Mo, n
        n += 1


class QParam(nn.Module):
    """Track tensor ranges and the corresponding affine quantization parameters."""

    def __init__(self, num_bits=8):
        super(QParam, self).__init__()
        self.num_bits = num_bits
        scale = torch.tensor([], requires_grad=False)
        zero_point = torch.tensor([], requires_grad=False)
        min = torch.tensor([], requires_grad=False)
        max = torch.tensor([], requires_grad=False)
        self.register_buffer('scale', scale)
        self.register_buffer('zero_point', zero_point)
        self.register_buffer('min', min)
        self.register_buffer('max', max)

    def update(self, tensor):
        # Accumulate calibration extrema while ensuring zero is representable.
        if self.max.nelement() == 0 or self.max.data < tensor.max().data:
            self.max.data = tensor.max().data
        self.max.clamp_(min=0)
        
        if self.min.nelement() == 0 or self.min.data > tensor.min().data:
            self.min.data = tensor.min().data
        self.min.clamp_(max=0)
        
        self.scale, self.zero_point = calcScaleZeroPoint(self.min, self.max, self.num_bits)
    
    def quantize_tensor(self, tensor):
        return quantize_tensor(tensor, self.scale, self.zero_point, num_bits=self.num_bits)

    def dequantize_tensor(self, q_x):
        return dequantize_tensor(q_x, self.scale, self.zero_point)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        # Quantization buffers are initially empty and must adopt checkpoint shapes.
        key_names = ['scale', 'zero_point', 'min', 'max']
        for key in key_names:
            value = getattr(self, key)
            value.data = state_dict[prefix + key].data
            state_dict.pop(prefix + key)

    def __str__(self):
        info = 'scale: %.10f ' % self.scale
        info += 'zp: %d ' % self.zero_point
        info += 'min: %.6f ' % self.min
        info += 'max: %.6f' % self.max
    
        return info



class QModule(nn.Module):
    """Base class for modules with optional input and output quantizers."""

    def __init__(self, qi=True, qo=True, num_bits=8):
        super(QModule, self).__init__()
        if qi:
            self.qi = QParam(num_bits=num_bits)
        if qo:
            self.qo = QParam(num_bits=num_bits)

    def freeze(self):
        pass

    def quantize_inference(self, x):
        raise NotImplementedError('quantize_inference should be implemented.')


class QConv2d(QModule):
    """Quantization wrapper for convolution layers."""

    def __init__(self, conv_module, qi=True, qo=True, num_bits=8):
        super(QConv2d, self).__init__(qi=qi, qo=qo, num_bits=num_bits)
        self.num_bits = num_bits
        self.conv_module = conv_module  
        self.qw = QParam(num_bits=num_bits)

        # M rescales the integer accumulator into the output quantization domain.
        self.register_buffer('M', torch.tensor([], requires_grad=False))
        self.register_buffer('weight_int', torch.tensor([], dtype=torch.uint8))
        self.register_buffer('bias_int', torch.tensor([], dtype=torch.int32))
        self.integer_storage = False

    def freeze(self, qi=None, qo=None):
        """Freeze weights and biases in the output quantization domain."""
        
        if hasattr(self, 'qi') and qi is not None:
            raise ValueError('qi has been provided in init function.')
        if not hasattr(self, 'qi') and qi is None:
            raise ValueError('qi is not existed, should be provided.')

        if hasattr(self, 'qo') and qo is not None:
            raise ValueError('qo has been provided in init function.')
        if not hasattr(self, 'qo') and qo is None:
            raise ValueError('qo is not existed, should be provided.')

        if qi is not None:
            self.qi = qi
        if qo is not None:
            self.qo = qo
        self.M.data = (self.qw.scale  / self.qo.scale).data

        self.conv_module.weight.data = self.qw.quantize_tensor(self.conv_module.weight.data)
        self.conv_module.weight.data = self.conv_module.weight.data - self.qw.zero_point
        # Fold the output rescaling factor into integer-valued parameters.
        self.conv_module.weight.data = (self.conv_module.weight.data * self.M.data).round_()
        self.conv_module.bias.data = quantize_tensor(self.conv_module.bias.data , scale= self.qw.scale,
                                                     zero_point=0, num_bits=self.num_bits, signed=True)
        self.conv_module.bias.data = (self.conv_module.bias.data * self.M.data).round_() + self.qo.zero_point

    def freezeW(self, qi=None, qo=None):
        """Freeze weights as quantized values while preserving their scale."""
        
        if hasattr(self, 'qi') and qi is not None:
            raise ValueError('qi has been provided in init function.')
        if not hasattr(self, 'qi') and qi is None:
            raise ValueError('qi is not existed, should be provided.')

        if hasattr(self, 'qo') and qo is not None:
            raise ValueError('qo has been provided in init function.')
        if not hasattr(self, 'qo') and qo is None:
            raise ValueError('qo is not existed, should be provided.')

        if qi is not None:
            self.qi = qi
        if qo is not None:
            self.qo = qo
        
        self.conv_module.weight.data = self.qw.quantize_tensor(self.conv_module.weight.data)
        self.conv_module.weight.data = self.conv_module.weight.data - self.qw.zero_point
        
        self.conv_module.bias.data = quantize_tensor(self.conv_module.bias.data , scale= self.qw.scale,
                                                     zero_point=0, num_bits=self.num_bits, signed=True)

    def convert_to_integer_storage(self):
        """Move frozen parameters from float containers into integer buffers.

        Weights are stored as unsigned quantization codes and centered by the
        zero point during inference. Biases use int32 accumulator storage.
        PyTorch convolution kernels require floating-point operands, so the
        buffers are cast only at the operation boundary.
        """
        if self.integer_storage:
            return
        if self.num_bits > 16:
            raise ValueError('integer storage currently supports num_bits <= 16')

        storage_dtype = torch.uint8 if self.num_bits <= 8 else torch.int32
        weight_code = self.conv_module.weight.detach() + self.qw.zero_point
        qmax = 2 ** self.num_bits - 1
        if torch.any(weight_code < 0) or torch.any(weight_code > qmax):
            raise ValueError('quantized weight code is outside num_bits range')
        self.weight_int = weight_code.round().to(storage_dtype)
        if self.conv_module.bias is not None:
            self.bias_int = self.conv_module.bias.detach().round().to(torch.int32)

        self.conv_module.register_parameter('weight', None)
        self.conv_module.register_parameter('bias', None)
        self.integer_storage = True

    def _integer_conv(self, x):
        """Run convolution from integer buffers while preserving step dimensions."""
        weight = self.weight_int.to(dtype=x.dtype) - self.qw.zero_point.to(dtype=x.dtype)
        bias = self.bias_int.to(dtype=x.dtype) if self.bias_int.numel() else None
        original_shape = x.shape
        if x.dim() == 5:
            x = x.flatten(0, 1)
        x = F.conv2d(x, weight, bias, self.conv_module.stride,
                     self.conv_module.padding, self.conv_module.dilation,
                     self.conv_module.groups)
        if len(original_shape) == 5:
            x = x.view(original_shape[0], original_shape[1], *x.shape[1:])
        return x
       

    def forward(self, x):
        """Collect calibration ranges and apply fake weight quantization."""
        if hasattr(self, 'qi'):
            self.qi.update(x)

        # Calibrate weights before applying straight-through fake quantization.
        self.qw.update(self.conv_module.weight.data)
        conv_copy = self.conv_module
        conv_copy.weight.data = FakeQuantize.apply(conv_copy.weight, self.qw)
        x = conv_copy(x)

        if hasattr(self, 'qo'):
            self.qo.update(x)

        return x

    def quantize_inference(self, x):
        """Run convolution and clamp its output to the quantized code range."""
        x = self.conv_module(x)
        x.clamp_(0., 2.**self.num_bits-1.).round_()
        return x

    def quantize_inferenceW(self, x):
        """Run convolution with frozen quantized weights."""
        x = self._integer_conv(x) if self.integer_storage else self.conv_module(x)
        return x
    
class QLinear(QModule):
    """Quantization wrapper for fully connected layers."""

    def __init__(self, fc_module, qi=True, qo=True, num_bits=8):
        super(QLinear, self).__init__(qi=qi, qo=qo, num_bits=num_bits)
        self.num_bits = num_bits
        self.fc_module = fc_module
        self.qw = QParam(num_bits=num_bits)
        # M rescales the integer accumulator into the output quantization domain.
        self.register_buffer('M', torch.tensor([], requires_grad=False))
        self.register_buffer('weight_int', torch.tensor([], dtype=torch.uint8))
        self.register_buffer('bias_int', torch.tensor([], dtype=torch.int32))
        self.integer_storage = False
        self.Fla= layer.Flatten()
        
    def freeze(self, qi=None, qo=None):
        """Freeze weights and biases in the output quantization domain."""

        if hasattr(self, 'qi') and qi is not None:
            raise ValueError('qi has been provided in init function.')
        if not hasattr(self, 'qi') and qi is None:
            raise ValueError('qi is not existed, should be provided.')

        if hasattr(self, 'qo') and qo is not None:
            raise ValueError('qo has been provided in init function.')
        if not hasattr(self, 'qo') and qo is None:
            raise ValueError('qo is not existed, should be provided.')

        if qi is not None:
            self.qi = qi
        if qo is not None:
            self.qo = qo
        self.M.data = (self.qw.scale / self.qo.scale).data

        self.fc_module.weight.data = self.qw.quantize_tensor(self.fc_module.weight.data)
        self.fc_module.weight.data = self.fc_module.weight.data - self.qw.zero_point
        # Fold the output rescaling factor into integer-valued parameters.
        self.fc_module.weight.data = (self.fc_module.weight.data * self.M.data).round_()
        self.fc_module.bias.data = quantize_tensor(self.fc_module.bias.data, scale= self.qw.scale,
                                                   zero_point=0, num_bits=self.num_bits, signed=True)
        self.fc_module.bias.data = (self.fc_module.bias.data * self.M.data).round_() + self.qo.zero_point

    def freezeW(self, qi=None, qo=None):
        """Freeze weights as quantized values while preserving their scale."""

        if hasattr(self, 'qi') and qi is not None:
            raise ValueError('qi has been provided in init function.')
        if not hasattr(self, 'qi') and qi is None:
            raise ValueError('qi is not existed, should be provided.')

        if hasattr(self, 'qo') and qo is not None:
            raise ValueError('qo has been provided in init function.')
        if not hasattr(self, 'qo') and qo is None:
            raise ValueError('qo is not existed, should be provided.')

        if qi is not None:
            self.qi = qi
        if qo is not None:
            self.qo = qo
        

        self.fc_module.weight.data = self.qw.quantize_tensor(self.fc_module.weight.data)
        self.fc_module.weight.data = self.fc_module.weight.data - self.qw.zero_point
        
        self.fc_module.bias.data = quantize_tensor(self.fc_module.bias.data, scale= self.qw.scale,
                                                   zero_point=0, num_bits=self.num_bits, signed=True)

    def convert_to_integer_storage(self):
        """Move frozen linear parameters into integer buffers."""
        if self.integer_storage:
            return
        if self.num_bits > 16:
            raise ValueError('integer storage currently supports num_bits <= 16')
        storage_dtype = torch.uint8 if self.num_bits <= 8 else torch.int32
        weight_code = self.fc_module.weight.detach() + self.qw.zero_point
        qmax = 2 ** self.num_bits - 1
        if torch.any(weight_code < 0) or torch.any(weight_code > qmax):
            raise ValueError('quantized weight code is outside num_bits range')
        self.weight_int = weight_code.round().to(storage_dtype)
        if self.fc_module.bias is not None:
            self.bias_int = self.fc_module.bias.detach().round().to(torch.int32)
        self.fc_module.register_parameter('weight', None)
        self.fc_module.register_parameter('bias', None)
        self.integer_storage = True
        

    def forward(self, x):
        """Collect calibration ranges and run fake-quantized linear inference."""
        if hasattr(self, 'qi'):
            self.qi.update(x)
            x = FakeQuantize.apply(x, self.qi)

        self.qw.update(self.fc_module.weight.data)
        fc_copy = self.fc_module
        fc_copy.weight.data = FakeQuantize.apply(fc_copy.weight, self.qw)
        
        x = fc_copy(x)

        if hasattr(self, 'qo'):
            self.qo.update(x)

        return x

    def quantize_inference(self, x):
        """Run the linear layer and clamp to the quantized code range."""
        x = self.fc_module(x)
        x.clamp_(0., 2.**self.num_bits-1.).round_()
        return x

    def quantize_inferenceW(self, x):
        """Run the linear layer with frozen quantized weights."""
        if self.integer_storage:
            # Re-center stored unsigned codes before the linear operation.
            weight = self.weight_int.to(dtype=x.dtype) - self.qw.zero_point.to(dtype=x.dtype)
            bias = self.bias_int.to(dtype=x.dtype) if self.bias_int.numel() else None
            x = F.linear(x, weight, bias)
        else:
            x = self.fc_module(x)
        return x
    

class QIF(QModule):
    """Quantized wrapper for integrate-and-fire neurons."""

    def __init__(self,vth, qi=False, num_bits=None):
        super(QIF, self).__init__(qi=qi, num_bits=num_bits)
        self.IF1= neuron.IFNode(surrogate_function=surrogate.ATan(),v_threshold =vth ,store_v_seq = False )
        self.IF2= neuron.IFNode(surrogate_function=surrogate.ATan(),v_threshold =vth ,store_v_seq = False)
        self.vth = vth
        self.register_buffer('vth_int', torch.tensor([], dtype=torch.int32))

    def convert_to_integer_storage(self, weight_scale):
        """Store the firing threshold in the weight accumulator domain."""
        self.vth_int = torch.round(
            torch.as_tensor(self.vth, device=weight_scale.device) / weight_scale
        ).to(torch.int32)

    def freeze(self, qi=None):
        if hasattr(self, 'qi') and qi is not None:
            raise ValueError('qi has been provided in init function.')
        if not hasattr(self, 'qi') and qi is None:
            raise ValueError('qi is not existed, should be provided.')
        if qi is not None:
            self.qi = qi

    def freezeW(self, qi=None):
        if hasattr(self, 'qi') and qi is not None:
            raise ValueError('qi has been provided in init function.')
        if not hasattr(self, 'qi') and qi is None:
            raise ValueError('qi is not existed, should be provided.')
        if qi is not None:
            self.qi = qi

    def forward(self, x):
        if hasattr(self, 'qi'):
            self.qi.update(x)
            x = FakeQuantize.apply(x, self.qi)
        x = self.IF1(x)
        return x

    def quantize_inference(self, x):
        """Run the IF neuron in the activation quantization domain."""
        vth = self.vth / self.qi.scale
        vth = vth.round_()
        self.IF2.v_threshold = vth
        # Center activations before signed membrane accumulation.
        x = x - self.qi.zero_point
        x = self.IF2(x)

    def quantize_inferenceW(self, x,Sw):
        """Run the IF neuron using a weight-scale-derived threshold."""
        vth = self.vth_int if self.vth_int.numel() else torch.round(
            torch.as_tensor(self.vth, device=Sw.device) / Sw
        )
        self.IF2.v_threshold = vth
        x = self.IF2(x)
        return x

class QMaxPooling2d(QModule):
    """Quantization-aware wrapper for multi-step max pooling."""

    def __init__(self, kernel_size=3, stride=1, padding=0, qi=False, num_bits=None):
        super(QMaxPooling2d, self).__init__(qi=qi, num_bits=num_bits)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.MaxPool1 = layer.MaxPool2d(self.kernel_size, self.stride,self.padding)

    def freeze(self, qi=None):
        if hasattr(self, 'qi') and qi is not None:
            raise ValueError('qi has been provided in init function.')
        if not hasattr(self, 'qi') and qi is None:
            raise ValueError('qi is not existed, should be provided.')
        if qi is not None:
            self.qi = qi

    def freezeW(self, qi=None):
        if hasattr(self, 'qi') and qi is not None:
            raise ValueError('qi has been provided in init function.')
        if not hasattr(self, 'qi') and qi is None:
            raise ValueError('qi is not existed, should be provided.')
        if qi is not None:
            self.qi = qi

    def forward(self, x):
        if hasattr(self, 'qi'):
            self.qi.update(x)
            x = FakeQuantize.apply(x, self.qi)

        x = self.MaxPool1(x)

        return x

    def quantize_inference(self, x):
        x = self.MaxPool1(x)
        return x

    def quantize_inferenceW(self, x):
        x = self.MaxPool1(x)
        return x
    
class QConvBNReLU(QModule):
    """Quantization wrapper for a fused convolution, batch norm, and ReLU."""

    def __init__(self, conv_module, bn_module, qi=True, qo=True, num_bits=8):
        super(QConvBNReLU, self).__init__(qi=qi, qo=qo, num_bits=num_bits)
        self.num_bits = num_bits
        self.conv_module = conv_module
        self.bn_module = bn_module
        self.qw = QParam(num_bits=num_bits)
        self.qb = QParam(num_bits=32)
        # M maps convolution accumulators into the output quantization domain.
        self.register_buffer('M', torch.tensor([], requires_grad=False))

    def fold_bn(self, mean, std):
        """Fold batch-normalization statistics into convolution parameters."""
        if self.bn_module.affine:
            gamma_ = self.bn_module.weight / std
            weight = self.conv_module.weight * gamma_.view(self.conv_module.out_channels, 1, 1, 1)
            if self.conv_module.bias is not None:
                bias = gamma_ * self.conv_module.bias - gamma_ * mean + self.bn_module.bias
            else:
                bias = self.bn_module.bias - gamma_ * mean
        else:
            gamma_ = 1 / std
            weight = self.conv_module.weight * gamma_
            if self.conv_module.bias is not None:
                bias = gamma_ * self.conv_module.bias - gamma_ * mean
            else:
                bias = -gamma_ * mean
            
        return weight, bias


    def forward(self, x):

        if hasattr(self, 'qi'):
            self.qi.update(x)
            x = FakeQuantize.apply(x, self.qi)

        if self.training:
            # Compute per-channel statistics over the batch and spatial axes.
            y = F.conv2d(x, self.conv_module.weight, self.conv_module.bias, 
                            stride=self.conv_module.stride,
                            padding=self.conv_module.padding,
                            dilation=self.conv_module.dilation,
                            groups=self.conv_module.groups)
            y = y.permute(1, 0, 2, 3)  # NCHW -> CNHW
            y = y.contiguous().view(self.conv_module.out_channels, -1)
            mean = y.mean(1).detach()
            var = y.var(1).detach()
            # Match BatchNorm's exponential moving-average update.
            self.bn_module.running_mean = \
                (1 - self.bn_module.momentum) * self.bn_module.running_mean + \
                self.bn_module.momentum * mean
            self.bn_module.running_var = \
                (1 - self.bn_module.momentum) * self.bn_module.running_var + \
                self.bn_module.momentum * var
        else:
            mean = Variable(self.bn_module.running_mean)
            var = Variable(self.bn_module.running_var)

        std = torch.sqrt(var + self.bn_module.eps)

        weight, bias = self.fold_bn(mean, std)

        self.qw.update(weight.data)

        x = F.conv2d(x, FakeQuantize.apply(weight, self.qw), bias, 
                stride=self.conv_module.stride,
                padding=self.conv_module.padding, dilation=self.conv_module.dilation, 
                groups=self.conv_module.groups)

        x = F.relu(x)

        if hasattr(self, 'qo'):
            self.qo.update(x)
            x = FakeQuantize.apply(x, self.qo)

        return x

    def freeze(self, qi=None, qo=None):
        """Fold batch norm and freeze the fused module for integer inference."""
        if hasattr(self, 'qi') and qi is not None:
            raise ValueError('qi has been provided in init function.')
        if not hasattr(self, 'qi') and qi is None:
            raise ValueError('qi is not existed, should be provided.')

        if hasattr(self, 'qo') and qo is not None:
            raise ValueError('qo has been provided in init function.')
        if not hasattr(self, 'qo') and qo is None:
            raise ValueError('qo is not existed, should be provided.')

        if qi is not None:
            self.qi = qi
        if qo is not None:
            self.qo = qo
        self.M.data = (self.qw.scale * self.qi.scale / self.qo.scale).data

        std = torch.sqrt(self.bn_module.running_var + self.bn_module.eps)

        weight, bias = self.fold_bn(self.bn_module.running_mean, std)
        self.conv_module.weight.data = self.qw.quantize_tensor(weight.data)
        self.conv_module.weight.data = self.conv_module.weight.data - self.qw.zero_point

        self.conv_module.bias.data = quantize_tensor(bias, scale=self.qi.scale * self.qw.scale,
                                                     zero_point=0, num_bits=32, signed=True)

    def quantize_inference(self, x):
        """Run the frozen fused module in quantized activation space."""
        x = x - self.qi.zero_point
        x = self.conv_module(x)
        x = self.M * x
        x.round_() 
        x = x + self.qo.zero_point        
        x.clamp_(0., 2.**self.num_bits-1.).round_()
        return x
        

class QSigmoid(QModule):
    """Quantized sigmoid implemented with a calibrated lookup table."""

    def __init__(self, qi=True, qo=True, num_bits=8, lut_size=64):
        super(QSigmoid, self).__init__(qi=qi, qo=qo, num_bits=num_bits)
        self.num_bits = num_bits
        self.lut_size = lut_size
    
    def forward(self, x):
        if hasattr(self, 'qi'):
            self.qi.update(x)
            x = FakeQuantize.apply(x, self.qi)

        x = torch.sigmoid(x)

        if hasattr(self, 'qo'):
            self.qo.update(x)
            x = FakeQuantize.apply(x, self.qo)

        return x
    
    def freeze(self, qi=None, qo=None):
        """Build the sigmoid lookup table from frozen quantization ranges."""
        if hasattr(self, 'qi') and qi is not None:
            raise ValueError('qi has been provided in init function.')
        if not hasattr(self, 'qi') and qi is None:
            raise ValueError('qi is not existed, should be provided.')

        if hasattr(self, 'qo') and qo is not None:
            raise ValueError('qo has been provided in init function.')
        if not hasattr(self, 'qo') and qo is None:
            raise ValueError('qo is not existed, should be provided.')

        if qi is not None:
            self.qi = qi
        if qo is not None:
            self.qo = qo

        # Uniformly sample input codes and quantize their sigmoid responses.
        lut_qx = torch.tensor(np.linspace(0, 2 ** self.num_bits - 1, self.lut_size), dtype=torch.uint8)
        lut_x = self.qi.dequantize_tensor(lut_qx)
        lut_y = torch.sigmoid(lut_x)
        lut_qy = self.qo.quantize_tensor(lut_y)

        self.register_buffer('lut_qy', lut_qy)
        self.register_buffer('lut_qx', lut_qx)


    def quantize_inference(self, x):
        """Approximate sigmoid by interpolation in the quantized lookup table."""
        y = interp(x, self.lut_qx, self.lut_qy)
        y = y.round_().clamp_(0., 2.**self.num_bits-1.)
        return y
