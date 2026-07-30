import math
import numpy as np
from spikingjelly.activation_based import neuron, surrogate, layer 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from function import FakeQuantize, interp


# 固定随机种子
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # 如果使用numpy，也需要固定numpy的种子
    np.random.seed(seed)
 

class MultiAttention(nn.Module):
    def __init__(self, T, reduction_t, reduction_c, kernel_size, C):
       
        super().__init__()

        assert T >= reduction_t, 'reduction_t cannot be greater than T'
        assert C >= reduction_c, 'reduction_c cannot be greater than C'
        
        from einops import rearrange
        
        # Attention
        class TimeAttention(nn.Module):
            def __init__(self, in_planes, ratio=16):
                super(TimeAttention, self).__init__()
                self.avg_pool = nn.AdaptiveAvgPool3d(1)
                self.max_pool = nn.AdaptiveMaxPool3d(1)
                # 三维卷积，将T维当作通道维度处理
                self.conv1 = nn.Conv3d(in_planes, in_planes // ratio, 1, bias=False)
                self.relu = nn.ReLU()
                self.conv2 = nn.Conv3d(in_planes // ratio, in_planes, 1, bias=False)

                self.sigmoid = nn.Sigmoid()

            def forward(self, x):
                avgout = self.avg_pool(x)
                avgout = self.conv1(avgout)
                avgout = self.relu(avgout)
                avgout = self.conv2(avgout)

                maxout = self.max_pool(x)
                maxout = self.conv1(maxout)
                maxout = self.relu(maxout)
                maxout = self.conv2(maxout)

                return self.sigmoid(avgout + maxout)


        class ChannelAttention(nn.Module):
            def __init__(self, in_planes, ratio=16):
                super(ChannelAttention, self).__init__()
                self.avg_pool = nn.AdaptiveAvgPool3d(1)
                self.max_pool = nn.AdaptiveMaxPool3d(1)
                self.sharedMLP = nn.Sequential(
                    nn.Conv3d(in_planes, in_planes // ratio, 1, bias=False),
                    nn.ReLU(),
                    nn.Conv3d(in_planes // ratio, in_planes, 1, bias=False),
                )
                self.sigmoid = nn.Sigmoid()
                
            def forward(self, x):
                x = rearrange(x, "b f c h w -> b c f h w")
                avgout = self.sharedMLP(self.avg_pool(x))
                maxout = self.sharedMLP(self.max_pool(x))
                out = self.sigmoid(avgout + maxout)
                out = rearrange(out, "b c f h w -> b f c h w")
                return out


        class SpatialAttention(nn.Module):
            def __init__(self, kernel_size=3):
                super(SpatialAttention, self).__init__()
                assert kernel_size in (3, 7), "kernel size must be 3 or 7"
                padding = 3 if kernel_size == 7 else 1
                self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
                self.sigmoid = nn.Sigmoid()

            def forward(self, x):
                x = rearrange(x, "b f c h w -> b (f c) h w")
                avgout = torch.mean(x, dim=1, keepdim=True)
                maxout, _ = torch.max(x, dim=1, keepdim=True)
                x = torch.cat([avgout, maxout], dim=1)
                x = self.conv(x)
                x = x.unsqueeze(1)
                return self.sigmoid(x)
            
        self.ta = TimeAttention(T, reduction_t)
        self.ca = ChannelAttention(C, reduction_c)
        self.sa = SpatialAttention(kernel_size)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor):
        assert x.dim() == 5, ValueError(
            f'expected 5D input with shape [T, N, C, H, W], but got input with shape {x.shape}')
        x = x.transpose(0, 1)
        out = self.ta(x) * x
        out = self.ca(out) * out
        out = self.sa(out) * out
        out = self.relu(out)
        out = out.transpose(0, 1)
        return out



def calcScaleZeroPoint(min_val, max_val, num_bits=8):
    qmin = 0.
    qmax = 2. ** num_bits - 1.
    scale = (max_val - min_val) / (qmax - qmin)

    zero_point = qmax - max_val / scale

    if zero_point < qmin:
        zero_point = torch.tensor([qmin], dtype=torch.float32).to(min_val.device)
    elif zero_point > qmax:
        # zero_point = qmax
        zero_point = torch.tensor([qmax], dtype=torch.float32).to(max_val.device)
    
    zero_point.round_()

    return scale, zero_point

def quantize_tensor(x, scale, zero_point, num_bits=8, signed=False):
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
    return scale * (q_x - zero_point)


def search(M):
    P = 7000
    n = 1
    while True:
        Mo = int(round(2 ** n * M))
        # Mo 
        approx_result = Mo * P >> n
        result = int(round(M * P))
        error = approx_result - result

        print("n=%d, Mo=%f, approx=%d, result=%d, error=%f" % \
            (n, Mo, approx_result, result, error))

        if math.fabs(error) < 1e-9 or n >= 22:
            return Mo, n
        n += 1


class QParam(nn.Module):

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
        if self.max.nelement() == 0 or self.max.data < tensor.max().data:
            self.max.data = tensor.max().data
        self.max.clamp_(min=0)
        
        if self.min.nelement() == 0 or self.min.data > tensor.min().data:
            self.min.data = tensor.min().data
        self.min.clamp_(max=0)
        
#         self.min.data = torch.min(self.min.data, tensor.min().data)  
#         self.max.data = torch.max(self.max.data, tensor.max().data)  
        
        self.scale, self.zero_point = calcScaleZeroPoint(self.min, self.max, self.num_bits)
    
    def quantize_tensor(self, tensor):
        return quantize_tensor(tensor, self.scale, self.zero_point, num_bits=self.num_bits)

    def dequantize_tensor(self, q_x):
        return dequantize_tensor(q_x, self.scale, self.zero_point)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
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

    def __init__(self, conv_module, qi=True, qo=True, num_bits=8):
        super(QConv2d, self).__init__(qi=qi, qo=qo, num_bits=num_bits)
        self.num_bits = num_bits
        self.conv_module = conv_module  
        self.qw = QParam(num_bits=num_bits)
        #self.conv1 = layer.Conv2d(self.conv_module.in_channels,self.conv_module.out_channels,kernel_size=self.conv_module.kernel_size[0], padding=self.conv_module.padding)

        self.register_buffer('M', torch.tensor([], requires_grad=False))  # 将M注册为buffer
        self.register_buffer('weight_int', torch.tensor([], dtype=torch.uint8))
        self.register_buffer('bias_int', torch.tensor([], dtype=torch.int32))
        self.integer_storage = False

    def freeze(self, qi=None, qo=None):
        
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
        self.conv_module.weight.data = (self.conv_module.weight.data * self.M.data).round_()    #新权重取整
        self.conv_module.bias.data = quantize_tensor(self.conv_module.bias.data , scale= self.qw.scale,
                                                     zero_point=0, num_bits=self.num_bits, signed=True) #16位应该也可以
        self.conv_module.bias.data = (self.conv_module.bias.data * self.M.data).round_() + self.qo.zero_point    #新偏置取整

    def freezeW(self, qi=None, qo=None):
        
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
                                                     zero_point=0, num_bits=self.num_bits, signed=True) #16位应该也可以

    def convert_to_integer_storage(self):
        """将 freezeW 后的整数值从 float Parameter 转为真正的整型 buffer。

        权重保存为无符号量化码，计算时再减去 zero_point，因此与
        freezeW() 原有的浮点容器整数值完全等价。bias 按累加器常用的 int32
        保存。PyTorch 通用 CUDA conv 不接受整型权重，所以只在卷积运算
        边界临时转换为输入 dtype，持久模型中不再保留浮点权重副本。"""
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
        if hasattr(self, 'qi'):
            self.qi.update(x)
           # x = FakeQuantize.apply(x, self.qi)  #量化后反量化

        self.qw.update(self.conv_module.weight.data)#将max min s z 统计好放入qw Qaram中
        conv_copy = self.conv_module
        conv_copy.weight.data = FakeQuantize.apply(conv_copy.weight, self.qw)
       # self.conv1.bias.data = self.conv_module.bias.data
        x = conv_copy(x)
       # x = self.conv1(x)

        if hasattr(self, 'qo'):
            self.qo.update(x)
           # x = FakeQuantize.apply(x, self.qo)

        return x

    def quantize_inference(self, x): #利用公式4卷积
       # x = x - self.qi.zero_point  zero看作0
        x = self.conv_module(x)
       # x = self.M * x

       # x.round_() 
       # x = x + self.qo.zero_point        
        x.clamp_(0., 2.**self.num_bits-1.).round_()
        return x

    def quantize_inferenceW(self, x): #利用公式4卷积
       # x = x - self.qi.zero_point  zero看作0
        x = self._integer_conv(x) if self.integer_storage else self.conv_module(x)
       # x = self.M * x

       # x.round_() 
       # x = x + self.qo.zero_point        
        return x
    
class QLinear(QModule):

    def __init__(self, fc_module, qi=True, qo=True, num_bits=8):
        super(QLinear, self).__init__(qi=qi, qo=qo, num_bits=num_bits)
        self.num_bits = num_bits
        self.fc_module = fc_module
        self.qw = QParam(num_bits=num_bits)
        self.register_buffer('M', torch.tensor([], requires_grad=False))  # 将M注册为buffer
        self.register_buffer('weight_int', torch.tensor([], dtype=torch.uint8))
        self.register_buffer('bias_int', torch.tensor([], dtype=torch.int32))
        self.integer_storage = False
        self.Fla= layer.Flatten()
        #self.fc = layer.Linear(4 * 7 * 7, 10)
        
    def freeze(self, qi=None, qo=None):

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
        self.fc_module.weight.data = (self.fc_module.weight.data * self.M.data).round_()    #新权重取整
        self.fc_module.bias.data = quantize_tensor(self.fc_module.bias.data, scale= self.qw.scale,
                                                   zero_point=0, num_bits=self.num_bits, signed=True)
        self.fc_module.bias.data = (self.fc_module.bias.data * self.M.data).round_() + self.qo.zero_point    #新偏置取整

    def freezeW(self, qi=None, qo=None):

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
        if hasattr(self, 'qi'):
            self.qi.update(x)
            x = FakeQuantize.apply(x, self.qi)

        self.qw.update(self.fc_module.weight.data)
        fc_copy = self.fc_module
        fc_copy.weight.data = FakeQuantize.apply(fc_copy.weight, self.qw)
        
        x = fc_copy(x)

        if hasattr(self, 'qo'):
            self.qo.update(x)
            # x = FakeQuantize.apply(x, self.qo)

        return x

    def quantize_inference(self, x):
       # x = x - self.qi.zero_point
        x = self.fc_module(x)
        # x = self.M * x
        # x.round_() 
        # x = x + self.qo.zero_point
        x.clamp_(0., 2.**self.num_bits-1.).round_()
        return x
    
    def quantize_inferenceW(self, x):
       # x = x - self.qi.zero_point
        if self.integer_storage:
            weight = self.weight_int.to(dtype=x.dtype) - self.qw.zero_point.to(dtype=x.dtype)
            bias = self.bias_int.to(dtype=x.dtype) if self.bias_int.numel() else None
            x = F.linear(x, weight, bias)
        else:
            x = self.fc_module(x)
        # x = self.M * x
        # x.round_() 
        # x = x + self.qo.zero_point
        
        return x
    

class QIF(QModule):
    def __init__(self,vth, qi=False, num_bits=None):
        super(QIF, self).__init__(qi=qi, num_bits=num_bits)
        self.IF1= neuron.IFNode(surrogate_function=surrogate.ATan(),v_threshold =vth ,store_v_seq = False )#store_v_seq = False
        self.IF2= neuron.IFNode(surrogate_function=surrogate.ATan(),v_threshold =vth ,store_v_seq = False)
        self.vth = vth
        self.register_buffer('vth_int', torch.tensor([], dtype=torch.int32))

    def convert_to_integer_storage(self, weight_scale):
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
        if hasattr(self, 'qi'):           # 未执行
            self.qi.update(x)
            x = FakeQuantize.apply(x, self.qi)  #可以试一下，对v_threshold量化+反量化                                                                      
        x = self.IF1(x)
        return x
    
    def quantize_inference(self, x):  #IF神经元加油！
        vth = self.vth / self.qi.scale
        vth = vth.round_()  # 
        self.IF2.v_threshold = vth
        x = x - self.qi.zero_point  #对准0，涉及负数累加 ，不能量化成0-255
        x = self.IF2(x)

    def quantize_inferenceW(self, x,Sw):  #IF神经元加油！
        vth = self.vth_int if self.vth_int.numel() else torch.round(
            torch.as_tensor(self.vth, device=Sw.device) / Sw
        )
        self.IF2.v_threshold = vth
        x = self.IF2(x)
        return x

class QMaxPooling2d(QModule):

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

    def __init__(self, conv_module, bn_module, qi=True, qo=True, num_bits=8):
        super(QConvBNReLU, self).__init__(qi=qi, qo=qo, num_bits=num_bits)
        self.num_bits = num_bits
        self.conv_module = conv_module
        self.bn_module = bn_module
        self.qw = QParam(num_bits=num_bits)
        self.qb = QParam(num_bits=32)
        self.register_buffer('M', torch.tensor([], requires_grad=False))  # 将M注册为buffer

    def fold_bn(self, mean, std):
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
            y = F.conv2d(x, self.conv_module.weight, self.conv_module.bias, 
                            stride=self.conv_module.stride,
                            padding=self.conv_module.padding,
                            dilation=self.conv_module.dilation,
                            groups=self.conv_module.groups)
            y = y.permute(1, 0, 2, 3) # NCHW -> CNHW
            y = y.contiguous().view(self.conv_module.out_channels, -1) # CNHW -> C,NHW
            # mean = y.mean(1)
            # var = y.var(1)
            mean = y.mean(1).detach()
            var = y.var(1).detach()
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
        x = x - self.qi.zero_point
        x = self.conv_module(x)
        x = self.M * x
        x.round_() 
        x = x + self.qo.zero_point        
        x.clamp_(0., 2.**self.num_bits-1.).round_()
        return x
        

class QSigmoid(QModule):

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

        lut_qx = torch.tensor(np.linspace(0, 2 ** self.num_bits - 1, self.lut_size), dtype=torch.uint8)
        lut_x = self.qi.dequantize_tensor(lut_qx)
        lut_y = torch.sigmoid(lut_x)
        lut_qy = self.qo.quantize_tensor(lut_y)

        self.register_buffer('lut_qy', lut_qy)
        self.register_buffer('lut_qx', lut_qx)


    def quantize_inference(self, x):
        y = interp(x, self.lut_qx, self.lut_qy)
        y = y.round_().clamp_(0., 2.**self.num_bits-1.)
        return y
