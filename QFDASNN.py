"""Post-training quantization and integer inference for FDASNN."""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import functional, layer, neuron, surrogate
from torchvision import transforms
from tqdm import tqdm
import parameter
from dataloader import AugMNIST, center_crop_if_oversized
from module import QConv2d, QIF, QLinear, QMaxPooling2d, set_seed
from parameter import spiking

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _prepare_batch(images, labels):
    """Generate and distance-normalize spikes for one labeled batch."""
    images = images.to(device, non_blocking=True)
    # Augmented labels encode class * 10 + simulated distance.
    class_labels = (labels // 10).long().to(device, non_blocking=True)
    distances = (labels % 10).to(device, non_blocking=True)
    spike_sequence = spiking(images, distances)

    # Preserve the original flow: use exact distances for SPAD simulation and
    # integer distances only for the crop-and-resize operation.
    crop_distances = torch.floor(distances)
    height, width = images.shape[-2:]
    crop_sizes = width * (crop_distances - 1) // (2 * crop_distances)
    normalized_samples = []

    # Remove distance padding before restoring the fixed field of view.
    for sample_index, distance in enumerate(crop_distances):
        crop_size = int(crop_sizes[sample_index].item())
        cropped = transforms.functional.crop(
            spike_sequence[:, sample_index],
            top=crop_size,
            left=crop_size,
            height=height - 2 * crop_size,
            width=width - 2 * crop_size,
        )
        upsampled = F.interpolate(cropped, scale_factor=float(distance), mode="nearest")
        upsampled = center_crop_if_oversized(
            upsampled, target_height=height, target_width=width
        )
        normalized_samples.append(upsampled)

    normalized_spikes = torch.stack(normalized_samples).transpose(0, 1).float()
    class_targets = F.one_hot(class_labels, 10).float()
    return normalized_spikes, class_labels, class_targets


class QFDASNN(nn.Module):
    """FDASNN with fake-quantization and integer-inference paths."""

    def __init__(self, T: int, channels: int):
        super().__init__()
        self.T = T
        self.vth = 1.0

        self.conv0 = layer.Conv2d(1, channels, kernel_size=5, stride=2, padding=2)
        self.IF0 = neuron.IFNode(
            surrogate_function=surrogate.ATan(), v_threshold=self.vth, store_v_seq=False
        )

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
            channels // 2, channels // 2, kernel_size=5, stride=3, padding=2, groups=2
        )
        self.IF13 = neuron.IFNode(
            surrogate_function=surrogate.ATan(), v_threshold=self.vth, store_v_seq=False
        )

        self.conv14 = layer.Conv2d(
            channels // 2, channels, kernel_size=3, stride=1, padding=1, groups=2
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

        self.conv2 = layer.Conv2d(
            channels, channels * 2, kernel_size=3, stride=2, padding=1
        )
        self.IF2 = neuron.IFNode(
            surrogate_function=surrogate.ATan(), v_threshold=self.vth, store_v_seq=False
        )

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
            channels, channels, kernel_size=5, stride=3, padding=1, groups=4
        )
        self.IF33 = neuron.IFNode(
            surrogate_function=surrogate.ATan(), v_threshold=self.vth, store_v_seq=False
        )

        self.conv34 = layer.Conv2d(
            channels, channels * 2, kernel_size=3, stride=1, padding=1, groups=4
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

        self.conv4 = layer.Conv2d(channels * 2, channels // 2, kernel_size=3, padding=1)
        self.IF4 = neuron.IFNode(
            surrogate_function=surrogate.ATan(), v_threshold=self.vth, store_v_seq=False
        )

        self.Fla = layer.Flatten()

        self.fc5 = layer.Linear(72 * 2, 10)
        functional.set_step_mode(self, step_mode="m")

    def forward(self, x: torch.Tensor):
        """Return full-precision per-time-step logits."""

        x = self.conv0(x)
        x = self.IF0(x)

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

    def quantize(self, num_bits=8):
        """Attach quantization wrappers to the trained layers."""
        # Connected wrappers later share adjacent calibration parameters.
        self.qconv0 = QConv2d(self.conv0, qi=True, qo=True, num_bits=num_bits)
        self.qIF0 = QIF(self.vth)

        self.qconv11 = QConv2d(self.conv11, qi=False, qo=True, num_bits=num_bits)
        self.qIF11 = QIF(self.vth)

        self.qconv12 = QConv2d(self.conv12, qi=False, qo=True, num_bits=num_bits)
        self.qIF12 = QIF(self.vth)

        self.qmaxpool1 = QMaxPooling2d(kernel_size=3, stride=3, padding=0)
        self.qconv13 = QConv2d(self.conv13, qi=False, qo=True, num_bits=num_bits)
        self.qIF13 = QIF(self.vth)

        self.qconv14 = QConv2d(self.conv14, qi=False, qo=True, num_bits=num_bits)
        self.qIF14 = QIF(self.vth)

        self.qconv15 = QConv2d(self.conv15, qi=False, qo=True, num_bits=num_bits)
        self.qIF15 = QIF(self.vth)

        self.qconv16 = QConv2d(self.conv16, qi=False, qo=True, num_bits=num_bits)
        self.qIF16 = QIF(self.vth)

        self.qconv2 = QConv2d(self.conv2, qi=False, qo=True, num_bits=num_bits)
        self.qIF2 = QIF(self.vth)

        self.qconv31 = QConv2d(self.conv31, qi=False, qo=True, num_bits=num_bits)
        self.qIF31 = QIF(self.vth)

        self.qconv32 = QConv2d(self.conv32, qi=False, qo=True, num_bits=num_bits)
        self.qIF32 = QIF(self.vth)

        self.qmaxpool3 = QMaxPooling2d(kernel_size=3, stride=3, padding=0)
        self.qconv33 = QConv2d(self.conv33, qi=False, qo=True, num_bits=num_bits)
        self.qIF33 = QIF(self.vth)

        self.qconv34 = QConv2d(self.conv34, qi=False, qo=True, num_bits=num_bits)
        self.qIF34 = QIF(self.vth)

        self.qconv35 = QConv2d(self.conv35, qi=False, qo=True, num_bits=num_bits)
        self.qIF35 = QIF(self.vth)

        self.qconv36 = QConv2d(self.conv36, qi=False, qo=True, num_bits=num_bits)
        self.qIF36 = QIF(self.vth)

        self.qconv4 = QConv2d(self.conv4, qi=False, qo=True, num_bits=num_bits)
        self.qIF4 = QIF(self.vth)

        self.qfc5 = QLinear(self.fc5, qi=False, qo=True, num_bits=num_bits)
        functional.set_step_mode(self, step_mode="m")

    def quantize_forward(self, x):
        """Run fake-quantized inference while collecting quantization ranges."""

        # Fake quantization retains floating-point execution during calibration.
        x = self.qconv0(x)
        x = self.qIF0(x)

        x = self.qconv11(x)
        y0 = x
        y0 = self.qmaxpool1(y0)

        x = self.qIF11(x)
        x = self.qconv12(x)
        x = self.qIF12(x)

        x = self.qconv13(x)
        x = self.qIF13(x)
        x = self.qconv14(x)
        x = self.qIF14(x)

        x = self.qconv15(x)
        x = self.qIF15(x + y0)

        x = self.qconv16(x)
        x = self.qIF16(x)

        x = self.qconv2(x)
        x = self.qIF2(x)

        x = self.qconv31(x)
        y1 = x
        y1 = self.qmaxpool3(y1)

        x = self.qIF31(x)
        x = self.qconv32(x)
        x = self.qIF32(x)

        x = self.qconv33(x)
        x = self.qIF33(x)
        x = self.qconv34(x)
        x = self.qIF34(x)

        x = self.qconv35(x)
        x = self.qIF35(x + y1)

        x = self.qconv36(x)
        x = self.qIF36(x)

        x = self.qconv4(x)
        x = self.qIF4(x)

        x = self.Fla(x)

        x = self.qfc5(x)

        return x

    def freezeW(self):
        """Freeze quantization parameters and convert weights for inference."""

        # Propagate output quantizers so connected layers use consistent scales.
        self.qconv0.freezeW()
        self.qIF0.freezeW(self.qconv0.qo)

        self.qconv11.freezeW(qi=self.qconv0.qo)
        self.qIF11.freezeW(self.qconv11.qo)
        self.qmaxpool1.freezeW(self.qconv11.qo)

        self.qconv12.freezeW(qi=self.qconv11.qo)
        self.qIF12.freezeW(self.qconv12.qo)

        self.qconv13.freezeW(qi=self.qconv12.qo)
        self.qIF13.freezeW(self.qconv13.qo)

        self.qconv14.freezeW(qi=self.qconv13.qo)
        self.qIF14.freezeW(self.qconv14.qo)

        self.qconv15.freezeW(qi=self.qconv14.qo)
        self.qIF15.freezeW(self.qconv15.qo)

        self.qconv16.freezeW(qi=self.qconv15.qo)
        self.qIF16.freezeW(self.qconv16.qo)

        self.qconv2.freezeW(qi=self.qconv16.qo)
        self.qIF2.freezeW(self.qconv2.qo)

        self.qconv31.freezeW(qi=self.qconv2.qo)
        self.qIF31.freezeW(self.qconv31.qo)
        self.qmaxpool3.freezeW(self.qconv31.qo)

        self.qconv32.freezeW(qi=self.qconv31.qo)
        self.qIF32.freezeW(self.qconv32.qo)

        self.qconv33.freezeW(qi=self.qconv32.qo)
        self.qIF33.freezeW(self.qconv33.qo)

        self.qconv34.freezeW(qi=self.qconv33.qo)
        self.qIF34.freezeW(self.qconv34.qo)

        self.qconv35.freezeW(qi=self.qconv34.qo)
        self.qIF35.freezeW(self.qconv35.qo)

        self.qconv36.freezeW(qi=self.qconv35.qo)
        self.qIF36.freezeW(self.qconv36.qo)

        self.qconv4.freezeW(qi=self.qconv36.qo)
        self.qIF4.freezeW(self.qconv4.qo)

        self.qfc5.freezeW(qi=self.qconv4.qo)

    def quantize_inferenceW(self, x, belta):
        """Run integer inference with residual scale balancing."""

        # Activations remain integer-valued until the final classifier output.
        qx = self.qconv0.quantize_inferenceW(x)
        qx = self.qIF0.quantize_inferenceW(qx, self.qconv0.qw.scale)

        qx = self.qconv11.quantize_inferenceW(qx)
        y0 = qx
        y0 = self.qmaxpool1.quantize_inferenceW(y0)

        qx = self.qIF11.quantize_inferenceW(qx, self.qconv11.qw.scale)
        qx = self.qconv12.quantize_inferenceW(qx)
        qx = self.qIF12.quantize_inferenceW(qx, self.qconv12.qw.scale)

        qx = self.qconv13.quantize_inferenceW(qx)
        qx = self.qIF13.quantize_inferenceW(qx, self.qconv13.qw.scale)
        qx = self.qconv14.quantize_inferenceW(qx)
        qx = self.qIF14.quantize_inferenceW(qx, self.qconv14.qw.scale)

        qx = self.qconv15.quantize_inferenceW(qx)

        m = belta

        # Align main and residual branches before their integer addition.
        if self.qconv11.qw.scale / self.qconv15.qw.scale >= 1:
            y0 = y0 * torch.round(self.qconv11.qw.scale / self.qconv15.qw.scale * m)
            qx = self.qIF15.quantize_inferenceW(qx * m + y0, self.qconv15.qw.scale / m)
        else:
            qx = qx * torch.round(self.qconv15.qw.scale / self.qconv11.qw.scale * m)
            qx = self.qIF15.quantize_inferenceW(qx + y0 * m, self.qconv11.qw.scale / m)

        qx = self.qconv16.quantize_inferenceW(qx)
        qx = self.qIF16.quantize_inferenceW(qx, self.qconv16.qw.scale)

        qx = self.qconv2.quantize_inferenceW(qx)
        qx = self.qIF2.quantize_inferenceW(qx, self.qconv2.qw.scale)

        qx = self.qconv31.quantize_inferenceW(qx)
        y1 = qx
        y1 = self.qmaxpool3.quantize_inferenceW(y1)

        qx = self.qIF31.quantize_inferenceW(qx, self.qconv31.qw.scale)
        qx = self.qconv32.quantize_inferenceW(qx)
        qx = self.qIF32.quantize_inferenceW(qx, self.qconv32.qw.scale)

        qx = self.qconv33.quantize_inferenceW(qx)
        qx = self.qIF33.quantize_inferenceW(qx, self.qconv33.qw.scale)
        qx = self.qconv34.quantize_inferenceW(qx)
        qx = self.qIF34.quantize_inferenceW(qx, self.qconv34.qw.scale)

        qx = self.qconv35.quantize_inferenceW(qx)

        n = belta

        # Apply the same alignment to the second residual block.
        if self.qconv31.qw.scale / self.qconv35.qw.scale >= 1:
            y1 = y1 * torch.round(self.qconv31.qw.scale / self.qconv35.qw.scale * n)
            qx = self.qIF35.quantize_inferenceW(qx * n + y1, self.qconv35.qw.scale / n)
        else:
            qx = qx * torch.round(self.qconv35.qw.scale / self.qconv31.qw.scale * n)
            qx = self.qIF35.quantize_inferenceW(qx + y1 * n, self.qconv31.qw.scale / n)

        qx = self.qconv36.quantize_inferenceW(qx)
        qx = self.qIF36.quantize_inferenceW(qx, self.qconv36.qw.scale)

        qx = self.qconv4.quantize_inferenceW(qx)
        qx = self.qIF4.quantize_inferenceW(qx, self.qconv4.qw.scale)

        qx = self.Fla(qx)

        qx = self.qfc5.quantize_inferenceW(qx)

        qx = qx * self.qfc5.qw.scale
        return qx


def direct_quantize(model, test_loader):
    """Calibrate quantization ranges and report validation accuracy."""

    test_loss = 0
    test_acc = 0
    test_samples = 0
    for img, label in tqdm(test_loader):
        stacked_tensor, label1, label_targets = _prepare_batch(img, label)

        out_fr = model.quantize_forward(stacked_tensor)
        out_fr = out_fr.mean(0)
        loss = F.cross_entropy(out_fr, label_targets)

        test_samples += label.numel()
        test_loss += loss.item() * label.numel()
        test_acc += (out_fr.argmax(1) == label1).float().sum().item()

        functional.reset_net(model)

    test_loss /= test_samples
    test_acc /= test_samples
    print(
        f"\ndirect quantization finish:  test_loss ={test_loss: .4f}, test_acc ={test_acc: .4f}"
    )


def quantize_inferenceW(model, test_loader, belta):
    """Evaluate the frozen integer model."""
    test_loss = 0
    test_acc = 0
    test_samples = 0
    for img, label in tqdm(test_loader):
        stacked_tensor, label1, label_targets = _prepare_batch(img, label)
        out_fr = model.quantize_inferenceW(stacked_tensor, belta)
        out_fr = out_fr.mean(0)
        loss = F.cross_entropy(out_fr, label_targets)

        test_samples += label.numel()
        test_loss += loss.item() * label.numel()
        test_acc += (out_fr.argmax(1) == label1).float().sum().item()

        functional.reset_net(model)
    test_loss /= test_samples
    test_acc /= test_samples
    print(
        f"\nTest set: Quant Model Accuracy:  test_loss ={test_loss: .4f}, test_acc ={test_acc: .4f}"
    )


def full_inference(model, test_loader):
    """Evaluate the full-precision model."""
    test_loss = 0
    test_acc = 0
    test_samples = 0
    for img, label in tqdm(test_loader):
        stacked_tensor, label1, label_targets = _prepare_batch(img, label)
        out_fr = model(stacked_tensor)
        out_fr = out_fr.mean(0)
        loss = F.cross_entropy(out_fr, label_targets)

        test_samples += label.numel()
        test_loss += loss.item() * label.numel()
        test_acc += (out_fr.argmax(1) == label1).float().sum().item()
        functional.reset_net(model)

    test_loss /= test_samples
    test_acc /= test_samples

    print(
        f"\nTest set: Full Model Accuracy: test_loss ={test_loss: .4f}, test_acc ={test_acc: .4f}"
    )


def cut_inferenceW(model, test_loader, num_decimal):
    """Truncate parameters to a fixed number of decimals and evaluate."""
    test_loss = 0
    test_acc = 0
    test_samples = 0
    num_decimal = 10**num_decimal
    params_list = []
    with torch.no_grad():

        for param in model.parameters():

            param.data = torch.floor(param.data * num_decimal) / num_decimal

            params_list.append(param.data.flatten())

    all_params = torch.cat(params_list)

    max_val = all_params.max().item()
    min_val = all_params.min().item()

    print(f"Maximum parameter value: {max_val:.2f}")
    print(f"Minimum parameter value: {min_val:.2f}")

    for img, label in tqdm(test_loader):
        stacked_tensor, label1, label_targets = _prepare_batch(img, label)

        out_fr = model(stacked_tensor)
        out_fr = out_fr.mean(0)
        loss = F.cross_entropy(out_fr, label_targets)

        test_samples += label.numel()
        test_loss += loss.item() * label.numel()
        test_acc += (out_fr.argmax(1) == label1).float().sum().item()
        functional.reset_net(model)

    test_loss /= test_samples
    test_acc /= test_samples

    print(
        f"\nTest set: Cut Model Accuracy: test_loss ={test_loss: .4f}, test_acc ={test_acc: .4f}"
    )


if __name__ == "__main__":

    set_seed(130)
    bin_number = parameter.bin_number

    batchsize = 32
    channels = 8
    xsize = 240
    ysize = 240

    Qnet = QFDASNN(T=bin_number, channels=channels)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    Qnet = Qnet.to(device)

    _, test_data_loader = AugMNIST(xsize, ysize, batchsize)

    Qnet.load_state_dict(
        torch.load(os.path.join(os.path.dirname(__file__), "FDASNN.pth"))
    )
    num_bits = 4

    Qnet.eval()

    # Calibrate ranges, freeze integer parameters, then evaluate inference.
    Qnet.quantize(num_bits=num_bits)
    Qnet.eval()
    print("Quantization bit: %d" % num_bits)
    direct_quantize(Qnet, test_data_loader)

    Qnet.freezeW()

    Qnet.eval()
    quantize_inferenceW(Qnet, test_data_loader, belta=2)
