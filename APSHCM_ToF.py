"""Evaluate SPAD time-of-flight estimation on distance-scaled MNIST images."""

import csv
import math
import os
import numpy as np
import torch
import torchvision
from scipy.stats import norm
from spikingjelly.activation_based import encoding
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
import parameter


def set_seed(seed):
    """Set random seeds for reproducible experiments."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


xsize = 240
ysize = 240
min_grey_value = parameter.min_grey_value
batch_size = 100


class CustomTransform:
    """Normalize image values to the reflectivity range [0.4, 1.0]."""

    def __call__(self, tensor):
        max_val = tensor.max()
        min_val = tensor.min()
        value_range = max_val - min_val
        if value_range == 0:
            return torch.full_like(tensor, min_grey_value)
        return (tensor - min_val) / value_range * (1 - min_grey_value) + min_grey_value


class AugmentedMNIST(Dataset):
    """Repeat MNIST images and scale them for the specified distances."""

    def __init__(self, base_dataset, xsize, ysize, label_offsets=None):
        self.base_dataset = base_dataset
        self.xsize = xsize
        self.ysize = ysize
        self.label_offsets = label_offsets or [3.6] * 5

    def __len__(self):
        return len(self.base_dataset) * len(self.label_offsets)

    def __getitem__(self, idx):
        original_idx = idx // len(self.label_offsets)
        offset_idx = idx % len(self.label_offsets)
        distance = self.label_offsets[offset_idx]
        scale_version = (distance - 1) / 2
        image, label = self.base_dataset[original_idx]

        if scale_version != 0:
            pad_size = int(scale_version * self.xsize)
            image = transforms.functional.pad(
                image,
                padding=(pad_size, pad_size, pad_size, pad_size),
                fill=min_grey_value,
            )
            image = transforms.functional.resize(
                image,
                (self.ysize, self.xsize),
                interpolation=transforms.InterpolationMode.NEAREST,
            )

        return image, label * 10 + distance


transform = transforms.Compose(
    [
        transforms.Resize((ysize, xsize)),
        transforms.ToTensor(),
        CustomTransform(),
    ]
)

script_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.environ.get("FDASNN_DATA_ROOT", os.path.join(script_dir, "datasets"))
data_root = os.path.join(dataset_dir, "mnist")


if __name__ == "__main__":
    set_seed(48)

    test_dataset = torchvision.datasets.MNIST(
        root=data_root,
        train=True,
        transform=transform,
        download=True,
    )

    tau_opt = parameter.tau_opt
    rho_constant = parameter.rho_constant
    FF = parameter.FF
    bandwidth = parameter.bandwidth
    d_lens = parameter.d_lens
    P_BG = parameter.P_BG
    background = P_BG
    A_pix = parameter.A_pix
    F_fac = parameter.F_fac
    bin_number = 128
    Tstep = 1e-9
    Tobs = 128e-9
    Hold_off = parameter.Hold_off
    HO = int(Hold_off / Tstep)
    PDE = parameter.PDE
    Ep = parameter.Ep
    R_DCR = parameter.R_DCR
    N_pixel = parameter.N_pixel
    c = parameter.c
    sita = parameter.sita
    offset = parameter.offset

    mu = parameter.mu
    sigma = parameter.sigma
    Txstep = parameter.Txstep
    TxT = round(Tstep / Txstep)

    t = np.arange(0, Tobs + Txstep, Txstep)
    Gaussian = norm.pdf(t, mu, sigma)
    Gaussian_norm = (Gaussian - np.min(Gaussian)) / (
        np.max(Gaussian) - np.min(Gaussian)
    )
    num_full_groups = len(Gaussian_norm) // TxT
    reshaped_Gaussian = Gaussian_norm[: num_full_groups * TxT].reshape(
        num_full_groups, TxT
    )

    save_dir = os.environ.get("FDASNN_TOF_OUTPUT", os.path.join(script_dir, "results"))
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, "ToF_result.csv")

    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "Distance",
                "P_tx",
                "Mean distance",
                "Distance variance",
                "Accuracy",
            ]
        )

    distance_range = np.arange(3.6, 5.0, 0.8)
    for dist in tqdm(distance_range, desc="Distance sweep", unit="distance"):
        dist = round(float(dist), 1)

        if dist <= 2.0:
            P_tx = 0.0001
        elif dist < 7.0:
            P_tx = 0.001
        else:
            P_tx = 0.005

        summed_signal = reshaped_Gaussian.sum(axis=1) * P_tx

        set_seed(48)
        augmented_test = AugmentedMNIST(
            test_dataset, xsize, ysize, label_offsets=[dist] * 5
        )
        test_loader = DataLoader(augmented_test, batch_size=batch_size, shuffle=False)
        imgs, labels = next(iter(test_loader))
        Z = labels[0] % 10

        P_circ = summed_signal / (
            math.pi * (4 * Z**2 + d_lens**2) * math.tan(sita) ** 2
        )
        AC_imgs = parameter.set_0_4_to_zero(imgs.clone())
        P_circ_expanded = P_circ[:, np.newaxis, np.newaxis]
        P_receive = AC_imgs[np.newaxis, :, :] * P_circ_expanded
        P_receive = P_receive.transpose(0, 2)

        y = (tau_opt * FF * A_pix * P_receive) / (F_fac**2)

        shift_amount = math.ceil((2 * Z.item() / c) / Tstep) + offset
        P_pix_s = np.roll(y, shift_amount, axis=0)

        P_pix_bg = (
            tau_opt * imgs * rho_constant * FF * background * bandwidth * A_pix
        ) / (F_fac**2 * N_pixel * 4)
        N_BG = P_pix_bg / Ep
        N_BG = N_BG.unsqueeze(0).repeat(bin_number, 1, 1, 1, 1)
        N_SIG = torch.from_numpy(P_pix_s / Ep)
        lambda_spad = (N_SIG + N_BG) * PDE

        P_DET = 1 - np.exp(-(lambda_spad + R_DCR) * Tstep)

        result = P_DET.clone()
        for T in range(1, result.size(0)):
            T_min = max(0, T - HO)
            history = result[T_min:T]
            sum_history = history.sum(dim=0)
            result[T] = (1 - sum_history) * P_DET[T]

        pe = encoding.PoissonEncoder()
        out_spike = pe(result)
        out_spike1 = pe(result)
        out_spike2 = pe(result)
        out_spike3 = pe(result)
        out_spike4 = pe(result)
        combined_spikes = out_spike + out_spike1 + out_spike2 + out_spike3 + out_spike4

        histogram_sums = combined_spikes.sum(dim=(2, 3, 4))
        max_positions = np.argmax(histogram_sums, axis=0)
        total_offset = mu / Tstep + offset
        estimated_distance = (max_positions - total_offset + 1) * (Tstep * c / 2)

        mean_distance = estimated_distance.mean()
        variance_distance = estimated_distance.var()
        accuracy = (torch.abs(estimated_distance - Z) <= 0.15 * 5).float().mean() * 100

        print(
            f"Mean: {mean_distance.item():.4f} | "
            f"Variance: {variance_distance.item():.4f} | "
            f"Accuracy: {accuracy.item():.2f}%"
        )

        with open(csv_path, "a", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(
                [
                    f"{Z.item():.1f}",
                    f"{P_tx * 1000:.1f}",
                    f"{mean_distance.item():.4f}",
                    f"{variance_distance.item():.4f}",
                    f"{accuracy.item():.4f}",
                ]
            )

    print(f"All distance statistics saved to: {csv_path}")
