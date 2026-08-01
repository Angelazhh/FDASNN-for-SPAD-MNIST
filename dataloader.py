"""Dataset builders and distance-aware MNIST transformations."""

import os
import numpy as np
import scipy.io as sio
import torch
import torch.nn.functional as F
import torch.utils.data as data
import torchvision
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from torchvision import transforms
import parameter
from parameter import spiking


min_grey_value = parameter.min_grey_value
DATA_ROOT = os.environ.get(
    "FDASNN_DATA_ROOT", os.path.join(os.path.dirname(__file__), "datasets")
)


def center_crop_if_oversized(tensor, target_height, target_width):
    """Center-crop spatial dimensions only when they exceed the target size."""
    current_height, current_width = tensor.shape[-2:]
    crop_height = min(current_height, target_height)
    crop_width = min(current_width, target_width)
    top = max((current_height - target_height) // 2, 0)
    left = max((current_width - target_width) // 2, 0)
    return tensor[..., top : top + crop_height, left : left + crop_width]


class CustomTransform:
    """Normalize reflectivity to ``[min_grey_value, 1]``."""

    def __call__(self, tensor):

        min_val = tensor.min()
        max_val = tensor.max()

        rho_target = (tensor - min_val) / (max_val - min_val) * (
            1 - min_grey_value
        ) + min_grey_value

        return rho_target


class AugmentedMNIST(Dataset):
    """Repeat MNIST samples at a configurable set of simulated distances."""

    def __init__(self, base_dataset, label_offsets=None):
        self.base_dataset = base_dataset

        if label_offsets is not None:
            self.label_offsets = label_offsets
        else:
            self.label_offsets = [1.8, 3.6, 4.4, 6.2, 8.0]

    def __len__(self):
        return len(self.base_dataset) * 5

    def __getitem__(self, idx):
        # Consecutive groups of five represent one image at five distances.
        original_idx = idx // 5
        i = idx % 5
        version = (self.label_offsets[i] - 1) / 2

        img, label = self.base_dataset[original_idx]
        xsize = img.shape[2]
        ysize = img.shape[1]
        if version != 0:

            pad_size = int(version * xsize)
            img = transforms.functional.pad(
                img,
                padding=(pad_size, pad_size, pad_size, pad_size),
                fill=min_grey_value,
            )
            img = transforms.functional.resize(
                img, (ysize, xsize), interpolation=transforms.InterpolationMode.NEAREST
            )
            img = center_crop_if_oversized(img, ysize, xsize)

        # Pack class and distance into one scalar for the existing loaders.
        new_label = label * 10 + self.label_offsets[i]

        return img, new_label


class AugmentedSingleMNIST(Dataset):
    """Apply one simulated distance and generate spikes per sample."""

    def __init__(self, base_dataset, label_offsets=None):
        self.base_dataset = base_dataset

        if label_offsets is not None:
            self.label_offsets = label_offsets
        else:
            self.label_offsets = [3.6]

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):

        original_idx = idx
        version = (self.label_offsets[0] - 1) / 2

        img, label = self.base_dataset[original_idx]
        xsize = img.shape[2]
        ysize = img.shape[1]
        if version != 0:

            pad_size = int(version * xsize)
            img = transforms.functional.pad(
                img,
                padding=(pad_size, pad_size, pad_size, pad_size),
                fill=min_grey_value,
            )
            img = transforms.functional.resize(
                img, (ysize, xsize), interpolation=transforms.InterpolationMode.NEAREST
            )
            img = center_crop_if_oversized(img, ysize, xsize)

        new_label = label * 10 + self.label_offsets[0]

        label2 = torch.tensor(new_label % 10, dtype=torch.float32)
        img_batch = img.unsqueeze(0)
        out_spike = spiking(img_batch, label2)
        out_spike = out_spike.squeeze(1)

        return out_spike, new_label


def MNIST(xsize, ysize, bachsize):
    # FDASNN_DATA_ROOT can redirect downloads outside the source tree.
    MNIST_dir = os.path.join(DATA_ROOT, "mnist")

    transform1 = transforms.Compose(
        [
            transforms.Resize((ysize, xsize)),
            transforms.ToTensor(),
            CustomTransform(),
        ]
    )

    train_dataset = torchvision.datasets.MNIST(
        root=MNIST_dir, train=True, transform=transform1, download=True
    )
    test_dataset = torchvision.datasets.MNIST(
        root=MNIST_dir, train=False, transform=transform1, download=True
    )

    train_data_loader = data.DataLoader(
        dataset=train_dataset,
        batch_size=bachsize,
        shuffle=True,
        drop_last=True,
        num_workers=8,
        pin_memory=True,
    )
    test_data_loader = data.DataLoader(
        dataset=test_dataset,
        batch_size=bachsize,
        shuffle=False,
        drop_last=True,
        num_workers=8,
        pin_memory=True,
    )
    return train_data_loader, test_data_loader


def AugMNIST(xsize, ysize, bachsize, distance=None):

    MNIST_dir = os.path.join(DATA_ROOT, "mnist")

    transform1 = transforms.Compose(
        [
            transforms.Resize((ysize, xsize)),
            transforms.ToTensor(),
            CustomTransform(),
        ]
    )

    train_dataset = torchvision.datasets.MNIST(
        root=MNIST_dir, train=True, transform=transform1, download=True
    )
    test_dataset = torchvision.datasets.MNIST(
        root=MNIST_dir, train=False, transform=transform1, download=True
    )

    augmented_train = AugmentedMNIST(train_dataset, distance)
    augmented_test = AugmentedMNIST(test_dataset, distance)

    train_data_loader = data.DataLoader(
        dataset=augmented_train,
        batch_size=bachsize,
        shuffle=True,
        drop_last=True,
        num_workers=8,
        pin_memory=True,
    )
    test_data_loader = data.DataLoader(
        dataset=augmented_test,
        batch_size=bachsize,
        shuffle=False,
        drop_last=True,
        num_workers=8,
        pin_memory=True,
    )
    return train_data_loader, test_data_loader


def AugSingleMNIST(xsize, ysize, bachsize, distance=None):

    MNIST_dir = os.path.join(DATA_ROOT, "mnist")

    transform1 = transforms.Compose(
        [
            transforms.Resize((ysize, xsize)),
            transforms.ToTensor(),
            CustomTransform(),
        ]
    )

    train_dataset = torchvision.datasets.MNIST(
        root=MNIST_dir, train=True, transform=transform1, download=True
    )
    test_dataset = torchvision.datasets.MNIST(
        root=MNIST_dir, train=False, transform=transform1, download=True
    )

    augmented_train = AugmentedSingleMNIST(train_dataset, distance)
    augmented_test = AugmentedSingleMNIST(test_dataset, distance)

    train_data_loader = data.DataLoader(
        dataset=augmented_train,
        batch_size=bachsize,
        shuffle=True,
        drop_last=True,
        num_workers=8,
        pin_memory=True,
    )
    test_data_loader = data.DataLoader(
        dataset=augmented_test,
        batch_size=bachsize,
        shuffle=False,
        drop_last=True,
        num_workers=8,
        pin_memory=True,
    )
    return train_data_loader, test_data_loader


class CustomDataset(Dataset):

    def __init__(self, image_dir, label_file, transform=None):
        super().__init__()
        self.image_dir = image_dir
        self.label_file = label_file
        self.transform = transform
        self.samples = self._load_samples()

    def _load_samples(self):
        samples = []
        with open(self.label_file, "r") as f:
            for line in f:
                image_name, label = line.strip().split(",")
                image_path = os.path.join(self.image_dir, image_name)
                samples.append((image_path, int(label)))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("L")
        if self.transform:
            image = self.transform(image)
        return image, label


def AugCustomDataset(xsize, ysize, bachsize):

    pre_image_dir = os.path.join(DATA_ROOT, "mnist", "pre_images")
    pre_label_file = os.path.join(DATA_ROOT, "mnist", "pre_label.txt")

    transform = transforms.Compose(
        [
            transforms.Resize((ysize, xsize)),
            transforms.ToTensor(),
            CustomTransform(),
        ]
    )

    pre_dataset = CustomDataset(pre_image_dir, pre_label_file, transform=transform)

    augmented_test = AugmentedMNIST(pre_dataset)

    pre_data_loader = data.DataLoader(
        dataset=augmented_test,
        batch_size=bachsize,
        shuffle=False,
        drop_last=True,
        num_workers=8,
        pin_memory=True,
    )
    return pre_data_loader


def FashionMNIST(xsize, ysize, bachsize):

    MNIST_dir = os.path.join(DATA_ROOT, "fashion-mnist")

    transform1 = transforms.Compose(
        [
            transforms.Resize((ysize, xsize)),
            transforms.ToTensor(),
            CustomTransform(),
        ]
    )

    train_dataset = torchvision.datasets.FashionMNIST(
        root=MNIST_dir, train=True, transform=transform1, download=True
    )
    test_dataset = torchvision.datasets.FashionMNIST(
        root=MNIST_dir, train=False, transform=transform1, download=True
    )

    train_data_loader = data.DataLoader(
        dataset=train_dataset,
        batch_size=bachsize,
        shuffle=True,
        drop_last=True,
        num_workers=8,
        pin_memory=True,
    )
    test_data_loader = data.DataLoader(
        dataset=test_dataset,
        batch_size=bachsize,
        shuffle=False,
        drop_last=True,
        num_workers=8,
        pin_memory=True,
    )
    return train_data_loader, test_data_loader


def AugFashionMNIST(xsize, ysize, bachsize, distance=None):

    MNIST_dir = os.path.join(DATA_ROOT, "fashion-mnist")

    transform1 = transforms.Compose(
        [
            transforms.Resize((ysize, xsize)),
            transforms.ToTensor(),
            CustomTransform(),
        ]
    )

    train_dataset = torchvision.datasets.FashionMNIST(
        root=MNIST_dir, train=True, transform=transform1, download=True
    )
    test_dataset = torchvision.datasets.FashionMNIST(
        root=MNIST_dir, train=False, transform=transform1, download=True
    )

    augmented_train = AugmentedMNIST(train_dataset, distance)
    augmented_test = AugmentedMNIST(test_dataset, distance)

    train_data_loader = data.DataLoader(
        dataset=augmented_train,
        batch_size=bachsize,
        shuffle=True,
        drop_last=True,
        num_workers=8,
        pin_memory=True,
    )
    test_data_loader = data.DataLoader(
        dataset=augmented_test,
        batch_size=bachsize,
        shuffle=False,
        drop_last=True,
        num_workers=8,
        pin_memory=True,
    )
    return train_data_loader, test_data_loader


class CustomDataset_Real(Dataset):
    """In-memory SPAD samples with class and distance labels."""

    def __init__(self, imgs, label1, label2):
        self.imgs = imgs
        self.label1 = label1
        self.label2 = label2

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):

        return self.imgs[idx], self.label1[idx], self.label2[idx]


def load_all_mat_files(data_dir, num_pusle=1):
    """Load ``label_distance.mat`` files and return padded SPAD samples."""
    cropped_imgs = []
    label1_list = []
    label2_list = []

    for filename in os.listdir(data_dir):
        if filename.endswith(".mat"):

            filename_without_ext = filename[:-4]
            try:
                label_str, distance_str = filename_without_ext.split("_")
                label1 = torch.tensor(int(label_str))
                label2 = torch.tensor(float(distance_str))
            except ValueError:
                print(
                    f"Skipping malformed filename {filename!r}; expected "
                    "'label_distance.mat'."
                )
                continue

            mat_path = os.path.join(data_dir, filename)
            mat_data = sio.loadmat(mat_path)

            cropped_img_ori = mat_data["combined_data"]
            ori_T, ori_N, ori_H, ori_W = cropped_img_ori.shape

            # Split interleaved pulses, move pulse count before time, then merge them.
            cropped_img_reshaped = cropped_img_ori.reshape(
                ori_T, ori_N // num_pusle, num_pusle, ori_H, ori_W
            )

            cropped_img_transposed = cropped_img_reshaped.transpose(2, 0, 1, 3, 4)

            cropped_img = cropped_img_transposed.reshape(
                num_pusle * ori_T, ori_N // num_pusle, ori_H, ori_W
            )

            img = torch.from_numpy(cropped_img.transpose(1, 0, 2, 3))

            label2 = torch.floor(label2)

            N, T, H, W = img.shape

            for i in range(N):

                label1_list.append(label1)
                label2_list.append(label2)

            img = img.unsqueeze(2)

            for i in range(N):

                upsampled_imgs = F.interpolate(
                    img[i], scale_factor=float(label2), mode="nearest"
                )
                upsampled_imgs = center_crop_if_oversized(
                    upsampled_imgs, target_height=240, target_width=240
                )

                current_size = upsampled_imgs.shape[-1]
                # Center-pad scaled measurements to the network's fixed resolution.
                if upsampled_imgs.shape[-1] != 240:

                    pad_total = 240 - current_size
                    pad_left = pad_total // 2
                    pad_right = pad_total - pad_left
                    pad_top = pad_left
                    pad_bottom = pad_right

                    padded_imgs = F.pad(
                        upsampled_imgs,
                        pad=(
                            pad_left,
                            pad_right,
                            pad_top,
                            pad_bottom,
                        ),
                        mode="constant",
                        value=0,
                    )

                    cropped_imgs.append(padded_imgs)
                else:

                    cropped_imgs.append(upsampled_imgs)

    le = LabelEncoder()
    le.fit(label1_list)
    num_classes = len(le.classes_)

    imgs_np = np.array(cropped_imgs)
    label1_np = np.array(label1_list)
    label2_np = np.array(label2_list)

    print("Data loading complete.")
    print(f"Samples: {len(imgs_np)}")
    print(f"Classes: {num_classes} (labels: {le.classes_})")
    print(f"Sample shape: (T={T}, H={H}, W={W})")
    print(f"Distance range: {label2_np.min():.2f} to {label2_np.max():.2f}")

    return imgs_np, label1_np, label2_np, num_classes
