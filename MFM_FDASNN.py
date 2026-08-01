"""Train and evaluate FDASNN with multi-focus mapping (MFM)."""

import csv
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import functional
from torchvision.transforms import functional as TF
from tqdm import tqdm
import parameter
from dataloader import AugMNIST, center_crop_if_oversized
from module import set_seed
from net import FDASNN
from parameter import spiking


EPOCHS = 50
LEARNING_RATE = 0.001
BATCH_SIZE = 32
CHANNELS = 8
IMAGE_SIZE = 240
SEED = 130


def apply_mfm(spike_sequence, distances):
    """Crop and restore each spike sequence according to its target distance."""
    # MFM removes distance-dependent padding, then restores a fixed input size.
    distances = torch.floor(distances)
    crop_sizes = IMAGE_SIZE * (distances - 1) // (2 * distances)
    mapped_samples = []

    for sample_index, distance in enumerate(distances):
        crop_size = int(crop_sizes[sample_index].item())
        cropped = TF.crop(
            spike_sequence[:, sample_index],
            top=crop_size,
            left=crop_size,
            height=IMAGE_SIZE - 2 * crop_size,
            width=IMAGE_SIZE - 2 * crop_size,
        )
        restored = F.interpolate(
            cropped,
            scale_factor=float(distance),
            mode="nearest",
        )
        restored = center_crop_if_oversized(restored, IMAGE_SIZE, IMAGE_SIZE)
        mapped_samples.append(restored)

    return torch.stack(mapped_samples).transpose(0, 1).float()


def run_epoch(model, data_loader, device, optimizer=None):
    """Run one training or evaluation epoch and return aggregate metrics."""
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0.0
    total_samples = 0
    start_time = time.time()

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, labels in tqdm(data_loader):
            images = images.to(device, non_blocking=True)
            # Each augmented label stores class * 10 + simulated distance.
            class_labels = (labels // 10).long().to(device, non_blocking=True)
            distances = (labels % 10).to(device, non_blocking=True)
            class_targets = F.one_hot(class_labels, 10).float()

            if training:
                optimizer.zero_grad()

            spike_sequence = spiking(images, distances)
            mapped_spikes = apply_mfm(spike_sequence, distances)
            # The network returns one prediction per time step.
            logits = model(mapped_spikes).mean(0)
            loss = F.cross_entropy(logits, class_targets)

            if training:
                loss.backward()
                optimizer.step()

            sample_count = labels.numel()
            total_samples += sample_count
            total_loss += loss.item() * sample_count
            total_correct += (logits.argmax(1) == class_labels).sum().item()
            functional.reset_net(model)

    elapsed = time.time() - start_time
    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
        "speed": total_samples / elapsed,
    }


def append_metrics(filename, epoch, train_metrics, test_metrics, max_test_accuracy):
    """Append one epoch of metrics to a CSV file."""
    fieldnames = [
        "epoch",
        "train_loss",
        "train_accuracy",
        "test_loss",
        "test_accuracy",
        "max_test_accuracy",
    ]
    row = {
        "epoch": epoch,
        "train_loss": train_metrics["loss"],
        "train_accuracy": train_metrics["accuracy"],
        "test_loss": test_metrics["loss"],
        "test_accuracy": test_metrics["accuracy"],
        "max_test_accuracy": max_test_accuracy,
    }

    with open(filename, mode="a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if csv_file.tell() == 0:
            writer.writeheader()
        writer.writerow(row)


def main():
    """Train FDASNN and save the checkpoint with the best test accuracy."""
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    set_seed(SEED)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = FDASNN(T=parameter.bin_number, channels=CHANNELS)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, dim=1)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train_loader, test_loader = AugMNIST(IMAGE_SIZE, IMAGE_SIZE, BATCH_SIZE)

    output_dir = os.path.dirname(__file__)
    max_test_accuracy = -1.0
    for epoch in range(1, EPOCHS + 1):
        train_metrics = run_epoch(model, train_loader, device, optimizer)
        test_metrics = run_epoch(model, test_loader, device)

        # Keep only the checkpoint with the best validation accuracy.
        if test_metrics["accuracy"] > max_test_accuracy:
            max_test_accuracy = test_metrics["accuracy"]
            state_dict = (
                model.module.state_dict()
                if isinstance(model, nn.DataParallel)
                else model.state_dict()
            )
            torch.save(state_dict, os.path.join(output_dir, "FDASNN.pth"))

        print(
            f"epoch={epoch}, train_loss={train_metrics['loss']:.4f}, "
            f"train_accuracy={train_metrics['accuracy']:.4f}, "
            f"test_loss={test_metrics['loss']:.4f}, "
            f"test_accuracy={test_metrics['accuracy']:.4f}, "
            f"max_test_accuracy={max_test_accuracy:.4f}"
        )
        print(
            f"train_speed={train_metrics['speed']:.2f} images/s, "
            f"test_speed={test_metrics['speed']:.2f} images/s"
        )
        append_metrics(
            os.path.join(output_dir, "result.csv"),
            epoch,
            train_metrics,
            test_metrics,
            max_test_accuracy,
        )


if __name__ == "__main__":
    main()
