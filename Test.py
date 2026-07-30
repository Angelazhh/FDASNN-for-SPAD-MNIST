"""Evaluate trained FDASNN models on distance-augmented MNIST."""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import functional
from torchvision import transforms
from tqdm import tqdm

import parameter
from dataloader import AugMNIST, center_crop_if_oversized
from module import set_seed
from net import FDASNN
from parameter import spiking


SEEDS = [130]
BATCH_SIZE = 32
CHANNELS = 8
IMAGE_SIZE = 240
NUM_CLASSES = 10


def prepare_spikes(images, labels, device):
    """Generate SPAD spikes and apply the original MFM preprocessing."""
    images = images.to(device)
    class_labels = (labels // 10).long().to(device, non_blocking=True)
    distances = (labels % 10).to(device, non_blocking=True)
    class_targets = F.one_hot(class_labels, NUM_CLASSES).float()

    # Preserve fractional distances in the optical simulation.
    spike_sequence = spiking(images, distances).float()

    # MFM uses integer distances only for spatial crop-and-resize operations.
    crop_distances = torch.floor(distances)
    crop_sizes = IMAGE_SIZE * (crop_distances - 1) // (2 * crop_distances)
    mapped_samples = []
    for sample_index, distance in enumerate(crop_distances):
        crop_size = int(crop_sizes[sample_index].item())
        cropped = transforms.functional.crop(
            spike_sequence[:, sample_index],
            top=crop_size,
            left=crop_size,
            height=IMAGE_SIZE - 2 * crop_size,
            width=IMAGE_SIZE - 2 * crop_size,
        )
        upsampled = F.interpolate(cropped, scale_factor=float(distance), mode="nearest")
        upsampled = center_crop_if_oversized(
            upsampled, target_height=IMAGE_SIZE, target_width=IMAGE_SIZE
        )
        mapped_samples.append(upsampled)

    mapped_spikes = torch.stack(mapped_samples).transpose(0, 1).float()
    return spike_sequence, mapped_spikes, class_labels, class_targets


def evaluate(model, data_loader, device, use_mfm=True):
    """Evaluate one model and return aggregate and class-wise metrics."""
    model.eval()
    test_loss = 0.0
    test_correct = 0.0
    test_samples = 0
    confusion_matrix = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.int64)

    with torch.no_grad():
        for images, labels in tqdm(data_loader):
            spike_sequence, mapped_spikes, class_labels, class_targets = prepare_spikes(
                images, labels, device
            )
            model_input = mapped_spikes if use_mfm else spike_sequence
            logits = model(model_input).mean(0)
            loss = F.cross_entropy(logits, class_targets)
            predictions = logits.argmax(1)

            sample_count = labels.numel()
            test_samples += sample_count
            test_loss += loss.item() * sample_count
            test_correct += (predictions == class_labels).sum().item()

            # Accumulate the confusion matrix on CPU to avoid scalar GPU transfers.
            flat_indices = (
                class_labels.detach().cpu() * NUM_CLASSES + predictions.detach().cpu()
            )
            confusion_matrix += torch.bincount(
                flat_indices, minlength=NUM_CLASSES**2
            ).reshape(NUM_CLASSES, NUM_CLASSES)
            functional.reset_net(model)

    true_positive = confusion_matrix.diag()
    false_negative = confusion_matrix.sum(1) - true_positive
    false_positive = confusion_matrix.sum(0) - true_positive
    recall = true_positive / (true_positive + false_negative + 1e-9)
    precision = true_positive / (true_positive + false_positive + 1e-9)
    f1_score = 2 * precision * recall / (precision + recall + 1e-9)

    return {
        "test_acc": test_correct / test_samples,
        "test_loss": test_loss / test_samples,
        "macro_precision": precision.mean().item(),
        "macro_recall": recall.mean().item(),
        "macro_f1": f1_score.mean().item(),
        "per_class_recall": recall.numpy(),
        "per_class_precision": precision.numpy(),
        "per_class_f1": f1_score.numpy(),
        "support": confusion_matrix.sum(1).numpy(),
    }


def print_metrics(metrics):
    """Print aggregate and class-wise evaluation metrics."""
    print(
        f"test_acc={metrics['test_acc']:.4f} | " f"test_loss={metrics['test_loss']:.4f}"
    )
    print(
        f"Macro Precision={metrics['macro_precision']:.4f} | "
        f"Macro Recall={metrics['macro_recall']:.4f} | "
        f"Macro F1={metrics['macro_f1']:.4f}"
    )

    print("\nClass-wise metrics:")
    for class_index in range(NUM_CLASSES):
        print(
            f"Class {class_index}: "
            f"Recall={metrics['per_class_recall'][class_index]:.4f}, "
            f"Precision={metrics['per_class_precision'][class_index]:.4f}, "
            f"F1={metrics['per_class_f1'][class_index]:.4f}, "
            f"Support={metrics['support'][class_index]}"
        )
    print("-" * 80)


def main():
    """Load each checkpoint and evaluate it for every configured seed."""
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    models = {
        "FDASNN": FDASNN(T=parameter.bin_number, channels=CHANNELS),
    }
    model_paths = {
        "FDASNN": os.path.join(os.path.dirname(__file__), "FDASNN.pth"),
    }

    for model_name, model in models.items():
        print("#" * 60)
        print(f"Evaluating model: {model_name}")
        print("#" * 60)

        model.load_state_dict(torch.load(model_paths[model_name], map_location=device))
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model, dim=1, device_ids=[0, 1, 2, 3])
        model = model.to(device)

        all_results = {}
        for seed in SEEDS:
            set_seed(seed)
            _, test_loader = AugMNIST(IMAGE_SIZE, IMAGE_SIZE, BATCH_SIZE)

            # Attention-only baselines consume the unscaled spike sequence.
            use_mfm = model_name not in {"STCJA", "SCTFA"}
            metrics = evaluate(model, test_loader, device, use_mfm=use_mfm)
            print_metrics(metrics)
            all_results[seed] = metrics

    print("\nAll model evaluations completed.")


if __name__ == "__main__":
    main()
