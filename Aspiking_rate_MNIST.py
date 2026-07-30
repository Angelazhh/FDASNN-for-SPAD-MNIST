"""Measure input and layer-wise firing rates for a trained FDASNN model."""

import os

import torch
import torch.nn as nn
from spikingjelly.activation_based import functional, monitor, neuron

import parameter
from dataloader import AugMNIST
from module import set_seed
from net import FDASNN
from parameter import spiking


SEED = 130
BATCH_SIZE = 1
CHANNELS = 8
IMAGE_SIZE = 240
MAX_BATCHES = 10_000


def _accumulate_layer_stats(layer_stats, layer_names, records):
    """Accumulate spike counts for monitored layers."""
    for layer_name, spikes in zip(layer_names, records):
        if not isinstance(spikes, torch.Tensor):
            continue

        # A neuron is identified by its non-temporal, non-batch coordinates.
        neurons = spikes[0, 0].numel() if spikes.ndim >= 3 else spikes.shape[-1]
        stats = layer_stats.setdefault(
            layer_name,
            {"neurons": neurons, "spikes": 0, "active_neurons": 0},
        )
        stats["spikes"] += spikes.sum().item()
        stats["active_neurons"] += (spikes.sum(dim=0) > 0).sum().item()


def main():
    """Load the checkpoint and report average firing statistics."""
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    set_seed(SEED)

    time_steps = parameter.bin_number * 2
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = FDASNN(T=parameter.bin_number, channels=CHANNELS)
    checkpoint = os.path.join(os.path.dirname(__file__), "FDASNN.pth")
    model.load_state_dict(torch.load(checkpoint, map_location=device))

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, dim=1)
    model = model.to(device)
    model.eval()

    _, test_loader = AugMNIST(IMAGE_SIZE, IMAGE_SIZE, BATCH_SIZE)
    layer_stats = {}
    total_input_spikes = 0
    total_input_values = 0
    processed_batches = 0
    # Hooks collect every IFNode output without changing the forward path.
    output_monitor = monitor.OutputMonitor(model, neuron.IFNode)

    try:
        with torch.no_grad():
            for image, label in test_loader:
                image = image.to(device)
                distance = (label % 10).to(device, non_blocking=True)
                input_spikes = spiking(image, distance).float()

                total_input_spikes += input_spikes.sum().item()
                total_input_values += input_spikes.numel()
                model(input_spikes)

                _accumulate_layer_stats(
                    layer_stats,
                    output_monitor.monitored_layers,
                    output_monitor.records,
                )
                processed_batches += 1

                functional.reset_net(model)
                output_monitor.clear_recorded_data()

                if processed_batches % 1000 == 0:
                    print(
                        f"Processed {processed_batches}/{MAX_BATCHES} batches...",
                        end="\r",
                    )
                if processed_batches >= MAX_BATCHES:
                    break
    finally:
        output_monitor.remove_hooks()

    if processed_batches == 0:
        raise RuntimeError("The test data loader produced no batches.")

    print("\nLayer firing statistics")
    print("=" * 80)
    print(f"Averages over {processed_batches} batches")
    print("=" * 80)
    print("Input:")
    print(f"  Average spike count: {total_input_spikes / processed_batches:.2f}")
    print(f"  Average firing rate: {total_input_spikes / total_input_values:.6f}")
    print("-" * 60)

    total_neurons = 0
    total_average_spikes = 0
    for layer_name, stats in layer_stats.items():
        average_spikes = stats["spikes"] / processed_batches
        average_active = stats["active_neurons"] / processed_batches
        # Normalize spike counts by neuron count and simulated time steps.
        firing_rate = average_spikes / (stats["neurons"] * time_steps)
        total_neurons += stats["neurons"]
        total_average_spikes += average_spikes

        print(f"{layer_name}:")
        print(f"  Neurons: {stats['neurons']:,}")
        print(f"  Average spike count: {average_spikes:.2f}")
        print(f"  Average active neurons: {average_active:.2f}")
        print(f"  Average firing rate: {firing_rate:.6f}")
        print("-" * 60)

    network_rate = total_average_spikes / (total_neurons * time_steps)
    print(f"Network average firing rate: {network_rate:.6f}")


if __name__ == "__main__":
    main()
