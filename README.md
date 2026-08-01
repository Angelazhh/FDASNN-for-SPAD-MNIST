# FDASNN for SPAD-MNIST

This repository contains the training, evaluation, firing-rate analysis, and
post-training quantization code for the Feature-Decoupled Attention Spiking
Neural Network (FDASNN). The input images are converted into SPAD event
sequences with a distance-dependent optical model before being processed by
the multi-step spiking network.

## Requirements

Python 3.10 or 3.11 is recommended. A CUDA-capable GPU is strongly recommended
for training and quantized evaluation.

The project requires the following third-party packages:

| Package | Purpose | Version used during development |
| --- | --- | --- |
| PyTorch | Neural-network training and tensor operations | 2.5.1 |
| TorchVision | MNIST datasets and image transformations | 0.20.1 |
| SpikingJelly | Spiking neurons, monitors, and functional utilities | 0.0.0.0.14 |
| NumPy | Numerical processing | 2.3.5 |
| SciPy | Gaussian pulse generation, MAT loading, and statistics | 1.16.3 |
| scikit-learn | Label encoding | 1.8.0 |
| Pillow | Image loading | 10.4.0 |
| tqdm | Progress bars | 4.66.5 |

Standard-library modules such as `os`, `math`, `time`, and `csv` do not need to
be installed separately.

## Installation

Create and activate an isolated environment first:

```bash
conda create -n fdasnn python=3.11 -y
conda activate fdasnn
```

Install PyTorch separately so that it matches the CUDA version installed on
the machine. For CUDA 12.1:

```bash
pip install torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu121
```

For CPU-only execution:

```bash
pip install torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cpu
```

Install the remaining dependencies:

```bash
pip install \
  spikingjelly==0.0.0.0.14 \
  numpy==2.3.5 \
  scipy==1.16.3 \
  scikit-learn==1.8.0 \
  Pillow==10.4.0 \
  tqdm==4.66.5
```

If a different CUDA version is required, select the corresponding PyTorch
installation command from the official PyTorch installation guide.

## Project Structure

```text
FDASNN/
├── MFM_FDASNN.py           # FDASNN training with multi-focus mapping
├── Test.py                 # Accuracy and class-wise metric evaluation
├── QFDASNN.py              # Post-training quantization and integer inference
├── Aspiking_rate_MNIST.py  # Layer-wise firing-rate analysis
├── APSHCM_ToF.py           # SPAD time-of-flight mean-std and accuracy experiment
├── net.py                  # FDASNN and attention-based SNN architectures
├── module.py               # Quantized layers and quantization parameters
├── parameter.py            # Optical and SPAD simulation model
├── dataloader.py           # MNIST and SPAD dataset utilities
├── function.py             # Fake-quantization and interpolation helpers
└── FDASNN.pth              # Trained checkpoint used by evaluation scripts
```

## Dataset

MNIST is downloaded automatically by TorchVision. By default, datasets are
stored in the `datasets` directory next to the source files.

To use a different data directory, set `FDASNN_DATA_ROOT` before running a
script:

```bash
export FDASNN_DATA_ROOT=/path/to/datasets
```

On Windows PowerShell:

```powershell
$env:FDASNN_DATA_ROOT = "D:\datasets"
```

The augmented label packs the class and simulated distance into one value:

```text
encoded_label = class_id * 10 + distance
```

The default simulated distances are `1.8`, `3.6`, `4.4`, `6.2`, and `8.0`.
Fractional distances are retained during SPAD simulation. Integer distances
are used only by the MFM crop-and-resize operation.

## Usage

Run all commands from the FDASNN directory.

### Train the model

```bash
python MFM_FDASNN.py
```

The best checkpoint is saved as `FDASNN.pth`, and epoch metrics are appended
to `result.csv`.

### Evaluate the trained model

```bash
python Test.py
```

The script reports overall accuracy, loss, macro precision, macro recall,
macro F1 score, and class-wise metrics.

### Run post-training quantization

```bash
python QFDASNN.py
```

The current configuration performs 4-bit calibration, freezes the quantized
weights, and evaluates integer inference with residual-scale balancing.

### Measure firing rates

```bash
python Aspiking_rate_MNIST.py
```

This script loads `FDASNN.pth` and reports input, layer-wise, and network-wide
average firing rates.

### Evaluate time-of-flight estimation

```bash
python APSHCM_ToF.py
```

By default, the script evaluates 100 samples at from `1.0` to `8.0` metres
and writes `results/ToF_result.csv`. Set `FDASNN_DATA_ROOT` to change the dataset
directory and `FDASNN_TOF_OUTPUT` to change the output directory:

```bash
FDASNN_DATA_ROOT=/path/to/datasets \
FDASNN_TOF_OUTPUT=/path/to/results \
python APSHCM_ToF.py
```

## Checkpoint and GPU Notes

- `FDASNN.pth` must exist before running `Test.py`, `QFDASNN.py`, or
  `Aspiking_rate_MNIST.py`.
- The scripts use `cuda:0` as the primary device when CUDA is available.
- Multi-GPU execution uses the batch dimension of the multi-step tensor
  (`dim=1`). Adjust the configured device IDs if fewer than four GPUs are
  available.
- CPU execution is supported for basic testing but is significantly slower.
