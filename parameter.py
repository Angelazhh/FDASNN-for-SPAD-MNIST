import math

import numpy as np
import torch
from scipy.stats import norm
from spikingjelly.activation_based import encoding

"""Optical and SPAD simulation parameters."""

tau_opt = 0.66  # Optical transmittance.
rho_constant = 1  # Background reflection coefficient.

# Optical system and detector parameters.
FF = 0.39  # SPAD fill factor.
bandwidth = 10  # Optical filter bandwidth in nanometers.
f_lens = 6e-3  # Lens focal length in meters.
d_lens = 5e-3  # Lens aperture diameter in meters.
P_BG = 0.29  # Background spectral power in watts per nanometer.
A_pix = 100e-12  # Active pixel area in square meters.
F_fac = f_lens / d_lens  # Lens f-number.
PDP = 0.0526  # Photon detection probability.
bin_number = 10  # Number of temporal bins in one simulated sequence.
Tstep = 5e-9  # Duration of one temporal bin in seconds.
Tobs = Tstep * bin_number  # Total observation time in seconds.
PDE = PDP * FF  # Effective photon detection efficiency.
h = 6.62607015e-34  # Planck constant in joule-seconds.
c = 3e8  # Speed of light in meters per second.
lambda_e = 905e-9  # Emitted laser wavelength in meters.
Ep = h * c / lambda_e  # Energy of one emitted photon in joules.
R_DCR = 1000  # SPAD dark count rate in counts per second.
N_pixel = 1  # Pixel-count normalization factor.
background = P_BG  # Background power used by the simulator.

P_tx = 0.001  # Peak transmitted laser power in watts.

# Gaussian laser-pulse parameters.
theta_H = 0.03  # Horizontal beam-divergence angle in degrees.
theta_V = 0.03  # Vertical beam-divergence angle in degrees.
sita = (theta_H + theta_V) / 2  # Mean beam-divergence angle in degrees.
mu = 18e-9  # Temporal center of the laser pulse in seconds.
sigma = 2.9e-9  # Standard deviation of the laser pulse in seconds.
Txstep = 0.1e-9  # Sampling interval of the laser waveform in seconds.
sita = sita / 180 * math.pi  # Mean beam-divergence angle in radians.
TxT = 50  # Laser samples accumulated into each detector bin.
min_grey_value = 0.4  # Synthetic background reflectivity.


def SM_signal():
    """Generate the binned Gaussian laser pulse."""

    t = np.arange(0, Tobs + Txstep, Txstep)

    Gaussian = norm.pdf(t, mu, sigma)

    signal = (
        (Gaussian - np.min(Gaussian)) / (np.max(Gaussian) - np.min(Gaussian)) * P_tx
    )

    num_full_groups = len(signal) // TxT

    reshaped_signal = signal[: num_full_groups * TxT].reshape(num_full_groups, TxT)
    summed_signal = reshaped_signal.sum(axis=1)
    return summed_signal


def set_0_4_to_zero(img):
    """Remove the synthetic background reflectivity in place."""
    mask = img == min_grey_value
    img[mask] = 0.0
    return img


def spiking(img, label):
    """Simulate two independent SPAD spike sequences on the input device."""
    # Distance labels are broadcast over time and spatial dimensions.
    Z = label.unsqueeze(0)
    signal = SM_signal()
    signal = torch.from_numpy(signal).float().unsqueeze(1)
    signal = signal.to(img.device)
    P_circ = signal / (math.pi * (4 * Z**2 + d_lens**2) * math.tan(sita) ** 2)

    P_circ_expanded = P_circ.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(img.device)
    AC_img = img.clone()
    AC_img = set_0_4_to_zero(AC_img)
    AC_img = AC_img.unsqueeze(0)
    P_receive = AC_img * P_circ_expanded

    y = (tau_opt * FF * A_pix * P_receive) / (F_fac**2)

    # Convert received signal and background power to photon arrival rates.
    N_SIG = y / Ep

    P_pix_bg = (tau_opt * img * rho_constant * FF * background * bandwidth * A_pix) / (
        F_fac**2 * N_pixel * 4
    )

    N_BG = P_pix_bg / Ep
    lambda_spad = (N_SIG + N_BG) * PDE

    P_DET = 1 - torch.exp(-(lambda_spad + R_DCR) * Tstep)

    result = P_DET.clone()
    # Approximate detector dead time by suppressing consecutive-bin detections.
    for t in range(1, result.size(0)):

        result[t] = (1 - result[t - 1]) * P_DET[t]

    pe = encoding.PoissonEncoder()

    # Two independent draws double the simulated observation interval.
    out_spike = pe(result)
    out_spike1 = pe(result)

    return torch.cat([out_spike, out_spike1], dim=0)


def spikingCPU(img, label):
    """CPU variant of :func:`spiking`."""
    return spiking(img.cpu(), label.cpu())
