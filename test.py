import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

def process_tie_to_phase_noise(tie_vector, f0, nperseg=None):
    """
    Computes Time-Domain RMS Jitter, computes Phase Noise spectrum L(f),
    and reconstructs the RMS Jitter from the spectrum.
    
    Parameters:
    - tie_vector: 1D array-like of TIE values in seconds (sampled 1 per clock edge).
    - f0: Nominal clock frequency in Hz (e.g., 100e6 for 100 MHz).
    - nperseg: Segment length for Welch PSD estimate (default: len(tie_vector)//4).
    
    Returns:
    - freq: Offset frequency array (Hz)
    - L_f_dbc: Phase noise array (dBc/Hz)
    - rms_time: Time-domain RMS TIE jitter (s)
    - rms_freq: Frequency-domain recovered RMS TIE jitter (s)
    """
    tie = np.asarray(tie_vector, dtype=np.float64)
    N = len(tie)
    
    # 1. Remove DC offset
    tie = tie - np.mean(tie)
    
    # 2. Time-domain RMS Jitter
    rms_time = np.std(tie)
    
    # 3. Convert TIE to phase error in radians
    phi = 2.0 * np.pi * f0 * tie  # Sampling rate Fs = f0
    
    # 4. Compute One-Sided PSD S_phi(f) using Welch's method (rad^2 / Hz)
    # Using 'boxcar' window or Hann window with density scaling
    if nperseg is None:
        nperseg = min(N, 65536)
    
    freq, s_phi = signal.welch(
        phi,
        fs=f0,
        window='hann',
        nperseg=nperseg,
        scaling='density',
        detrend=False
    )
    
    # Remove DC component (f = 0)
    freq = freq[1:]
    s_phi = s_phi[1:]
    
    # 5. Calculate Single-Sideband Phase Noise L(f) in dBc/Hz
    # L(f) = 10 * log10(S_phi(f) / 2)
    L_f_dbc = 10.0 * np.log10(s_phi / 2.0)
    
    # 6. Reconstruct RMS Jitter from L(f)
    # Integrate: S_phi_recon = 2 * 10^(L(f)/10)
    s_phi_recon = 2.0 * 10.0 ** (L_f_dbc / 10.0)
    sigma_phi_sq = np.trapz(s_phi_recon, freq) # np.trapz for older numpy
    sigma_phi = np.sqrt(sigma_phi_sq)
    
    rms_freq = sigma_phi / (2.0 * np.pi * f0)
    
    # 7. Print Verification
    print("=" * 45)
    print(f"Clock Frequency (f0)        : {f0/1e6:.3f} MHz")
    print(f"Time-Domain RMS TIE Jitter  : {rms_time * 1e12:.4f} ps")
    print(f"Recovered RMS TIE Jitter    : {rms_freq * 1e12:.4f} ps")
    print(f"Discrepancy                 : {abs(rms_time - rms_freq)/rms_time * 100:.2f} %")
    print("=" * 45)
    
    return freq, L_f_dbc, rms_time, rms_freq

# # --- Demonstration with Synthetic Noisy Clock ---
# if __name__ == "__main__":
#     np.random.seed(42)
#     f0 = 100e6          # 100 MHz clock
#     num_samples = 200000
    
#     # Generate random white + 1/f noise for TIE
#     white_noise = np.random.normal(0, 2e-12, num_samples) # 2 ps RMS
#     t = np.arange(num_samples) / f0
#     tie_sample = white_noise
    
#     freq_axis, pn_dbc, tj_time, tj_freq = process_tie_to_phase_noise(tie_sample, f0)
    
#     # Plotting Phase Noise
#     plt.figure(figsize=(9, 5))
#     plt.semilogx(freq_axis, pn_dbc, color='navy', label=r'$\mathcal{L}(f)$ Phase Noise')
#     plt.title(f"Phase Noise Plot from TIE ($f_0 = {f0/1e6:.0f}$ MHz)")
#     plt.xlabel("Offset Frequency (Hz)")
#     plt.ylabel(r"$\mathcal{L}(f)$ [dBc/Hz]")
#     plt.grid(True, which="both", ls="--", alpha=0.6)
#     plt.legend()
#     plt.tight_layout()
#     plt.show()

import numpy as np
import matplotlib.pyplot as plt

def tie_to_phase_noise_fft(tie_vector, f0):
    """
    Computes direct FFT-based Phase Noise L(f) and reconstructs RMS Jitter
    with exact Parseval energy equivalence (no Welch averaging).
    
    Parameters:
    - tie_vector: 1D array-like of TIE values in seconds
    - f0: Nominal clock / carrier frequency in Hz (Fs = f0)
    
    Returns:
    - freq: Positive offset frequency array (Hz)
    - L_f_dbc: Phase noise array (dBc/Hz)
    - rms_time: Time-domain RMS TIE jitter (s)
    - rms_freq: Recovered frequency-domain RMS TIE jitter (s)
    """
    tie = np.asarray(tie_vector, dtype=np.float64)
    N = len(tie)
    
    # 1. Remove DC offset
    tie_centered = tie - np.mean(tie)
    
    # 2. Time-domain RMS Jitter (Sample variance / Standard deviation)
    # Using ddof=0 to match exact N-point DFT normalization
    rms_time = np.std(tie_centered, ddof=0)
    
    # 3. Convert TIE to Phase Error Sequence (rad)
    phi = 2.0 * np.pi * f0 * tie_centered
    
    # 4. Compute Real FFT (rfft automatically returns positive frequencies)
    phi_fft = np.fft.rfft(phi)
    freq = np.fft.rfftfreq(N, d=1.0/f0)
    
    # 5. Calculate One-Sided Power Spectral Density S_phi (rad^2 / Hz)
    # Energy scaling factor: 1 / (N * f0)
    psd_raw = (np.abs(phi_fft) ** 2) / (N * f0)
    
    # Account for single-sided folding (multiply positive AC bins by 2)
    s_phi = psd_raw.copy()
    if N % 2 == 0:
        s_phi[1:-1] *= 2.0  # Exclude DC (index 0) and Nyquist (index -1)
    else:
        s_phi[1:] *= 2.0    # Exclude DC only
        
    # Exclude DC bin (f = 0) from phase noise plot and integration
    freq = freq[1:]
    s_phi = s_phi[1:]
    
    # 6. Single-Sideband Phase Noise L(f) in dBc/Hz
    # L(f) = 10 * log10(S_phi(f) / 2)
    L_f_dbc = 10.0 * np.log10(s_phi / 2.0)
    
    # 7. Exact Jitter Reconstruction from L(f)
    df = f0 / N  # Frequency bin width (Hz)
    
    # Reconvert L(f) to linear density: S_phi = 2 * 10^(L/10)
    s_phi_recon = 2.0 * 10.0 ** (L_f_dbc / 10.0)
    
    # Integrate discrete sum: Var = Sum(S_phi * df)
    sigma_phi_sq = np.sum(s_phi_recon * df)
    sigma_phi = np.sqrt(sigma_phi_sq)
    
    rms_freq = sigma_phi / (2.0 * np.pi * f0)
    
    # Verification Printout
    print("=" * 48)
    print(f"Clock Frequency (f0)        : {f0/1e6:.3f} MHz")
    print(f"Total Points (N)            : {N}")
    print(f"Resolution Bandwidth (df)   : {df:.2f} Hz")
    print(f"Time-Domain RMS TIE         : {rms_time * 1e12:.6f} ps")
    print(f"Recovered Frequency RMS TIE : {rms_freq * 1e12:.6f} ps")
    print(f"Absolute Error              : {abs(rms_time - rms_freq) * 1e15:.4f} fs")
    print("=" * 48)
    
    return freq, L_f_dbc, rms_time, rms_freq


# --- Verification Example ---
if __name__ == "__main__":
    np.random.seed(42)
    f0 = 27e6         # 27 MHz clock
    N = 100000         # 100k clock edges
    
    # Synthesize random TIE jitter sequence (~1.5 ps RMS)
    tie_data = np.random.normal(0, 0.961e-12, N)
    
    freq_axis, pn_plot, t_rms, f_rms = tie_to_phase_noise_fft(tie_data, f0)
    
    # Plotting
    plt.figure(figsize=(9, 5))
    plt.semilogx(freq_axis, pn_plot, color='darkred', alpha=0.85, label=r'Direct FFT $\mathcal{L}(f)$')
    plt.title(f"Direct FFT Phase Noise from TIE ($f_0 = {f0/1e6:.0f}$ MHz)")
    plt.xlabel("Offset Frequency (Hz)")
    plt.ylabel(r"$\mathcal{L}(f)$ [dBc/Hz]")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()