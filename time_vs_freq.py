import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import os
from dotenv import load_dotenv

load_dotenv()

RISE_TIMES_TYP = os.getenv("PATH_MAX")

def get_rise_times(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    rise_times = [float(line.strip()) for line in lines]
    return rise_times

def get_periods(rise_times):
    periods = np.diff(rise_times)
    return periods

def get_frequency(periods):
    frequency = np.mean(1 / periods)
    return frequency

# def get_TIE(rise_times):
#     periods = get_periods(rise_times)
#     frequency = get_frequency(periods)
#     time = np.arange(len(rise_times)) / frequency
#     ideal_rise_times = rise_times[0] + time
#     TIE = np.cumsum(rise_times - ideal_rise_times)
#     return TIE - np.mean(TIE)
def get_TIE(periods):
    ideal_periods = np.mean(periods)
    TIE = periods - ideal_periods
    TIE = TIE - np.mean(TIE)
    return TIE

def get_phi(TIE, frequency):
    phi = TIE * 2 * np.pi * frequency
    centered_phi = phi #- np.mean(phi)
    return centered_phi

def get_SS_PSD(centered_phi, frequency):
    phi_fft = np.fft.rfft(centered_phi)
    #norm_phi_fft = phi_fft / len(centered_phi) # Normalized phi (phi/N)
    phi_freqs = np.fft.rfftfreq(len(centered_phi), d=1.0/frequency)
    psd_raw= (np.abs(phi_fft) ** 2) / (len(centered_phi) * frequency)
    s_phi = psd_raw.copy()
    if len(centered_phi) % 2 == 0:
        s_phi[1:-1] *= 2
    else:
        s_phi[1:] *= 2
    freq = phi_freqs
    s_phi = s_phi
    return freq, s_phi

    #ss_fft = norm_phi_fft[:len(norm_phi_fft)]*2 # Multiply by 2 to account for single-sided spectrum
    #ss_freqs = phi_freqs[:len(phi_freqs)]
    #ss_power = np.abs(ss_fft)**2
    #ss_psd = ss_power / frequency # Normalization for PSD
    return #ss_freqs, ss_psd

def plot_phase_noise(ss_freqs, ss_psd, label):
    mask = [-119, -120, -130, -140, -140, -140]
    mask_f = [1e3, 1e4, 1e5, 1e6, 1e7, 1e8]
    plt.plot(mask_f, mask, 'r--', label='Mask', alpha=0.7)
    plt.plot(ss_freqs, 10*np.log10(ss_psd/2), label=label, alpha=0.7)
    plt.xscale('log')
    plt.show()


plt.figure(figsize=(16, 6))
file = RISE_TIMES_TYP
label = 'Typical Voltage and Slew Rate corner'
rise_times = get_rise_times(file)
periods = get_periods(rise_times)
frequency = get_frequency(periods)
TIE = get_TIE(periods)
phi = get_phi(TIE, frequency)
ss_freqs, ss_psd = get_SS_PSD(phi, frequency) 
jitter_rms = np.std(TIE-np.mean(TIE))
#print(f"RMS Jitter: {jitter_rms}")

# nperseg = None
# if nperseg is None:
#     nperseg = min(len(phi), 65536)

# freq, s_phi = signal.welch(
#     phi,
#     fs=frequency,
#     window='hann',
#     nperseg=nperseg,
#     scaling='density',
#     detrend=False
# )

L_f_dbc = 10.0 * np.log10(ss_psd / 2.0)
plt.plot(ss_freqs, L_f_dbc, label=label)
mask = [-119, -120, -130, -140, -140, -140]
mask_f = [1e3, 1e4, 1e5, 1e6, 1e7, 1e8]
plt.plot(mask_f, mask, 'r--', label='Mask', alpha=0.7)
plt.xscale('log')
plt.xlim([1e3, 1e8])
plt.xlabel('Frequency (Hz)')
plt.ylabel('Phase Noise (dBc/Hz)')
plt.title('Phase Noise')
plt.legend()
plt.show()

df = frequency / len(TIE)  # Frequency bin width (Hz)
s_phi_recon = 2.0 * 10.0 ** (L_f_dbc / 10.0)
    
# Integrate discrete sum: Var = Sum(S_phi * df)
sigma_phi_sq = np.sum(s_phi_recon * df)
sigma_phi = np.sqrt(sigma_phi_sq)

rms_freq = sigma_phi / (2.0 * np.pi * frequency)

rms_time = np.std(TIE-np.mean(TIE))

# Verification Printout
print("=" * 48)
print(f"Clock Frequency (f0)        : {frequency/1e6:.3f} MHz")
print(f"Total Points (N)            : {len(TIE)}")
print(f"Resolution Bandwidth (df)   : {df:.2f} Hz")
print(f"Time-Domain RMS TIE         : {rms_time * 1e12:.6f} ps")
print(f"Recovered Frequency RMS TIE : {rms_freq * 1e12:.6f} ps")
print(f"Absolute Error              : {abs(rms_time - rms_freq) * 1e15:.4f} fs")
print("=" * 48)
print(f"RMS Phase Noise: {jitter_rms}")
#plot_phase_noise(freq, s_phi, label=label)