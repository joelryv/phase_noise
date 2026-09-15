import matplotlib.pyplot as plt
import numpy as np

files = [r"N:\pandi.clock.2\clk_sims\newport\rev_0p0\ULLoss\extclk\Clock_27MHz_LMK1C1104\Clock_27MHz_LMK1C1104_full_000001_cross_times.csv",
         r"N:\pandi.clock.2\clk_sims\newport\rev_0p0\ULLoss\extclk\Clock_27MHz_LMK1C1104\Clock_27MHz_LMK1C1104_full_000002_cross_times.csv",
         r"N:\pandi.clock.2\clk_sims\newport\rev_0p0\ULLoss\extclk\Clock_27MHz_LMK1C1104\Clock_27MHz_LMK1C1104_full_000003_cross_times.csv"]

labels = ["Min corner", "Nominal corner", "Max corner"]

plt.figure(figsize=(16, 6))
for index, file in enumerate(files):
    crossing_times = np.loadtxt(file, delimiter=",", dtype=float).reshape(-1)
    edge_index = np.arange(len(crossing_times))
    fitted_period, fitted_intercept = np.polyfit(edge_index, crossing_times, 1)
    fitted_freq = 1 / fitted_period
    print(f"Fitted frequency in {labels[index]}: {fitted_freq/1e6:.6f} MHz")
    print(f"Average frequency in {labels[index]}: {np.mean(1/np.diff(crossing_times))/1e6:.6f} MHz")
    ideal_crossing_times = fitted_intercept + fitted_period * edge_index
    tie = crossing_times - ideal_crossing_times
    phi = tie * 2 * np.pi * fitted_freq 
    phi_fft = np.fft.rfft(phi)
    freqs = np.fft.rfftfreq(len(phi), d=1/fitted_freq)
    psd = (np.abs(phi_fft)**2) / (fitted_freq * len(phi))
    if len(psd) % 2 == 0:
        psd[1:-1] *= 2.0
    else:
        psd[1:] *= 2.0
    l_psd = 10 * np.log10(psd/2)
    plt.plot(freqs, l_psd, label=labels[index])

db_mask = [-119, -120, -130, -140, -140, -140]
mask_f = [1e3, 1e4, 1e5, 1e6, 1e7, 1e8]
plt.plot(mask_f, db_mask, 'k--', label="Mask")
plt.xlabel("Frequency (Hz)")
plt.xlim([1e3, 1e8])
plt.xscale("log")
plt.ylabel("Phase Noise (dBc/Hz)")
plt.legend()
plt.show()