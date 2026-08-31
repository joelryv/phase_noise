import numpy as np
import os

file_path = str(os.getenv('MEASUREMENT'))

with open(file_path, 'r') as f:
    f.readline()
    time = []
    voltage = []
    line = f.readline()
    i = 0
    while line:
        line = line.strip().split(',')
        if len(line) == 1:
            break
        time.append(float(line[0]))
        voltage.append(float(line[1]))
        line = f.readline()

time = np.array(time)
voltage = np.array(voltage)

import matplotlib.pyplot as plt

# Estimate carrier frequency from the FFT (use the dominant spectral peak)
delta_time = (time[-1] - time[0])/(len(time)-1)
sample_rate = 1/delta_time
spectrum = np.fft.rfft(voltage)
freqs = np.fft.rfftfreq(voltage.size, d=1/sample_rate)
spectrum[0] = 0
measured_carrier_frequency = freqs[np.argmax(np.abs(spectrum))]
print(f"Measured carrier frequency: {measured_carrier_frequency:,.1f} Hz")

# Find rising-edge zero crossings of the noisy waveform via linear interpolation
voltage = voltage - np.mean(voltage)
signs = np.sign(voltage)
# indices where the signal crosses from negative to positive
crossing_indices = np.where((signs[:-1] < 0) & (signs[1:] >= 0))[0]

# Linearly interpolate to get sub-sample zero-crossing times
y0 = voltage[crossing_indices]
y1 = voltage[crossing_indices + 1]
fraction = -y0 / (y1 - y0)
crossing_times = (crossing_indices + fraction) / sample_rate

# Ideal zero-crossing times based on the measured carrier frequency
#carrier_period = 1 / measured_carrier_frequency
# Align the ideal reference to the first measured crossing
#first_ideal_edge = crossing_times[0]
#ideal_edge_numbers = np.arange(crossing_times.size) * carrier_period
#ideal_crossing_times = ideal_edge_numbers + first_ideal_edge

# Time Interval Error = actual crossing time - ideal crossing time
#tie = (crossing_times - ideal_crossing_times)

with open('TIE_test_matins.csv', 'w') as o:
    for value in crossing_times:
        o.write('{}\n'.format(value))