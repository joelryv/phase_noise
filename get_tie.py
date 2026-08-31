import numpy as np
from dotenv import load_dotenv
import os
import re
from read_waveform import read_csv, read_trN

def get_tie(nodes):
    node_names = list(nodes.keys())
    time = np.array(nodes[node_names[0]])
    voltage = np.array(nodes[node_names[1]])

    # Find rising-edge zero crossings of the noisy waveform via linear interpolation
    voltage = voltage - np.mean(voltage)
    signs = np.sign(voltage)
    # Index where the signal crosses from negative to positive
    crossing_indices = np.where((signs[:-1] < 0) & (signs[1:] >= 0))[0]

    # Linearly interpolate to get sub-sample zero-crossing times
    crossing_times = []
    for index in crossing_indices:
        y0 = voltage[index]
        y1 = voltage[index + 1]
        fraction = (0 - y0) / (y1 - y0)
        crossing_times.append(time[index] + fraction * (time[index+1] - time[index]))
    crossing_times = np.array(crossing_times)
    periods = np.diff(crossing_times)
    avg_period = np.mean(periods)
    period_jitter = periods - avg_period
    time_interval_error = np.cumsum(period_jitter)

    return time_interval_error

load_dotenv()
file_path = str(os.getenv('PATH_TR'))

if file_path.endswith('.csv'):
    nodes = read_csv(file_path)
elif re.match(r'.*\.tr\d+$', file_path):
    nodes = read_trN(file_path)
else:
    raise ValueError(f"Unsupported file format: {file_path}")
tie = get_tie(nodes)