import numpy as np

def get_cross_times(nodes):
    node_names = list(nodes.keys())
    time = np.array(nodes[node_names[0]])
    voltage = np.array(nodes[node_names[1]])

    # Find rising-edge zero crossings of the noisy waveform via linear interpolation
    voltage = voltage - np.mean(voltage)
    signs = np.sign(voltage)
    # Index where the signal crosses from negative to positive
    crossing_indices = np.where((signs[:-1] < 0) & (signs[1:] >= 0))[0]

    # Linearly interpolate to get sub-sample zero-crossing times
    cross_times = []
    for index in crossing_indices:
        y0 = voltage[index]
        y1 = voltage[index + 1]
        fraction = (0 - y0) / (y1 - y0)
        cross_times.append(time[index] + fraction * (time[index+1] - time[index]))
    cross_times = np.array(cross_times)
    return cross_times

if __name__ == "__main__":
    from read_waveform import read_bin
    file_path = r"C:\Users\jreyesva\OneDrive - Intel Corporation\Documents - EPSI Clock KIT 1\Lab\OKS\Experiments SSC ON_OFF\WW35p3 measurements\Measurements\01_A_CPU0_DB1206_FE_CPUTx_JT4K43_SSCON_CH1P_CH3N_110us.bin"
    nodes = read_bin(file_path)
    cross_times = get_cross_times(nodes)

    with open("01_A_CPU0_DB1206_FE_CPUTx_JT4K43_SSCON_CH1P_CH3N_110us.csv", "w") as f:
        for t in cross_times:
            f.write(f"{t}\n")
