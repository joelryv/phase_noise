import numpy as np
import struct

def read_csv(file_path: str, voltage_column: int = 1) -> dict:
    with open(file_path, 'r') as f:
        skip = 0
        header = []
        while True:
            line = [x for x in f.readline().strip().split(',')]
            try:
                line = [float(x) for x in line]
                break
            except ValueError:
                skip += 1
                header = line
                continue

    if len(line) != len(header):
        header = [f"col_{i}" for i in range(len(line))]
    data = np.loadtxt(file_path, delimiter=',', skiprows=skip)
    nodes = {}
    for i, name in enumerate(header):
        nodes[name] = data[:, i]
    return nodes

def read_trN(file_path: str):
    with open(file_path, "rb") as f:
        endianness = '<'
        preamble = f.read(16)
        preamble_int = struct.unpack(endianness + ('I' * 4), preamble) # Unpack the first 16 bytes of the file as four unsigned integers using the determined endianness
        if preamble_int[0] != 4: # Check if the first integer in the preamble is not 4, indicating a different endianness
            endianness = '>'
            preamble_int = struct.unpack(endianness + ('I' * 4), preamble) # Re-unpack the preamble with the new endianness
        block_size = preamble_int[3] # The fourth integer in the preamble represents the block size
        header = f.read(block_size) # Read the header block based on the block size
        n_nodes = int(header[:4]) # The number of nodes is stored in the first 4 bytes of the header
        version = int(header[16:24]) # The version information is stored in bytes 16 to 24 of the header
        footer = f.read(4) # Read the footer block, which is typically 4 bytes long
        node_names = [x.decode() for x in header.split()[-(n_nodes+1):-1]] # Extract the node names from the header based on the number of nodes
        if version == 2001:
            d_type = 'd'
            d_bytes = 8
        elif version == 9601:
            d_type = 'f'
            d_bytes = 4
        else:
            raise ValueError(f"Unsupported version: {version}")
        block_data = []
        while True:
            header = f.read(16)
            if not header:
                break
            header = struct.unpack(endianness + ('i' * 4), header)
            block_size = header[3]
            block_data += struct.unpack(endianness + (d_type * (block_size // d_bytes)), f.read(block_size))
            footer = f.read(4)
        nodes = {}
        for index, node_name in enumerate(node_names):
            nodes[node_name] = block_data[index:-1:len(node_names)] # Assign every nth element to the corresponding node, skipping the last element which is file's end marker (1e30)
    return nodes

def read_bin(file_path: str):
    with open(file_path, 'rb') as raw:
        file_cookie = raw.read(2).decode("utf-8")
        file_version = raw.read(2).decode("utf-8")
        file_size = struct.unpack('<i', raw.read(4))[0]
        n_waveforms = struct.unpack('<i', raw.read(4))[0]
        # Read header
        header_size = struct.unpack('<i', raw.read(4))[0]
        waveform_type = struct.unpack('<i', raw.read(4))[0]
        n_waveform_buffers = struct.unpack('<i', raw.read(4))[0]
        n_points = struct.unpack('<i', raw.read(4))[0]
        count = struct.unpack('<i', raw.read(4))[0]
        x_display_range = struct.unpack('<f', raw.read(4))[0]
        x_display_origin = struct.unpack('<d', raw.read(8))[0]
        x_step = struct.unpack('<d', raw.read(8))[0]
        x_origin = struct.unpack('<d', raw.read(8))[0]
        x_units = struct.unpack('<i', raw.read(4))[0]
        y_units = struct.unpack('<i', raw.read(4))[0]
        date = raw.read(16).decode("utf-8")
        time = raw.read(16).decode("utf-8")
        frame = raw.read(24).decode("utf-8")
        waveform_string = raw.read(16).decode("utf-8")
        time_tag = struct.unpack('<d', raw.read(8))[0]
        segment_index = struct.unpack('<I', raw.read(4))[0]
        remaining_byes = header_size - 140 # Total header bytes minus header bytes already readed
        raw.read(remaining_byes)
        
        # Generate time waveform
        x_values = np.arange(x_origin, (n_points*x_step)+x_origin, x_step)

        # Read waveform
        y_values = 0
        for i in range(n_waveform_buffers):
            header_size = struct.unpack('<i', raw.read(4))[0]
            buffer_type = struct.unpack('<h', raw.read(2))[0]
            bytes_per_point = struct.unpack('<h', raw.read(2))[0]
            buffer_size = struct.unpack('<i', raw.read(4))[0]
            remaining_byes = header_size - 12
            raw.read(remaining_byes)

            if buffer_type==1 or buffer_type==2 or buffer_type==3:
                y_values = struct.unpack('<'+'f'*n_points, raw.read(4*n_points))
            elif buffer_type==4:
                y_values = struct.unpack('<'+'i'*n_points, raw.read(4*n_points))
            elif buffer_type==5:
                y_values = struct.unpack('<'+'B'*n_points, raw.read(1*n_points))
            else:
                y_values = struct.unpack('<'+'B'*buffer_size, raw.read(1*buffer_size))
        y_values = np.array(y_values)
    return {"x_values": x_values, "y_values": y_values}


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    file_path = r"C:\Users\jreyesva\OneDrive - Intel Corporation\Documents\Proyectos Intel\Kaseville (KSV)\Measurements\03_CLKOUT_SRC_0_Jitter.bin"
    nodes = read_bin(file_path)
    node_names = list(nodes.keys())
    print(node_names)
    plt.plot(nodes[node_names[0]], nodes[node_names[1]])
    plt.show()
