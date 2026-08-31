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


def read_tr0(file_path: str):
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

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    file_path = r"C:\Users\jreyesva\OneDrive - Intel Corporation\Documents\CKpro\CKPro_test\Diff\CLK_100M_NE_PORT2_Diodes_000001.csv"
    nodes = read_csv(file_path)
    node_names = list(nodes.keys())
    print(node_names)
    plt.plot(nodes[node_names[0]], nodes[node_names[1]])
    plt.show()
