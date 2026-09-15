from read_waveform import read_trN
from get_cross_times import get_cross_times

files = [r"N:\pandi.clock.2\clk_sims\newport\rev_0p0\ULLoss\extclk\Clock_27MHz_LMK1C1104\Clock_27MHz_LMK1C1104_full_000001.tr0",
         r"N:\pandi.clock.2\clk_sims\newport\rev_0p0\ULLoss\extclk\Clock_27MHz_LMK1C1104\Clock_27MHz_LMK1C1104_full_000002.tr0",
         r"N:\pandi.clock.2\clk_sims\newport\rev_0p0\ULLoss\extclk\Clock_27MHz_LMK1C1104\Clock_27MHz_LMK1C1104_full_000003.tr0"]

for file in files:
    nodes = read_trN(file)
    cross_times = get_cross_times(nodes)

    output_file = file.replace(".tr0", "_cross_times.csv")
    with open(output_file, "w") as f:
        for t in cross_times:
            f.write(f"{t}\n")