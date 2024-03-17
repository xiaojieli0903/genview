import argparse
import numpy as np
import os

def map_fg_ratio_to_range(value, ranges):
    """
    Maps a fg_ratio value to a specific range.
    """
    index = np.digitize(value, ranges) - 1
    return ranges[index]

def distribute_fg_ratios(input_file, output_dir, ranges, mapping_values):
    """
    Reads fg_ratio values, maps them to specified ranges, and writes to separate files.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_files = [os.path.join(output_dir, f'fg_ratios_{value}.txt') for value in mapping_values]
    buffers = {value: [] for value in mapping_values}

    with open(input_file, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            image_path, fg_ratio_str = parts
            fg_ratio = float(fg_ratio_str)
            mapped_value = map_fg_ratio_to_range(fg_ratio, ranges)
            buffers[mapped_value].append(f"{image_path} {fg_ratio}\n")

    for value, lines in buffers.items():
        if lines:
            with open(os.path.join(output_dir, f'fg_ratios_{value}.txt'), 'w') as out_file:
                out_file.writelines(lines)

    print(f"Distribution completed. Files saved in {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Distribute FG Ratios into Ranges')
    parser.add_argument('--input-file', type=str, required=True, help='Path to the fg_ratios.txt file')
    parser.add_argument('--output-dir', type=str, required=True, help='Directory to save the distributed fg_ratio files')
    args = parser.parse_args()

    ranges = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    mapping_values = [0, 100, 200, 300, 400, 500]

    distribute_fg_ratios(args.input_file, args.output_dir, ranges, mapping_values)

if __name__ == '__main__':
    main()

