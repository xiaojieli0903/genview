import argparse
import numpy as np
import os

def map_fg_ratio_to_range(value, ranges, mapping_values):
    """
    Maps a fg_ratio value to a specific range and returns the corresponding mapping value.
    """
    index = np.digitize(value, ranges, right=True) - 1
    index = min(max(index, 0), len(mapping_values) - 1)  # Ensure index is valid
    return mapping_values[index]

def distribute_fg_ratios(input_file, output_dir, ranges, mapping_values):
    """
    Reads fg_ratio values, maps them to specified ranges, and writes to separate files.
    """
    os.makedirs(output_dir, exist_ok=True)
    buffers = {value: [] for value in mapping_values}

    with open(input_file, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) != 2:
                print(f"Skipping invalid line: {line.strip()}")
                continue
            image_path, fg_ratio_str = parts
            try:
                fg_ratio = float(fg_ratio_str)
                mapped_value = map_fg_ratio_to_range(fg_ratio, ranges, mapping_values)
                buffers[mapped_value].append(f"{image_path} {mapped_value}\n")
            except ValueError:
                print(f"Skipping invalid fg_ratio: {fg_ratio_str}")

    for value, lines in buffers.items():
        if lines:
            output_file_path = os.path.join(output_dir, f'fg_ratios_{value}.txt')
            with open(output_file_path, 'w') as out_file:
                out_file.writelines(lines)
            print(f"File saved: {output_file_path}")
        else:
            print(f"No data for fg_ratios_{value}.txt")

    print(f"Distribution completed. Files saved in {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Distribute FG Ratios into Ranges')
    parser.add_argument('--input-file', type=str, required=True, help='Path to the fg_ratios.txt file')
    parser.add_argument('--output-dir', type=str, required=True, help='Directory to save the distributed fg_ratio files')
    args = parser.parse_args()

    # Validate input file existence
    if not os.path.isfile(args.input_file):
        raise FileNotFoundError(f"Input file not found: {args.input_file}")

    # Define ranges and corresponding mapping values
    ranges = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    mapping_values = [0, 100, 200, 300, 400, 500]

    # Run distribution
    distribute_fg_ratios(args.input_file, args.output_dir, ranges, mapping_values)

if __name__ == '__main__':
    main()

