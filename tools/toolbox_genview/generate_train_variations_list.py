import os
import argparse


def generate_image_list(input_dir, output_file):
    """
    Generate a list of image paths from the directory structure.

    Args:
        input_dir (str): Path to the input directory ('train_variations').
        output_file (str): Path to the output list file.
    """
    with open(output_file, 'w') as f_out:
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    # Generate relative path from input_dir to file
                    rel_path = os.path.relpath(os.path.join(root, file),
                                               input_dir)
                    f_out.write(f"{rel_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a list of image paths from 'train_variations'.")
    parser.add_argument('--input-dir', required=True,
                        help='Path to the input directory containing train variations.')
    parser.add_argument('--output-list', required=True,
                        help='Path to the output list file.')

    args = parser.parse_args()

    # Make sure output directory exists
    os.makedirs(os.path.dirname(args.output_list), exist_ok=True)

    generate_image_list(args.input_dir, args.output_list)
