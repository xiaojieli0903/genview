import argparse
import numpy as np
import os
import torch
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import torchvision.transforms as transforms


def read_image_paths(image_paths_file, prefix=''):
    """
    Read image paths from file and optionally prepend a prefix.
    """
    with open(image_paths_file, 'r') as file:
        paths = file.read().splitlines()
    return [os.path.join(prefix, path) for path in paths]


def load_existing_ratios(output_file):
    """
    Load already processed image paths to skip redundant computations.
    """
    if os.path.exists(output_file):
        with open(output_file, 'r') as file:
            return {line.split(' ')[0] for line in file.readlines()}
    return set()


def load_pca_vectors(output_dir):
    """
    Load PCA vectors for feature transformation.
    """
    pca_path = os.path.join(output_dir, 'eigenvecters', 'pca_vectors.npy')
    if not os.path.exists(pca_path):
        raise FileNotFoundError(f"PCA vectors not found at {pca_path}")
    pca_vectors = np.load(pca_path)
    print(f"Loaded PCA vectors from {pca_path}. shape={pca_vectors.shape}")
    return pca_vectors


def calculate_ratios(image_paths, output_dir, feature_dir, pca_vectors, threshold, patch_number, batch_size, mask_type):
    """
    Calculate the foreground ratios for a batch of images using PCA features.

    Args:
        image_paths (list): List of image paths to process.
        output_dir (str): Directory to save outputs.
        pca_vectors (numpy.ndarray): PCA vectors for feature projection.
        threshold (float): Threshold to determine foreground/background.
        patch_number (int): Number of patches per dimension.
        batch_size (int): Number of images processed in one batch.
        mask_type (str): 'gt' for greater-than threshold, 'lt' for less-than threshold.
    """
    output_file = os.path.join(output_dir, 'fg_ratios.txt')
    existing_paths = load_existing_ratios(output_file)

    with open(output_file, 'a+') as fout:
        for start in tqdm(range(0, len(image_paths), batch_size), desc="Processing Batches"):
            batch_paths = image_paths[start:start + batch_size]
            batch_features, valid_paths = [], []

            # Load features for the batch
            for path in batch_paths:
                if path in existing_paths:
                    continue
                feature_path = os.path.join(feature_dir, 'features', os.path.basename(path) + '.npy')
                if os.path.exists(feature_path):
                    batch_features.append(np.load(feature_path))
                    valid_paths.append(path)
                else:
                    print(f'{feature_path} not found.')

            if not batch_features:  # Skip if no valid features are loaded
                print(f"No features found in {batch_features}")
                continue

            # Convert to tensors for computation
            batch_features_tensor = torch.tensor(batch_features, dtype=torch.float32).to('cuda')
            pca_vectors_tensor = torch.tensor(pca_vectors, dtype=torch.float32).to('cuda')

            # Project features using PCA vectors
            projected_features = torch.matmul(batch_features_tensor, pca_vectors_tensor)

            # Apply thresholding to calculate masks
            if mask_type == 'gt':
                masks = projected_features > threshold
            else:
                masks = projected_features <= threshold

            # Calculate foreground ratios
            ratios = masks.sum(dim=1).float() / (patch_number * patch_number)

            # Save results
            for path, ratio in zip(valid_paths, ratios.cpu().numpy()):
                fout.write(f"{path} {float(ratio)}\n")


def main():
    """
    Main function to calculate foreground ratios for images based on PCA-transformed features.
    """
    parser = argparse.ArgumentParser(description="Calculate Foreground Ratios Based on PCA Features")
    parser.add_argument('--input-list', type=str, required=True, help='Path to the text file with image file paths.')
    parser.add_argument('--input-prefix', type=str, default='', help='Optional prefix for image paths.')
    parser.add_argument('--input-dir', type=str, required=True, help='Directory to save features and PCA eigenvecters.')
    parser.add_argument('--output-dir', type=str, required=True, help='Directory to save outputs.')
    parser.add_argument('--fg-thre', type=float, required=True, help='Threshold for calculate foreground masks.')
    parser.add_argument('--batch-size', type=int, default=512, help='Number of images processed per batch.')
    parser.add_argument('--mask-type', type=str, choices=['gt', 'lt'], default='gt', help="Mask type: 'gt' or 'lt'.")
    parser.add_argument('--patch-size', type=int, default=14, help='Patch size for dividing images into patches')
    args = parser.parse_args()

    DEFAULT_SMALLER_EDGE_SIZE = 224
    patch_number = DEFAULT_SMALLER_EDGE_SIZE // args.patch_size

    # Load PCA vectors
    pca_vectors = load_pca_vectors(args.input_dir)

    # Prepare image paths and output directory
    image_paths = read_image_paths(args.input_list, prefix=args.input_prefix)

    # Calculate ratios
    calculate_ratios(
        image_paths=image_paths,
        output_dir=args.output_dir,
        feature_dir=args.input_dir,
        pca_vectors=pca_vectors,
        threshold=args.fg_thre,
        patch_number=patch_number,
        batch_size=args.batch_size,
        mask_type=args.mask_type
    )
    print("Foreground ratio calculation complete.")


if __name__ == '__main__':
    main()
