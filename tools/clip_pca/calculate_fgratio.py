import argparse
import numpy as np
import os
import torch
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import torchvision.transforms as transforms


def create_output_directory(output_dir):
    subdirs = ['features']
    for subdir in subdirs:
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)


def read_image_paths_from_file(image_paths_file):
    with open(image_paths_file, 'r') as file:
        return file.read().splitlines()


def calculate_ratios_batch(image_paths, output_dir, standard_array,
                           background_threshold=0.05, patch_number=16,
                           batch_size=512, mask_type='gt'):
    output_file = os.path.join(output_dir, 'fg_ratios.txt')
    if os.path.exists(output_file):
        with open(output_file, 'r') as file:
            existing_files = {line.split(' ')[0] for line in file.readlines()}
    else:
        existing_files = set()

    with open(output_file, 'a+') as fout:
        for start in tqdm(range(0, len(image_paths), batch_size),
                          desc="Processing"):
            batch_paths = image_paths[start:start + batch_size]
            batch_features, filtered_paths = [], []

            for path in batch_paths:
                if path in existing_files:
                    continue
                feature_path = os.path.join(output_dir, 'features',
                                            os.path.basename(path) + '.npy')
                if os.path.exists(feature_path):
                    batch_features.append(np.load(feature_path))
                    filtered_paths.append(path)

            if not batch_features:
                continue

            batch_features = torch.tensor(batch_features).float().to('cuda')
            standard_array_t = torch.tensor(standard_array).float().to('cuda')
            batch_pca_features = torch.matmul(batch_features, standard_array_t)
            batch_masks = batch_pca_features > background_threshold if mask_type == 'gt' else batch_pca_features <= background_threshold
            batch_ratios = batch_masks.sum(dim=1).float() / (
                        patch_number * patch_number)

            for path, ratio in zip(filtered_paths, batch_ratios.cpu().numpy()):
                fout.write(f'{path} {ratio}\n')


def main():
    parser = argparse.ArgumentParser(
        description='CLIP Feature Extraction and PCA')
    parser.add_argument('--input-list', type=str, required=True,
                        help='Path to the text file with image file paths')
    parser.add_argument('--input-prefix', type=str, default='',
                        help='Prefix to be added to each image path')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Directory to store output files')
    parser.add_argument('--background-threshold', type=float, default=-4.1825733,
                        help='Threshold for creating masks')
    parser.add_argument('--batch-size', type=int, default=512,
                        help='Number of images to process in a batch')
    parser.add_argument('--mask-type', type=str, choices=['gt', 'lt'],
                        default='lt',
                        help="Mask type based on threshold: 'gt' for greater than, 'lt' for less than")
    args = parser.parse_args()


    # model_name = 'ViT-B-16'
    # pretrained = 'laion2B-s34B-b88K'
    # background_threshold = -4.1825733

    patch_size = 16
    DEFAULT_SMALLER_EDGE_SIZE = 224
    patch_number = DEFAULT_SMALLER_EDGE_SIZE // patch_size

    standard_array_path = os.path.join(args.output_dir, 'eigenvalues',
                                       'pca_vectors.npy')
    standard_array = np.load(standard_array_path)

    print(f'Load {standard_array_path} done, shape = {standard_array.shape}')

    image_paths = read_image_paths_from_file(args.input_list)
    if args.input_prefix:
        image_paths = [os.path.join(args.input_prefix, path) for path in
                       image_paths]

    create_output_directory(args.output_dir)
    standard_array = np.load(
        os.path.join(args.output_dir, 'eigenvalues', 'pca_vectors.npy'))

    calculate_ratios_batch(image_paths, args.output_dir, standard_array,
                           background_threshold=args.background_threshold,
                           patch_number=patch_number, batch_size=args.batch_size,
                           mask_type=args.mask_type)


if __name__ == '__main__':
    main()
