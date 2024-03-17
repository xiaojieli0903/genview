import open_clip
import torch
import numpy as np
import os
import argparse
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import torchvision.transforms as transforms


# Set the device to use for computation. Default is 'cuda' for GPU acceleration.
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def PCA(data, num_components=2):
    """
    Performs PCA on the given data and returns the projected data along with eigenvalues and eigenvectors.
    """
    data = data.float()
    mean = torch.mean(data, dim=0)
    zero_centered_data = data - mean
    cov_matrix = torch.mm(zero_centered_data.t(), zero_centered_data) / (zero_centered_data.size(0) - 1)
    eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)
    sorted_indices = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]
    selected_eigenvectors = eigenvectors[:, :num_components]
    projected_data = torch.mm(data, selected_eigenvectors)
    return np.array(projected_data), np.array(selected_eigenvectors), np.array(mean)


def read_image_paths_from_file(image_paths_file):
    """
    Reads image paths from a given file and returns a list of paths.
    """
    with open(image_paths_file, 'r') as f:
        return f.read().splitlines()


def create_output_directory(output_dir):
    """
    Creates the necessary directories for saving output data.
    """
    directories = ['/masks', '/maps', '/features', '/original_images', '/eigenvalues']
    for dir in directories:
        os.makedirs(os.path.join(output_dir, dir), exist_ok=True)


def load_and_process_image(image_path, preprocess):
    """
    Loads an image from a given path, applies preprocessing, and returns the processed image tensor.
    """
    image = Image.open(image_path)
    clip_input = preprocess(image).unsqueeze(0).to(device)
    return clip_input


def extract_features(image_paths, output_dir, model, preprocess, batch_size=1024, save_all=False):
    """
    Extracts features from images in batches, optionally saves them to disk, and returns all extracted features
    along with any original images if required for further processing.
    Parameters:
        image_paths (list): List of paths to the images.
        output_dir (str): Directory where outputs will be saved.
        model (open_clip.CLIP): Initialized CLIP model.
        preprocess (function): Preprocessing function for the images compatible with the CLIP model.
        batch_size (int): Number of images to process in a single batch.
        save_all (bool): Whether to save all extracted features to disk.
    Returns:
        all_patch_features (np.array): Array of all extracted features.
        original_images (list): List of original images (PIL Images).
    """
    original_images = []
    all_patch_features_path = os.path.join(output_dir, 'features',
                                           f'features_all{len(image_paths)}.npy')
    if os.path.exists(all_patch_features_path):
        all_patch_features = np.load(all_patch_features_path)
        print(
            f'Load {all_patch_features_path} done, shape={all_patch_features.shape}.')
        return all_patch_features, original_images

    idx = 0
    batch_images = []
    batch_feature_files = []
    all_patch_features = []
    batch_features = []
    batch_paths = []

    for image_path in image_paths:
        if idx % 100 == 0:
            print(f'Batch processing: {idx}')
        batch_paths.append(image_path)
        batch_feature_files.append(os.path.join(output_dir, 'features',
                                                os.path.basename(
                                                    image_path) + '.npy'))
        if len(batch_paths) == batch_size:
            batch_feature_files = batch_feature_files[:batch_size]
            # Check if any of the images in this batch already have feature files
            extract_batch = False
            for feature_file in batch_feature_files:
                if not os.path.exists(feature_file):
                    extract_batch = True
                    batch_features = []
                    break
                else:
                    print(f'{idx}, Existing {feature_file}. Save_all={save_all}')
                    if save_all:
                        feature_array = np.load(feature_file)
                        feature_array = feature_array.reshape(1, feature_array.shape[0], -1)
                        batch_features.append(feature_array)

            if extract_batch:
                print(f'Extracting batch images...')
                idx_in = 0
                for image_path in batch_paths:
                    image = Image.open(image_path).convert('RGB')
                    if len(original_images) <= 100:
                        original_images.append(image)
                    clip_input = preprocess(image).unsqueeze(0).to(device)
                    batch_images.append(clip_input)
                    if idx_in % 100 == 0:
                        print(f'Batch processing image: {idx_in}')
                    idx_in += 1
                batch_images = torch.cat(batch_images, dim=0)
                with torch.no_grad(), torch.cuda.amp.autocast():
                    _, batch_patch_features = model.encode_image(
                        batch_images)
                    if len(batch_patch_features.shape) == 4:
                        batch_patch_features = batch_patch_features.permute(
                            0, 2, 3, 1).view(batch_patch_features.shape[0],
                                             -1,
                                             batch_patch_features.shape[1])
                        print(batch_patch_features.shape)
                batch_features = batch_patch_features.cpu().numpy()

                for i, feature_file in enumerate(batch_feature_files):
                    features = batch_features[i]
                    np.save(feature_file, features)

            if save_all:
                all_patch_features.append(
                    np.concatenate(batch_features, axis=0))

            batch_images = []
            batch_feature_files = []
            batch_features = []
            batch_paths = []

        idx += 1

    # Process the remaining images
    if batch_paths:
        idx_in = 0
        for image_path in batch_paths:
            image = Image.open(image_path).convert('RGB')
            if len(original_images) <= 100:
                original_images.append(image)
            clip_input = preprocess(image).unsqueeze(0).to(device)
            batch_images.append(clip_input)
            if idx_in % 100 == 0:
                print(f'Last-iter: Batch processing image: {idx_in}')
            idx_in += 1
        batch_images = torch.cat(batch_images, dim=0)
        batch_feature_files = batch_feature_files[:len(batch_images)]

        # Check if any of the images in this batch already have feature files
        extract_batch = False
        for feature_file in batch_feature_files:
            if not os.path.exists(feature_file):
                extract_batch = True
                batch_features = []
                break
            else:
                print(f'{idx}, Existing {feature_file}. Save_all={save_all}')
                if save_all:
                    feature_array = np.load(feature_file)
                    feature_array = feature_array.reshape(1, feature_array.shape[0], -1)
                    batch_features.append(feature_array)

        if extract_batch:
            print(f'Last-iter: Extracting batch images...')
            with torch.no_grad(), torch.cuda.amp.autocast():
                _, batch_patch_features = model.encode_image(batch_images)
                if len(batch_patch_features.shape) == 4:
                    batch_patch_features = batch_patch_features.permute(
                        0, 2, 3, 1).view(
                        batch_patch_features.shape[0], -1, batch_patch_features.shape[1])
                    print(batch_patch_features.shape)
            batch_features = batch_patch_features.cpu().numpy()

            for i, feature_file in enumerate(batch_feature_files):
                features = batch_features[i]
                np.save(feature_file, features)

        if save_all:
            all_patch_features.append(np.concatenate(batch_features, axis=0))

    if save_all:
        all_patch_features = [af for af in all_patch_features if
                              af is not None]
        all_patch_features = np.concatenate(all_patch_features, axis=0)
        all_patch_features = all_patch_features.reshape(len(image_paths), -1,
                                                        all_patch_features.shape[
                                                            -1])
        np.save(all_patch_features_path, all_patch_features)
    return all_patch_features, original_images


def apply_pca(features, n_components=3):
    """
    Applies PCA to the extracted features.
    """
    return PCA(torch.from_numpy(features), n_components)


def visualize_pca_histograms(pca_features, output_dir):
    fig, axs = plt.subplots(2, 2, figsize=(20, 20))
    value_dict = {}
    prob_dict = {}
    for i in range(pca_features.shape[1]):
        pca_feature_i = pca_features[:, i]
        plt.subplot(2, 2, i + 1)
        plt.hist(pca_feature_i, bins=200)
        plt.subplot(2, 2, i + 2)
        values, base = np.histogram(pca_feature_i, bins=200)
        # evaluate the cumulative
        cumulative = np.cumsum(values) / np.sum(values)
        for k in range(base[:-1].shape[0]):
            value_dict[base[:-1][k]] = cumulative[k]
        plt.plot(base[:-1], cumulative, c='blue')
        plt.subplot(2, 2, i + 3)
        plt.plot(cumulative, base[:-1], c='red')
        for k in range(base[:-1].shape[0]):
            prob_dict[cumulative[k]] = base[:-1][k]

    fig.savefig(os.path.join(output_dir, 'full_figure.png'))
    return value_dict, prob_dict


def create_masks(pca_features, background_threshold, output_dir, patch_number, mask_type=''):
    with tqdm(total=pca_features.shape[0], desc="Creating Masks") as pbar:
        for i in range(pca_features.shape[0]):
            for k in range(pca_features.shape[-1]):
                mask = pca_features[i, :, k] > background_threshold
                resized_mask = mask.reshape(patch_number, patch_number)
                mask_array = (resized_mask > 0).astype(np.uint8)
                mask_array_resized_img = Image.fromarray(
                    (mask_array * 255).astype(np.uint8))
                mask_file = os.path.join(output_dir, 'masks',
                                         f'{i}_mask{mask_type}-{k}_thre{background_threshold}.png')
                mask_array_resized_img.save(mask_file)
            pbar.update(1)


def create_maps(pca_features, background_threshold, output_dir, patch_number, mask_type=''):
    with tqdm(total=pca_features.shape[0], desc="Creating Maps") as pbar:
        for i in range(pca_features.shape[0]):
            for k in range(pca_features.shape[-1]):
                pca_feature = pca_features[i, :, k].copy()
                pca_feature = (pca_feature - pca_feature.min()) / (pca_feature.max() - pca_feature.min())
                resized_map = pca_feature.reshape(patch_number, patch_number)
                resized_map_img = Image.fromarray(
                    (resized_map * 255).astype(np.uint8))
                map_file = os.path.join(output_dir, 'maps',
                                        f'{i}_fg_map{mask_type}-{k}.png')
                resized_map_img.save(map_file)
                resized_map_bg = 1 - resized_map
                resized_map_bg_img = Image.fromarray(
                    (resized_map_bg * 255).astype(np.uint8))
                map_file = os.path.join(output_dir, 'maps',
                                        f'{i}_bg_map{mask_type}-{k}.png')
                resized_map_bg_img.save(map_file)
            pbar.update(1)


def save_original_images(original_images, output_dir):
    with tqdm(total=len(original_images),
              desc="Saving Original Images") as pbar:
        for i, image in enumerate(original_images):
            image_file = os.path.join(output_dir, 'original_images',
                                      f'original_image_{i}.jpg')
            image = resize_image(image, DEFAULT_SMALLER_EDGE_SIZE)
            image.save(image_file)
            pbar.update(1)


def main():
    """
    Main function to execute the feature extraction and PCA analysis workflow.
    This script extracts features from images using the CLIP model, performs
    Principal Component Analysis (PCA) on these features, and visualizes the
    results. It can also create masks and maps based on the PCA features to
    identify significant areas in images.
    """
    parser = argparse.ArgumentParser(description='CLIP Feature Extraction and PCA Analysis')
    parser.add_argument('--input-list', type=str, required=True, help='Path to the text file with image file paths')
    parser.add_argument('--output-dir', type=str, required=True, help='Directory to store feature files and PCA results')
    parser.add_argument('--batch-size', type=int, default=1024, help='Number of images to process in a batch')
    parser.add_argument('--save', action='store_true', help='Save extracted features to disk')
    parser.add_argument('--model-name', type=str, default='ViT-B-16', help='Model name to use for feature extraction')
    parser.add_argument('--pretrained-weights', type=str, default='laion2B-s34B-b88K', help='Pretrained weights for the model')
    parser.add_argument('--patch-size', type=int, default=16, help='Patch size for the model')
    parser.add_argument('--smaller-edge-size', type=int, default=224, help='Size of the smaller edge for image resizing')
    parser.add_argument('--num', type=int, default=10, help='Number of example images for PCA visualization and mask/map creation')
    parser.add_argument('--background-threshold', type=float, default=0.5, help='Background threshold for creating masks (optional)')
    args = parser.parse_args()

    # For adaptive view generation.
    # model_name = 'ViT-B-16'
    # pretrained = 'laion2B-s34B-b88K'
    # patch_size = 16

    # For quality-driven contrastive loss.
    # model_name = 'convnext_base_w'
    # pretrained = 'laion2B-s13B-b82K-augreg'
    # patch_size = 32

    patch_number = args.smaller_edge_size // args.patch_size

    print(f'Loading model: {args.model_name} with pretrained weights: {args.pretrained_weights}')
    model, _, preprocess = open_clip.create_model_and_transforms(args.model_name, pretrained=args.pretrained_weights, device=device)

    # Read image paths and create output directories
    image_paths = read_image_paths_from_file(args.input_list)
    create_output_directory(args.output_dir)

    # Extract features and optionally save them
    features, original_images = extract_features(image_paths, args.output_dir,
                                                 model, preprocess,
                                                 batch_size=args.batch_size,
                                                 save_all=args.save)
    if features is None:
        print('No features extracted. Exiting...')
        return

    # Apply PCA to the features
    pca_features, pca_vectors, mean_vectors = apply_pca(features,
                                                        n_components=3)
    # Save PCA vectors and mean vectors
    pca_vectors_path = os.path.join(args.output_dir, 'eigenvalues',
                                    'pca_vectors.npy')
    mean_vectors_path = os.path.join(args.output_dir, 'eigenvalues',
                                     'mean_vectors.npy')
    np.save(pca_vectors_path, pca_vectors)
    np.save(mean_vectors_path, mean_vectors)
    print(f'PCA vectors and mean vectors saved.')

    # Visualize PCA histograms
    value_dict, prob_dict = visualize_pca_histograms(pca_features,
                                                     args.output_dir)

    # Determine threshold for mask creation based on PCA histogram
    background_threshold = determine_threshold_from_histogram(prob_dict,
                                                              args.background_threshold)

    # Create masks and maps
    create_masks(pca_features.reshape(N, n_patches, -1)[:args.num], background_threshold, patch_number, args.output_dir)
    create_maps(pca_features.reshape(N, n_patches, -1)[:args.num], background_threshold, patch_number, args.output_dir)

    # Save a subset of the original images for reference
    save_original_images(original_images[:args.num], args.output_dir)

    print('Feature extraction and PCA analysis complete.')


def determine_threshold_from_histogram(prob_dict,
                                       default_threshold):
    """
    Determines an appropriate threshold for mask creation based on the PCA histograms.
    If a suitable threshold is found based on the criteria, it returns that value.
    Otherwise, it returns the default threshold.

    Parameters:
        prob_dict (dict): A dictionary mapping cumulative distribution values to histogram bin values.
        default_threshold (float): The default background threshold if no suitable value is found.
    Returns:
        float: The determined threshold for background separation in masks.
    """
    result_key_prob = None
    if default_threshold is not None:
        return default_threshold
    for key in prob_dict.keys():
        if key >= 0.6:  # This threshold value (0.6) is adjustable based on empirical results or specific criteria.
            result_key_prob = key
            result_key_prob_value = prob_dict[key]
            break
    print(
        f"Selected threshold probability key: {result_key_prob}, Value: {result_key_prob_value}")
    return result_key_prob_value


if __name__ == '__main__':
    main()
