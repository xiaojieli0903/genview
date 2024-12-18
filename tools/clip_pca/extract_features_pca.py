import argparse
import matplotlib.pyplot as plt
import numpy as np
import open_clip
import os
import random
import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm


# Set the device to use for computation. Default is 'cuda' for GPU acceleration.
device = 'cuda' if torch.cuda.is_available() else 'cpu'
DEFAULT_IMAGE_SIZE = 224

def resize_image(image, image_size):
    """Resize an image while maintaining aspect ratio."""
    w, h = image.size
    if w < h:
        new_w = image_size
        new_h = int(h * (image_size / w))
    else:
        new_h = image_size
        new_w = int(w * (image_size / h))
    return image.resize((new_w, new_h), Image.ANTIALIAS)


def PCA(data, num_components=1):
    """Performs PCA on the given data and returns the projected data along with eigenvecters and eigenvectors."""
    data = data.float()
    mean = torch.mean(data, dim=0)
    zero_centered_data = data - mean
    cov_matrix = torch.mm(zero_centered_data.t(), zero_centered_data) / (zero_centered_data.size(0) - 1)
    eigenvecters, eigenvectors = torch.linalg.eigh(cov_matrix)
    sorted_indices = torch.argsort(eigenvecters, descending=True)
    eigenvectors = eigenvectors[:, sorted_indices]
    selected_eigenvectors = eigenvectors[:, :num_components]
    projected_data = torch.mm(data, selected_eigenvectors)
    return np.array(projected_data), np.array(selected_eigenvectors), np.array(mean)


def read_image_paths_from_file(image_paths_file, num_extract):
    """
    Read image paths from a file and randomly sample if the number exceeds `num_extract`.

    Args:
        image_paths_file (str): Path to the file containing image paths.
        num_extract (int): Number of paths to extract; -1 means all paths.

    Returns:
        list: List of sampled or all image paths.
    """
    with open(image_paths_file, 'r') as f:
        image_paths = f.read().splitlines()
    if num_extract == -1 or num_extract <= len(image_paths):  # -1 indicates no limit, use all paths
        return image_paths
    else:
        return random.sample(image_paths, num_extract)


def create_output_directory(output_dir):
    """Create necessary directories for saving outputs."""
    directories = ['masks', 'maps', 'features', 'original_images', 'eigenvecters']
    for dir in directories:
        os.makedirs(os.path.join(output_dir, dir), exist_ok=True)


def extract_features(image_paths, output_dir, model, preprocess, patch_size=14, batch_size=1024, num_vis=20, num_extract=-1):
    """
    Extract features from images using the CLIP model.

    Args:
        image_paths (list): List of image file paths.
        output_dir (str): Directory to save extracted features.
        model (torch.nn.Module): CLIP model for feature extraction.
        preprocess (callable): Image preprocessing function for CLIP.
        batch_size (int): Number of images to process in a batch.
        num_vis (int): Number of images to save for visualization.
        num_extract (int): Number of images to extract; -1 means process all available images.

    Returns:
        tuple: (All extracted features, List of resized original images).
    """
    preprocess_no_norm = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    to_pil = transforms.ToPILImage()
    all_patch_features, original_images = [], []

    # Batch processing loop
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Extracting Features"):
        batch_paths = image_paths[i: i + batch_size]
        if num_extract == -1:
            extract_flag = False
            for idx, path in enumerate(batch_paths):
                save_path = os.path.join(output_dir, 'features', f"{os.path.basename(path)}.npy")
                if not os.path.exists(save_path):
                    extract_flag = True
            if not extract_flag:
                continue

        batch_images = []
        for path in batch_paths:
            image = Image.open(path).convert('RGB')
            batch_images.append(preprocess(image).unsqueeze(0).to(device))
            if len(original_images) < num_vis:  # Limit saving for visualization
                image_processed = preprocess_no_norm(image)
                original_images.append(to_pil(image_processed))
        batch_images = torch.cat(batch_images, dim=0)

        # Extract features
        with torch.no_grad(), torch.cuda.amp.autocast():
            _, batch_features = model.encode_image(batch_images)
            batch_features = batch_features.cpu().numpy()
        # Save features
        for idx, path in enumerate(batch_paths):
            save_path = os.path.join(output_dir, 'features', f"{os.path.basename(path)}.npy")
            np.save(save_path, batch_features[idx])
        if num_extract > 0:
            all_patch_features.append(np.concatenate(batch_features, axis=0))

    if num_extract > 0:
        # Combine and reshape all features
        all_patch_features = [af for af in all_patch_features if
                              af is not None]
        all_patch_features = np.concatenate(all_patch_features, axis=0)
        all_patch_features = all_patch_features.reshape(-1,
                                                        224 // patch_size * 224 // patch_size,
                                                        all_patch_features.shape[-1])
    return all_patch_features, original_images


def apply_pca(features, n_components=1):
    """
    Apply Principal Component Analysis (PCA) to reduce feature dimensionality.

    Args:
        features (np.ndarray): Input feature array.
        n_components (int): Number of principal components to retain.

    Returns:
        tuple: (PCA-transformed features, PCA vectors, mean vectors).
    """
    return PCA(torch.from_numpy(features), n_components)


def visualize_pca_histograms(pca_features, output_dir):
    """
    Generate and save histograms and cumulative distribution plots for PCA features.

    Args:
        pca_features (np.ndarray): PCA-transformed features.
        output_dir (str): Directory to save histogram plots.

    Returns:
        tuple: (Value dictionary for cumulative distribution, Probability dictionary).
    """
    os.makedirs(output_dir, exist_ok=True)

    assert pca_features.shape[1] == 1, "PCA features must have a single component."

    pca_feature = pca_features[:, 0]
    value_dict = {}
    prob_dict = {}

    # Plotting histogram
    fig, axs = plt.subplots(3, 1, figsize=(12, 12))
    axs[0].hist(pca_feature, bins=200)
    axs[0].set_title("PCA Histogram")

    # Cumulative Distribution
    values, base = np.histogram(pca_feature, bins=200)
    cumulative = np.cumsum(values) / np.sum(values)
    axs[1].plot(base[:-1], cumulative, c='blue')
    axs[1].set_title("Cumulative Distribution")

    # Inverted CDF
    axs[2].plot(cumulative, base[:-1], c='red')
    axs[2].set_title("Inverse Cumulative Distribution")

    # Store cumulative values and probabilities
    for k in range(base[:-1].shape[0]):
        value_dict[base[:-1][k]] = cumulative[k]
        prob_dict[cumulative[k]] = base[:-1][k]

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'pca_histograms.jpg'))
    plt.close(fig)

    return value_dict, prob_dict


def create_masks(pca_features, fg_thre, output_dir):
    """
    Create binary masks from PCA features based on a threshold.

    Args:
        pca_features (np.ndarray): PCA-transformed features.
        fg_thre (float): Foreground threshold for creating masks.
        output_dir (str): Directory to save mask images.

    Outputs:
        Binary masks are saved as images in the output directory.
    """
    with tqdm(total=pca_features.shape[0], desc="Creating Masks") as pbar:
        for i in range(pca_features.shape[0]):
            for k in range(pca_features.shape[-1]):
                mask = pca_features[i, :, k] > fg_thre
                num_patches = mask.shape[0]
                resized_mask = mask.reshape(int(np.sqrt(num_patches)), -1)
                mask_array = (resized_mask > 0).astype(np.uint8)

                # Save the mask image
                mask_array_resized_img = Image.fromarray((mask_array * 255).astype(np.uint8))
                mask_file = os.path.join(output_dir, 'masks', f'{i}_mask-{k}_thre{fg_thre}.jpg')
                mask_array_resized_img.save(mask_file)

            pbar.update(1)


def create_maps(pca_features, output_dir):
    """
    Generate foreground and background maps from PCA features.

    Args:
        pca_features (np.ndarray): PCA-transformed features.
        output_dir (str): Directory to save map images.

    Outputs:
        Foreground and background maps are saved as images.
    """
    with tqdm(total=pca_features.shape[0], desc="Creating Maps") as pbar:
        for i in range(pca_features.shape[0]):
            for k in range(pca_features.shape[-1]):
                pca_feature = pca_features[i, :, k].copy()
                pca_feature = (pca_feature - pca_feature.min()) / (pca_feature.max() - pca_feature.min())

                # Reshape and save the feature map
                num_patches = pca_feature.shape[0]
                resized_map = pca_feature.reshape(int(np.sqrt(num_patches)), -1)
                resized_map_img = Image.fromarray((resized_map * 255).astype(np.uint8))
                map_file = os.path.join(output_dir, 'maps', f'{i}_fg_map-{k}.jpg')
                resized_map_img.save(map_file)

                # Background map
                resized_map_bg = 1 - resized_map
                resized_map_bg_img = Image.fromarray((resized_map_bg * 255).astype(np.uint8))
                map_file_bg = os.path.join(output_dir, 'maps', f'{i}_bg_map-{k}.jpg')
                resized_map_bg_img.save(map_file_bg)

            pbar.update(1)


def save_original_images(original_images, output_dir):
    """
    Save resized original images for visualization.

    Args:
        original_images (list): List of original images (PIL format).
        output_dir (str): Directory to save resized images.

    Outputs:
        Images are saved in the 'original_images' directory.
    """
    with tqdm(total=len(original_images), desc="Saving Original Images") as pbar:
        for i, image in enumerate(original_images):
            image_file = os.path.join(output_dir, 'original_images', f'original_image_{i}.jpg')

            # Resize and save the image
            image_resized = resize_image(image, DEFAULT_IMAGE_SIZE)
            image_resized.save(image_file)
            pbar.update(1)


def determine_threshold_from_histogram(prob_dict, default_threshold):
    """
    Determine threshold for mask generation from cumulative distribution values.

    Args:
        prob_dict (dict): Cumulative distribution of PCA feature values.
        default_threshold (float): Default threshold value if no suitable value is found.

    Returns:
        float: Selected threshold value.
    """
    if default_threshold is not None:
        return default_threshold

    # If default threshold is not provided, we choose based on empirical cumulative distribution.
    for key in prob_dict.keys():
        if key >= 0.65:  # You can adjust this threshold based on experiments
            result_key_prob = key
            result_key_prob_value = prob_dict[key]
            print(f"Selected threshold probability key: {result_key_prob}, Value: {result_key_prob_value}")
            return result_key_prob_value

    # Default return if no suitable threshold found
    print(f"Using default threshold: {default_threshold}")
    return default_threshold


def main():
    """
    Main function to execute the CLIP feature extraction and PCA analysis pipeline.

    Workflow:
    1. Parse input arguments and load the CLIP model.
    2. Read image paths and extract features in batches.
    3. Apply PCA to extracted features and save PCA components.
    4. Visualize PCA histograms and determine threshold values.
    5. Generate binary masks and feature maps based on PCA features.
    6. Save a subset of resized original images for visualization.
    """
    parser = argparse.ArgumentParser(description='CLIP Feature Extraction and PCA Analysis')
    parser.add_argument('--input-list', type=str, required=True, help='Path to the file with image paths')
    parser.add_argument('--output-dir', type=str, default='tools/clip_pca/pca_results/', help='Directory for saving results')
    parser.add_argument('--batch-size', type=int, default=1024, help='Number of images to process per batch')
    parser.add_argument('--model', type=str, default='ViT-H-14', help='Name of the CLIP model to use')
    parser.add_argument('--training-data', type=str, default='laion2b_s32b_b79k', help='Pretrained model weights to use')
    parser.add_argument('--patch-size', type=int, default=16, help='Patch size for dividing images into patches')
    parser.add_argument('--image-size', type=int, default=224, help='Target size for resizing the shorter edge of images')
    parser.add_argument('--num-vis', type=int, default=10, help='Number of images for visualization')
    parser.add_argument('--num-extract', type=int, default=10000, help='Number of images for feature extraction')
    parser.add_argument('--fg-thre', type=float, default=None, help='Threshold for creating foreground masks based on PCA')
    args = parser.parse_args()

    model, _, preprocess = open_clip.create_model_and_transforms(args.model, pretrained=args.training_data,
                                                                 device=device)
    print(f'Loading model: {args.model} with pretrained weights: {args.training_data}. Preprocess: {preprocess}')

    image_paths = read_image_paths_from_file(args.input_list, args.num_extract)
    args.output_dir = os.path.join(args.output_dir, f'{args.model}-{args.training_data}')
    create_output_directory(args.output_dir)
    if args.num_extract == -1:
        print("Extracting features from all available images...")

    # Feature extraction
    features, original_images = extract_features(image_paths, args.output_dir, model, preprocess,
                                                 batch_size=args.batch_size, patch_size=args.patch_size,
                                                 num_vis=args.num_vis, num_extract=args.num_extract)
    if args.num_extract == -1:
        print('Feature extraction complete.')
        return

    # Convert to numpy if not already
    features = np.array(features)
    print(f"Extracted features shape: {features.shape}")

    # Apply PCA to the features
    N, n_patches, feat_dim = features.shape
    pca_features, pca_vectors, mean_vectors = apply_pca(features.reshape(-1, feat_dim), n_components=1)

    # Save PCA vectors and mean vectors
    os.makedirs(os.path.join(args.output_dir, 'eigenvecters'), exist_ok=True)
    np.save(os.path.join(args.output_dir, 'eigenvecters', 'pca_vectors.npy'), pca_vectors)
    np.save(os.path.join(args.output_dir, 'eigenvecters', 'mean_vectors.npy'), mean_vectors)
    print('PCA vectors and mean vectors saved.')

    # Visualize PCA histograms and determine threshold
    value_dict, prob_dict = visualize_pca_histograms(pca_features, args.output_dir)
    fg_thre = determine_threshold_from_histogram(prob_dict, args.fg_thre)

    # Create masks and maps
    create_masks(pca_features.reshape(N, n_patches, -1)[:args.num_vis], fg_thre, args.output_dir)
    create_maps(pca_features.reshape(N, n_patches, -1)[:args.num_vis], args.output_dir)

    # Save a subset of the original images for reference
    save_original_images(original_images, args.output_dir)

    print('Feature extraction and PCA analysis complete.')


if __name__ == '__main__':
    main()
