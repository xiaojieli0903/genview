# CLIP Feature Extraction and PCA Analysis

This Python script extracts features from images using a specified CLIP model and performs Principal Component Analysis (PCA) on these features. It supports processing images in batches, applying PCA, visualizing the results, and creating masks and maps based on PCA features to identify significant areas in images.


## Usage

The script is executed from the command line and includes several options to customize the processing. Below are the available command-line arguments:

- `--input-list`: Path to the text file containing paths of images to process.
- `--output-dir`: Directory where the output will be saved.
- `--batch-size`: Number of images to process in a batch. Default is 1024.
- `--save`: If set, extracted features will be saved to disk.
- `--model-name`: Name of the CLIP model to use for feature extraction. Default is 'ViT-B-16'.
- `--pretrained-weights`: Pretrained weights for the model. Default is 'laion2B-s34B-b88K'.
- `--patch-size`: Patch size for the model. Default is 16.
- `--smaller-edge-size`: Size of the smaller edge for image resizing. Default is 224.
- `--num`: Number of example images for PCA visualization and mask/map creation. Default is 10.
- `--background-threshold`: Background threshold for creating masks. Optional; Default is None.

## Running the Script

Navigate to the script's directory and use the following command to run the script with your desired options:

```sh
python extract_features_pca.py --input-list path/to/image_list.txt --output-dir path/to/output
```
Running this script generates several outputs for analyzing and visualizing image features:

- **Extracted Features**: Extracts features from images using the specified CLIP model and saves them for further analysis.
  - Features Directory: `features/` containing NumPy files with extracted features for each image, saved when `--save` is used.

- **PCA Results**: Applies PCA on the extracted features to reduce dimensionality and identify patterns.
  - PCA Vectors and Mean Vectors: Stored in `eigenvalues/`, including `pca_vectors.npy` and `mean_vectors.npy`, representing principal components and the mean of original features, respectively.

- **Visualization Outputs**: Generates histograms and cumulative distribution plots for PCA features.
  - PCA Histograms: A figure (`full_figure.png`) in the output directory showing histograms for data distribution along principal components.

- **Masks and Maps**: Based on PCA features, creates masks and maps highlighting significant areas in the images.
  - Masks: Binary images in `masks/` indicating significant variance areas per image.
  - Maps: Images in `maps/` displaying spatial distribution of PCA feature values, including foreground and background maps.

- **Original Images**
  - Subset of Original Images: Up to the specified `--num` of resized original images in `original_images/` for reference.