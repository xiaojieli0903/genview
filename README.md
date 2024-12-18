# GenView: Enhancing View Quality with Pretrained Generative Model for Self-Supervised Learning (ECCV 2024)

![GenView Framework](figs/framework.png)

This repository contains the official implementation of **GenView: Enhancing View Quality with Pretrained Generative Models for Self-Supervised Learning**, presented at ECCV 2024.

> **[GenView: Enhancing View Quality with Pretrained Generative Model for Self-Supervised Learning](https://arxiv.org/abs/2403.12003)**<br> 
> [Xiaojie Li](https://xiaojieli0903.github.io/)^1,2, [Yibo Yang](https://iboing.github.io/)^3, [Xiangtai Li](https://lxtgh.github.io/)^4, [Jianlong Wu](https://jlwu1992.github.io)^1, [Yue Yu](https://yuyue.github.io)^2, [Bernard Ghanem](https://www.bernardghanem.com/)^3, [Min Zhang](https://zhangminsuda.github.io)^1<br>
> ^1Harbin Institute of Technology (Shenzhen), ^2Peng Cheng Laboratory, ^3King Abdullah University of Science and Technology, ^4Nanyang Technological University

## 🔨 Installation

Follow the steps below to set up the environment and install dependencies.

### Step 1: Create and Activate a Conda Environment

Create a new Conda environment with Python 3.8 and activate it:

```bash
conda create --name env_genview python=3.8 -y
conda activate env_genview
```

### Step 2: Install Required Packages

You can install PyTorch, torchvision, and other dependencies via pip or Conda. Choose the command based on your preference and GPU compatibility.

```bash
# Using pip
pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu117

# Or using conda
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia

# Install additional dependencies
pip install timm==0.9.7 open_clip_torch==2.22.0 diffusers==0.21.4 huggingface_hub==0.17.3 transformers==4.33.3
```

### Step 3: Clone the Repository and Install Project Dependencies

Clone the GenView repository and install the required dependencies using `openmim`:

```bash
git clone https://github.com/xiaojieli0903/genview.git
cd genview
pip install -U openmim
mim install -e .
```

Apply modifications to `open_clip` and `timm`:

```bash
sh tools/toolbox_genview/change_openclip_timm.sh
```

## 📷 Adaptive View Generation

We utilize the pretrained **CLIP ViT-H/14** backbone, which serves as the conditional image encoder in **Stable UnCLIP v2-1**, to determine the proportion of foreground content before image generation. This backbone processes an input resolution of \(224 \times 224\) and generates 256 tokens, each with a dimension of 1280. 

For calculating PCA features needed for foreground-background separation, we randomly sample 10,000 images from the original dataset. The threshold \( \alpha \) in Equation (7) is selected to ensure that foreground tokens account for approximately 40% of the total tokens, providing a clear separation between foreground and background.

### **Step 1: Extract CLIP Image Features and Compute PCA**

We first extract features from 10,000 images using the CLIP ViT-H/14 backbone and then perform PCA analysis.
The calculated PCA vectors act as classifiers for distinguishing between foreground and background regions.

**Command to Extract Features and Perform PCA Analysis:**

```shell
python tools/clip_pca/extract_features_pca.py \
    --input-list tools/clip_pca/train_sampled_1000cls_10img.txt \
    --num-extract 10000 \
    --patch-size 14 \
    --num-vis 20 \
    --model ViT-H-14 \
    --training-data laion2b_s32b_b79k
```

- `--input-list`: Path to the file containing the list of sampled images (`tools/clip_pca/train_sampled_1000cls_10img.txt`).
- `--num-extract 10000`: Specifies the number of images to process.
- `--patch-size 14`: Patch size used by the model.
- `--num-vis 20`: Number of images to visualize.
- `--model ViT-H-14`: Specifies the CLIP model to use.
- `--training-data laion2b_s32b_b79k`: Pretrained weights for the model.

**Outputs:**

- **Extracted Features**: Saved in the `features/` directory.
- **PCA Eigenvectors**: Saved in the `eigenvecters/` directory.
- **Generated Masks, Maps, and Original Images**: Saved in the `masks/`, `maps/`, and `original_images/` directories, respectively.
- **Threshold for Foreground-Background Separation**: During the PCA analysis, a background threshold is also calculated and used for generating masks. This threshold helps to separate foreground from background regions by comparing the PCA-transformed feature values with the threshold. The resulting masks can then be used to compute the foreground ratio for each image in the next steps.

### **Step 2: Determine Suitable Noise Levels for Each Image**

To maintain semantic consistency while ensuring diversity, we determine appropriate noise levels for each image using the PCA vectors and the extracted image features.

#### **2.1 Extract Features for the Entire ImageNet Dataset**

First, we need to extract features for each image in the ImageNet dataset. This process may take around 4 hours with a batch size of 1024, and the extracted features will require approximately 4GB of storage.

**Command to Extract Features:**

```shell
python tools/clip_pca/extract_features_pca.py \
    --input-list data/imagenet/train.txt \
    --num-extract -1 \
    --patch-size 14 \
    --num-vis 20 \
    --model ViT-H-14 \
    --training-data laion2b_s32b_b79k
```

- `--input-list`: Path to the file containing the list of all training images.
- `--num-extract -1`: Processes all images in the list (no limit).
- Other parameters are the same as in Step 1.

#### **2.2 Calculate Foreground Ratios**

Using the previously computed PCA vectors and the foreground-background threshold (`fg_thre`), we calculate the foreground ratio (`fg_ratio`) for each image in the dataset. The `fg_ratio` helps quantify the proportion of foreground content within each image, which will later guide noise level determination for adaptive view generation.

**Command to Calculate `fg_ratio`**:

```shell
python tools/clip_pca/calculate_fgratio.py \
    --input-dir tools/clip_pca/pca_results/ViT-H-14-laion2b_s32b_b79k/ \
    --input-list data/imagenet/train.txt \
    --output-dir data/imagenet/ \
    --fg-thre {computed_threshold}
```
- `--input-dir`: Path to the directory to save extracted features and PCA eigenvecters.
- `--input-list`: Path to the file containing the list of all training images.
- `--output-dir`: Directory where the `fg_ratios.txt` file will be saved.
- `--fg-thre {computed_threshold}`: The foreground-background threshold value (`fg_thre`) calculated from **Step 1** using PCA analysis. This threshold ensures the proper separation of foreground and background regions.

A file named `fg_ratios.txt` will be generated in the specified output directory. This file contains a list of image paths paired with their respective `fg_ratio` values.  
Each line of `fg_ratios.txt` is structured as:  
    ```
    <image_path> <fg_ratio>
    ```
    Example:  
    ```
    data/imagenet/train/img_0001.jpg 0.42
    data/imagenet/train/img_0002.jpg 0.38
    ```
  
#### **2.3 Generate Adaptive Noise Levels**

Finally, we distribute the original `fg_ratios.txt` entries into separate files based on specified ranges and mapping values. Each output file is named after its corresponding mapped noise level value (e.g., `fg_ratios_0.txt`, `fg_ratios_100.txt`, etc.), containing image paths and their `fg_ratio` values that fall into the respective ranges.

**Command to Generate Noise Level Files:**

```shell
python tools/clip_pca/generate_ada_noise_level.py \
    --input-file data/imagenet/fg_ratios.txt \
    --output-dir data/imagenet/
```

- `--input-file`: Path to the `fg_ratios.txt` generated in the previous step.
- `--output-dir`: Directory where the noise level files will be saved.

These files categorize images based on their foreground ratios, allowing us to assign appropriate noise levels during image generation to achieve the desired balance between semantic consistency and diversity.
    ```

### **Step 3: Generate Image Dataset Variations**

In this step, we generate image variations for the dataset by applying the calculated noise levels. This ensures the generated data maintains semantic consistency while introducing controlled diversity for adaptive view generation.

#### **3.1 Generate Image Variations with Specified Noise Levels**

For each noise level file (e.g., `fg_ratios_*.txt`), use the following command to generate image variations:

```bash
python tools/toolbox_genview/generate_image_variations_noiselevels.py \
    --input-list data/imagenet/fg_ratios_{noise_level}.txt \
    --output-prefix data/imagenet/train_variations/ \
    --noise-level {noise_level}
```

- `--input-list`: Path to the text file that contains image paths and `fg_ratio` values.
- `--output-prefix`: Prefix for the output directory where the variations will be saved.
- `--noise-level`: Noise level to apply to the image variations (options: 0, 100, 200, 300, 400).

Repeat this command for all `fg_ratios_*.txt` files to generate the complete set of image variations.

Use the following shell script to parallelize the image generation process. This script splits the input list into multiple parts and processes them in parallel.

```bash
bash tools/toolbox_genview/run_parallel_levels.sh data/imagenet/fg_ratios_{noise_level}.txt data/imagenet/train_variations/ {noise_level}
```

Here’s the revised content in English:

---

#### **3.2 Quick Data Preparation: Use Pre-generated Image Variations**

To simplify the data preparation process, pre-generated image variations are available for download. You can access them on [https://huggingface.co/datasets/Xiaojie0903/Genview_syntheric_dataset_in1k](https://huggingface.co/datasets/Xiaojie0903/Genview_syntheric_dataset_in1k) and use them directly for your experiments and model training.

1. **Download and Merge the Dataset**:

   After downloading all parts of the compressed dataset, merge them into a single file and extract the contents:

   ```bash
   cd /path/to/download_tars/
   cat train_variations.tar.* > train_variations.tar
   tar -xvf train_variations.tar
   ```

2. **Create Symbolic Links**:

   To simplify access to the extracted data, create symbolic links in the `genview` project directory:

   ```bash
   cd genview
   mkdir -p data/imagenet
   cd data/imagenet
   ln -s /path/to/imagenet/train .
   ln -s /path/to/imagenet/val .
   ln -s /path/to/download_tars/train_variations/ .
   ```

   - `train/`: Link to the original ImageNet training data.
   - `val/`: Link to the ImageNet validation data.
   - `train_variations/`: Link to the directory containing pre-generated image variations.

#### **3.3 Generate the Synthetic Image List**

Once the image variations are prepared, generate a list of all synthetic images for further training and evaluation:

```bash
python tools/toolbox_genview/generate_train_variations_list.py \
    --input-dir data/imagenet/train_variations \
    --output-list data/imagenet/train_variations.txt
```

- `--input-dir`: Path to the directory containing the generated image variations.
- `--output-list`: Path to save the generated image list.

**Outputs:**
- **Image Variations**: Saved in the `train_variations/` directory, with noise applied according to the `fg_ratios_*.txt` files.
- **Synthetic Image List**: A text file (`train_variations.txt`) containing paths to all generated image variations, saved in `data/imagenet/`.

By completing this step, you will have a comprehensive dataset containing controlled image variations ready for self-supervised training with enhanced view quality.

## 🔍 Quality-Driven Contrastive Loss

We use the pretrained CLIP ConvNext-Base model as the encoder to extract feature maps from augmented positive views. These feature maps, with a resolution of 7² from a 224² input, are used to calculate foreground and background attention maps based on PCA.

We randomly sample 10,000 images to compute PCA features. The threshold \( \alpha \) ensures that 40% of the tokens represent the foreground, enabling clear separation.

Use the following command to extract features and compute PCA:

```bash
python tools/clip_pca/extract_features_pca.py \
    --input-list tools/clip_pca/train_sampled_1000cls_10img.txt \
    --num-extract 10000 \
    --patch-size 32 \
    --num-vis 20 \
    --model convnext_base_w \
    --training-data laion2b-s13b-b82k-augreg
```

### Outputs

- **Extracted Features**: Stored in `features/`.
- **PCA Eigenvectors**: Stored in `eigenvectors/`.
- **Masks, Maps, and Original Images**: Stored in `masks/`, `maps/`, and `original_images/`.

These PCA vectors are used to generate foreground and background attention maps during pretraining. We provide precomputed PCA vectors, which can be found at `tools/clip_pca/pca_results/convnext_base_w_laion2b-s13k-b82k-augreg/eigenvectors/pca_vectors.npy`

## 🔄 Training

Detailed commands for running pretraining and downstream tasks with single or multiple machines/GPUs:

**Training with Multiple GPUs**
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=29500 bash tools/dist_train.sh ${CONFIG_FILE} 8 [PY_ARGS] [--resume /path/to/latest/epoch_{number}.pth]
```

**Training with Multiple Machines**
```shell
CPUS_PER_TASK=8 GPUS_PER_NODE=8 GPUS=16 sh tools/slurm_train.sh $PARTITION $JOBNAME ${CONFIG_FILE} $WORK_DIR [--resume /path/to/latest/epoch_{number}.pth]
```

Ensure to replace `$PARTITION`, `$JOBNAME`, and `$WORK_DIR` with actual values for your setup.

## 🚀 Experiment Configurations

The following experiments provide various pretraining setups using different architectures, epochs, and GPU configurations.

**SimSiam + ResNet50 + 200 Epochs + 8 GPUs**

- **Pretraining**:
  ```shell
  CPUS_PER_TASK=8 GPUS_PER_NODE=8 GPUS=8 sh tools/slurm_train.sh $PARTITION simsiam_pretrain configs_genview/simsiam/simsiam_resnet50_8xb32-coslr-200e_in1k_singleview_clipmask.py work_dirs/simsiam_resnet50_8xb32-coslr-200e_in1k_singleview_clipmask
  ```
- **Linear Probe**:
  ```shell
  CPUS_PER_TASK=8 GPUS_PER_NODE=8 GPUS=8 sh tools/slurm_train.sh $PARTITION simsiam_linear configs_genview/simsiam/benchmarks/resnet50_8xb512-linear-coslr-90e_in1k_clip.py work_dirs/simsiam_resnet50_8xb32-coslr-200e_in1k_diffssl_prob1_128w_clipmask/linear --cfg-options model.backbone.init_cfg.checkpoint=work_dirs/simsiam_resnet50_8xb32-coslr-200e_in1k_diffssl_prob1_128w_clipmask/epoch_200.pth
  ```
  
**MoCo v3 + ResNet50 + 100 Epochs + 8 GPUs**

- **Pretraining**:
  ```shell
  CPUS_PER_TASK=8 GPUS_PER_NODE=8 GPUS=8 sh tools/slurm_train.sh $PARTITION mocov3r50_pretrain configs_genview/mocov3/mocov3_resnet50_8xb512-amp-coslr-100e_in1k_singleview_clipmask.py work_dirs/mocov3_resnet50_8xb512-amp-coslr-100e_in1k_singleview_clipmask
  ```
  
- **Linear Probe**:
  ```shell
  CPUS_PER_TASK=8 GPUS_PER_NODE=8 GPUS=8 sh tools/slurm_train.sh $PARTITION mocov3r50_linear configs_genview/mocov3/benchmarks/resnet50_8xb128-linear-coslr-90e_in1k_clip.py work_dirs/mocov3_resnet50_8xb512-amp-coslr-100e_in1k_singleview_clipmask/linear --cfg-options model.backbone.init_cfg.checkpoint=work_dirs/mocov3_resnet50_8xb512-amp-coslr-100e_in1k_singleview_clipmask/epoch_100.pth
  ```

**MoCo v3 + ViT-B + 300 Epochs + 16 GPUs**

- **Pretraining**:
  ```shell
  CPUS_PER_TASK=8 GPUS_PER_NODE=8 GPUS=16 sh tools/slurm_train.sh $PARTITION mocov3vit_pretrain configs_genview/mocov3/mocov3_vit-base-p16_16xb256-amp-coslr-300e_in1k_singleview_clipmask.py work_dirs/mocov3_vit-base-p16_16xb256-amp-coslr-300e_in1k_singleview_clipmask
  ```

- **Linear Probe**:
  ```shell
  CPUS_PER_TASK=8 GPUS_PER_NODE=8 GPUS=8 sh tools/slurm_train.sh $PARTITION mocov3vit_linear configs_genview/mocov3/benchmarks/vit-base-p16_8xb128-linear-coslr-90e_in1k_clip.py work_dirs/mocov3_vit-base-p16_16xb256-amp-coslr-300e_in1k_singleview_clipmask/linear --cfg-options model.backbone.init_cfg.checkpoint=work_dirs/mocov3_vit-base-p16_16xb256-amp-coslr-300e_in1k_singleview_clipmask/epoch_300.pth
  ```
  
## 📍 Model Zoo  

We have uploaded the pre-trained models to [https://huggingface.co/Xiaojie0903/genview_pretrained_models](https://huggingface.co/Xiaojie0903/genview_pretrained_models). Access them directly using the links below:

| **Method**        | **Backbone** | **Pretraining Epochs** | **Linear Probe Accuracy (%)** | **Model Link**                                                              |
|-------------------|--------------|-------------------------|-------------------------------|-----------------------------------------------------------------------------|
| MoCo v2 + GenView | ResNet-50    | 200                     | 70.0                          | [Download](https://huggingface.co/Xiaojie0903/genview_pretrained_models/resolve/main/mocov2_resnet50_8xb32-coslr-200e_in1k_genview.pth) |
| SwAV + GenView    | ResNet-50    | 200                     | 71.7                          | [Download](https://huggingface.co/Xiaojie0903/genview_pretrained_models/resolve/main/swav_resnet50_8xb32-mcrop-coslr-200e_in1k_genview.pth) |
| SimSiam + GenView | ResNet-50    | 200                     | 72.2                          | [Download](https://huggingface.co/Xiaojie0903/genview_pretrained_models/resolve/main/simsiam_resnet50_8xb32-coslr-200e_in1k_genview.pth) |
| BYOL + GenView    | ResNet-50    | 200                     | 73.2                          | [Download](https://huggingface.co/Xiaojie0903/genview_pretrained_models/resolve/main/byol_resnet50_16xb256-coslr-200e_in1k_genview.pth) |
| MoCo v3 + GenView | ResNet-50    | 100                     | 72.7                          | [Download](https://huggingface.co/Xiaojie0903/genview_pretrained_models/resolve/main/mocov3_resnet50_8xb512-amp-coslr-100e_in1k_genview.pth) |
| MoCo v3 + GenView | ResNet-50    | 300                     | 74.8                          | [Download](https://huggingface.co/Xiaojie0903/genview_pretrained_models/resolve/main/mocov3_resnet50_8xb512-amp-coslr-300e_in1k_genview.pth) |
| MoCo v3 + GenView | ViT-S        | 300                     | 74.5                          | [Download](https://huggingface.co/Xiaojie0903/genview_pretrained_models/resolve/main/mocov3_vit-small-p16_16xb256-amp-coslr-300e_in1k_genview.pth) |
| MoCo v3 + GenView | ViT-B        | 300                     | 77.8                          | [Download](https://huggingface.co/Xiaojie0903/genview_pretrained_models/resolve/main/mocov3_vit-base-p16_16xb256-amp-coslr-300e_in1k_genview.pth) |

## ✏️ Citation
If you find the repo useful for your research, please consider citing our paper:
```bibtex
@inproceedings{li2024genview,
  author={Li, Xiaojie and Yang, Yibo and Li, Xiangtai and Wu, Jianlong and Yu, Yue and Ghanem, Bernard and Zhang, Min},
  title={GenView: Enhancing View Quality with Pretrained Generative Model for Self-Supervised Learning}, 
  year={2024},
  pages={306--325},
  booktitle={Proceedings of the European Conference on Computer Vision},
  publisher="Springer"
}
```

## 👍 Acknowledgments

This codebase builds on [mmpretrain](https://github.com/open-mmlab/mmpretrain). Thanks to the contributors of this great codebase.
