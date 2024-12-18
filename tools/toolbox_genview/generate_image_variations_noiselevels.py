import argparse
import os
from PIL import Image
import torch
from diffusers import StableUnCLIPImg2ImgPipeline


class StableUnCLIPImg2Img:
    """
    Wrapper class for Stable UnCLIP image-to-image model.
    """

    def __init__(self, model_path=None, text_prompt=None):
        """
        Initialize the StableUnCLIP model.
        """
        home_path = os.environ.get('HOME', '/root')
        self.pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(
            model_path or f"{home_path}/.cache/huggingface/hub/models--stabilityai--stable-diffusion-2-1-unclip/snapshots/e99f66a92bdcd1b0fb0d4b6a9b81b3b37d8bea44",
            torch_dtype=torch.float16, revision="fp16"
        ).to("cuda")
        self.text_prompt = text_prompt

    def generate(self, img, scale=10, num_inference_steps=20, noise_level=0, num_images_per_prompt=1):
        """
        Generate image-to-image transformations.
        """
        kwargs = {
            'prompt': self.text_prompt,
            'guidance_scale': scale,
            'num_inference_steps': num_inference_steps,
            'noise_level': noise_level,
            'num_images_per_prompt': num_images_per_prompt
        }
        output = self.pipe(img, **kwargs).images
        return output


def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(
        description='Generate image variations using Stable UnCLIP')
    parser.add_argument('--input-list', required=True,
                        help='Path to the text file with image paths')
    parser.add_argument('--output-prefix', required=True,
                        help='Prefix for the output directory')
    parser.add_argument('--input-prefix', default='',
                        help='Prefix of the input images directory')
    parser.add_argument('--noise-level', type=int, default=0,
                        choices=[0, 100, 200, 300, 400],
                        help='Noise level for the images')
    parser.add_argument('--batch-size', type=int, default=12,
                        help='Batch size for image processing')
    parser.add_argument('--num-inference-steps', type=int, default=20,
                        help='Number of inference steps')
    parser.add_argument('--scale', type=float, default=10,
                        help='Guidance scale for image generation')
    parser.add_argument('--num-images-per-prompt', type=int, default=1,
                        help='Number of images to generate per prompt')
    args = parser.parse_args()
    return args


def process_images(args):
    """
    Process images in batches and generate variations.
    """
    # Read image paths
    img_paths = [os.path.join(args.input_prefix, line.strip().split()[0]) for line in open(args.input_list)]

    # Create output directory if it does not exist
    os.makedirs(args.output_prefix, exist_ok=True)

    # Initialize the model
    model = StableUnCLIPImg2Img(text_prompt="Your custom text prompt here")

    # Process images in batches
    for i in range(0, len(img_paths), args.batch_size):
        batch_paths = img_paths[i:i + args.batch_size]
        batch_imgs = [Image.open(img_path).convert('RGB') for img_path in batch_paths]

        for img, img_path in zip(batch_imgs, batch_paths):
            output_path = os.path.join(args.output_prefix, os.path.basename(img_path))
            output_folder = os.path.dirname(output_path)
            os.makedirs(output_folder, exist_ok=True)

            # Generate and save images
            out_imgs = model.generate(img, noise_level=args.noise_level, num_inference_steps=args.num_inference_steps,
                                      scale=args.scale, num_images_per_prompt=args.num_images_per_prompt)

            for idx, out_img in enumerate(out_imgs, start=1):
                out_img_resized = out_img.resize((512, 512))
                out_img_resized.save(output_path)
                print(f"Saved: {output_path}")


def main():
    """
    Main function to run image generation.
    """
    args = parse_args()
    process_images(args)


if __name__ == '__main__':
    main()
