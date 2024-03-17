import argparse
import os
from PIL import Image
import torch
from diffusers import StableUnCLIPImg2ImgPipeline


class StableUnCLIPImg2Img:
    """
    Wrapper class for Stable UnCLIP image-to-image model.
    """

    def __init__(self, text_prompt=None):
        home_path = os.environ['HOME']
        self.pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(
            f"{home_path}/.cache/huggingface/hub/models--stabilityai--stable-diffusion-2-1-unclip/snapshots/e99f66a92bdcd1b0fb0d4b6a9b81b3b37d8bea44",
            torch_dtype=torch.float16, revision="fp16"
        ).to("cuda")
        self.text_prompt = text_prompt

    def generate(self, img, scale=7, time=1, noise_level=0):
        """
        Generate image-to-image transformations.
        """
        kwargs = dict(
            prompt=self.text_prompt,
            guidance_scale=scale,
            num_inference_steps=20,
            noise_level=noise_level,
            num_images_per_prompt=time
        )
        output = self.pipe(img, **kwargs).images
        return output


def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(
        description='Generate image variations using Stable UnCLIP')
    parser.add_argument('--input_list', required=True,
                        help='Path to the text file with image paths')
    parser.add_argument('--output_prefix', required=True,
                        help='Prefix for the output directory')
    parser.add_argument('--input_prefix', default='',
                        help='Prefix of the input images directory')
    parser.add_argument('--noise_level', type=int, default=0,
                        choices=[0, 100, 200, 300, 400],
                        help='Noise level for the images')
    args = parser.parse_args()
    return args


def main():
    """
    Main function to generate image variations.
    """
    args = parse_args()

    # Read image paths and prepare output directory
    img_paths = [os.path.join(args.input_prefix, line.strip().split()[0]) for
                 line in open(args.input_list)]
    if not os.path.exists(args.output_prefix):
        os.makedirs(args.output_prefix, exist_ok=True)

    # Instantiate the model
    model = StableUnCLIPImg2Img()

    # Process images in batches
    batch_size = 12
    for i in range(0, len(img_paths), batch_size):
        batch_paths = img_paths[i:i + batch_size]
        batch_imgs = [Image.open(img_path).convert('RGB') for img_path in
                      batch_paths]
        for img, img_path in zip(batch_imgs, batch_paths):
            output_path = os.path.join(args.output_prefix,
                                       os.path.basename(img_path))
            output_folder = os.path.dirname(output_path)
            os.makedirs(output_folder, exist_ok=True)

            # Generate and save images
            out_imgs = model.generate(img, noise_level=args.noise_level)
            for idx, out_img in enumerate(out_imgs, start=1):
                out_img_resized = out_img.resize((512, 512))
                out_img_path = output_path.replace('.jpg',
                                                   f'_s-10-t1-l{args.noise_level}_{idx}.jpg')
                out_img_resized.save(out_img_path)
                print(f"Saved: {out_img_path}")


if __name__ == '__main__':
    main()
