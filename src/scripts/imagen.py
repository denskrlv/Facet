import os
import argparse
import torch

from facet import Facet
from PIL import Image


compositional_prompts = [
    "a man AND NOT Smiling AND NOT Glasses",
    "NOT a woman AND Smiling AND NOT Glasses",
    "a man AND Smiling AND Glasses",
    "NOT a man AND NOT Smiling AND NOT Glasses",
    "a woman AND Smiling AND NOT Glasses",
    "a woman AND Smiling AND Glasses"
]

standard_prompts = [
    "a man",
    "a woman",
    "smiling",
    "not smiling",
    "wearing glasses",
    "not wearing glasses"
]

NUM_VARIANTS = 20


def tensor_to_image(tensor):
    """Convert a PyTorch tensor to a PIL Image."""
    # Scale from [-1, 1] to [0, 255]
    scaled = ((tensor + 1) * 127.5).round().clamp(0, 255).to(torch.uint8).cpu()
    
    # Rearrange dimensions from CxHxW to HxWxC
    if scaled.dim() == 3:  # Single image
        img = scaled.permute(1, 2, 0).numpy()
    else:  # Batch of images
        img = scaled[0].permute(1, 2, 0).numpy()  # Take the first image
        
    return Image.fromarray(img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate face images with the fine-tuned GLIDE model")
    parser.add_argument("--output_dir", type=str, default=os.path.join("outputs", "standard", "1"),
                        help="Directory to save generated images")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    facet = Facet(model_name='glide_faces', verbose=True)
    print(facet)

    # for i, prompt in enumerate(compositional_prompts):
    #     for j in range(NUM_VARIANTS):

    #         print(f"Generating image for prompt {i} variant {j}")

    #         result, _ = facet.generate(
    #             prompt, 
    #             num_images=1, 
    #             upsample=True, 
    #             upsample_temp=0.995,
    #             save_intermediate_steps=10,
    #             return_attention_maps=True
    #         )

    #         image = tensor_to_image(result)
    #         image_path = os.path.join(args.output_dir, f"prompt_{i}_variant_{j}.png")
    #         image.save(image_path)
    #         print(f"Saved: {image_path}!")
    
    for i, prompt in enumerate(standard_prompts):
        for j in range(NUM_VARIANTS):

            print(f"Generating image for prompt {i} variant {j}")

            result, _ = facet.generate(
                prompt, 
                num_images=1, 
                upsample=True, 
                upsample_temp=0.995,
                save_intermediate_steps=10,
                return_attention_maps=True
            )

            image = tensor_to_image(result)
            image_path = os.path.join(args.output_dir, f"prompt_{i}_variant_{j}.png")
            image.save(image_path)
            print(f"Saved: {image_path}!")
