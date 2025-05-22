import PIL.Image
import numpy as np

import torch
import torch.nn.functional as F

from PIL import ImageDraw, ImageFont

from diffusers import StableDiffusionXLPipeline, AutoencoderTiny
from diffusers.utils import load_image, make_image_grid

from streamdiffusion import StreamDiffusion
from streamdiffusion.image_utils import postprocess_image



def pre_process_image(image: PIL.Image.Image, height, width, for_sd=True):

    # Convert to tensor (values 0-255), keeping HWC format
    image_pt = torch.from_numpy(np.array(image))

    # Move to device first
    image_pt = image_pt.to(device="mps", dtype=torch.float16)

    # adds the "batch" dimension to make shape (B, H, W, C)
    image_pt = image_pt.unsqueeze(0)

    # Do permute on GPU (BHWC → BCHW)
    image_pt = image_pt.permute(0, 3, 1, 2)

    # resize
    if image_pt.shape[2] != height or image_pt.shape[3] != width:
        print(f'Resizing Image! size={image_pt.shape[3]}x{image_pt.shape[2]}, should be: {width}x{height}')
        image_pt = F.interpolate(image_pt, size=(height, width), mode="bilinear", align_corners=False)

    # Scale to 0-1 range (PyTorch standard)
    image_pt = image_pt / 255.0

    # Normalize to -1-1 range (StableDiffusion standard)
    if for_sd:
        image_pt = image_pt * 2 - 1

    return image_pt


def add_label(image, text):
    draw = ImageDraw.Draw(image)
    font_size = 40
    try:
        font = ImageFont.truetype("Arial", font_size)
    except IOError:
        font = ImageFont.load_default()

    # Add white background for text
    text_width, text_height = draw.textbbox((0, 0), text, font=font)[2:4]
    draw.rectangle([(0, 0), (text_width + 10, text_height + 10)], fill=(255, 255, 255))
    draw.text((5, 5), text, fill=(0, 0, 0), font=font)
    return image


def test_t2i_adapter_controls():
    device = 'mps'
    dtype = torch.float16

    # Load pipeline
    print('Loading Pipeline')
    tiny_vae = AutoencoderTiny.from_pretrained("madebyollin/taesdxl")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        'stabilityai/sdxl-turbo',
        vae=tiny_vae,
        torch_dtype=dtype,
        variant='fp16'
    ).to(device, dtype=dtype)

    # Create StreamDiffusion with single inference step
    stream = StreamDiffusion(
        pipe,
        device=device,
        t_index_list=[0],
        original_inference_steps=100,
        torch_dtype=dtype,
        do_add_noise=True,
        height=512,
        width=512,
        frame_buffer_size=1
    )
    # Load control adapters
    print('Loading Control Adapters')
    stream.load_control_adapter()

    prompts = ["batman"]  # ["cyberpunk squirrel", "an oil painting"]
    stream.update_prompt(prompts[0])

    # Load test image
    print('Loading Test Image')
    input_image = load_image("/Users/himmelroman/Desktop/images/clean.png").resize((512, 512))
    input_tensor = pre_process_image(input_image, 512, 512, for_sd=True)

    # Test different control strengths
    timesteps = [5, 15, 25, 35]
    control_scales = [0.0]

    all_images = []
    for ts in timesteps:
        for cs in control_scales:
            print(f'Testing: {cs=} {ts=}')

            stream.set_timesteps([ts])
            stream.set_noise(seed=42)
            stream.control_scale = cs

            # Test with control
            print(f'Generating with control...')
            img_pt = stream(
                x=input_tensor,
                control=input_image,  # Use same image for control extraction
                encode_input=True,
                decode_output=True
            )

            img_pil = postprocess_image(img_pt)[0]
            img_pil = add_label(img_pil, f'{prompts[0][:10]} / cs={cs}, ts={ts}')
            all_images.append(img_pil)

    # Create and save grid
    grid = make_image_grid(all_images, cols=len(control_scales), rows=len(timesteps))
    grid.save('t2i_adapter_test_grid.jpg')
    grid.show()

    print("Test completed! Check 't2i_adapter_test_grid.jpg'")

if __name__ == "__main__":

    # Then run comprehensive test
    test_t2i_adapter_controls()
