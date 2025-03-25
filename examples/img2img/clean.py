import torch
from PIL import Image
from PIL import  ImageDraw, ImageFont

from diffusers import StableDiffusionXLPipeline, AutoencoderTiny
from diffusers.utils import load_image, make_image_grid

from streamdiffusion import StreamDiffusion
from streamdiffusion.image_utils import postprocess_image
from streamdiffusion.ip_adapter import patch_attention_processors, patch_unet_ip_adapter_projection


def add_label(image, text):
    draw = ImageDraw.Draw(image)
    # Use a default font if custom font is not available
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

device = 'mps'
dtype = torch.float16

# load vae
print('Loading Tiny VAE')
vae = AutoencoderTiny.from_pretrained(
    "madebyollin/taesdxl",
    torch_dtype=dtype,
).to(device)

# load pipeline
print('Loading Pipeline')
pipe = StableDiffusionXLPipeline.from_pretrained(
    'stabilityai/sdxl-turbo',
    vae=vae,
    torch_dtype=dtype,
    variant='fp16'
).to(device, dtype=dtype)

# Load and add IP-Adapter to the pipeline
print('Loading IP Adapter')
pipe.load_ip_adapter(
    "h94/IP-Adapter",
    subfolder="sdxl_models",
    weight_name="ip-adapter_sdxl.safetensors",
    torch_dtype=dtype
)
patch_attention_processors(pipe)
patch_unet_ip_adapter_projection(pipe)

# Load images
print('Loading Images')
input_image = load_image("/Users/himmelroman/Desktop/albert.png").resize((512, 512))
reference_image = load_image("/Users/himmelroman/Desktop/Gur.jpg")

# Loop through different scales
ip_scales = [0.7] #, 0.7, 0.8, 0.9]
strengths = [0.1, 0.2, 0.3, 0.4]
all_images = []

for ip in ip_scales:
    for s in strengths:

        # prepare
        ts_1 = 99 - int(100 * s)
        ts_2 = min(ts_1 + 10, 99)
        ts_3 = min(ts_2 + 10, 99)

        # StreamDiffusion
        print('Loading StreamDiffusion')
        stream = StreamDiffusion(
            pipe,
            device=device,
            t_index_list=[ts_1, ts_2, ts_3],
            original_inference_steps=100,
            torch_dtype=dtype,
            do_add_noise=True,
            height=512,
            width=904
        )
        stream.load_ip_adapter()
        stream.set_image_prompt_scale(ip)
        stream.generate_image_embedding(reference_image)

        print(f'Preparing for {ip=}, {s=}')
        stream.denoising_steps_num = len(stream.t_list)
        stream.prepare(
            num_inference_steps=100,
            seed=123
        )

        print(f'Generating for {ip=}, {s=}')
        for step in range(stream.denoising_steps_num - 1):
            stream(input_image, encode_input=True, decode_output=True)
        img_pt = stream(input_image, encode_input=True, decode_output=True)
        img_pil = postprocess_image(img_pt)[0]
        img_pil = add_label(img_pil, f'ip={ip}, str={s}, step={step}')
        all_images.append(img_pil)

grid = make_image_grid(all_images, rows=1, cols=len(all_images))
# grid = make_image_grid(all_images, rows=4, cols=3)
grid.save('grid5.jpg')
grid.show()
