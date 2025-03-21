import torch
from PIL import  ImageDraw, ImageFont

from diffusers import StableDiffusionXLPipeline, AutoencoderTiny
from diffusers.utils import load_image, make_image_grid

from streamdiffusion import StreamDiffusion
from streamdiffusion.image_utils import postprocess_image


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

# torch.cuda.empty_cache()

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
).to(device)

# Load and add IP-Adapter to the pipeline
print('Loading IP Adapter')
pipe.load_ip_adapter(
    "h94/IP-Adapter",
    subfolder="sdxl_models",
    weight_name="ip-adapter_sdxl.safetensors",
    torch_dtype=dtype
)

# StreamDiffusion
print('Loading StreamDiffusion')
stream = StreamDiffusion(
    pipe,
    device=device,
    t_index_list=[33],
    original_inference_steps=100,
    torch_dtype=dtype,
    do_add_noise=False,
    height=512,
    width=904
)

print('Loading Images')
# Load images
input_image = load_image("https://img.freepik.com/free-vector/duck-with-green-head-cartoon-character_1308-96950.jpg").resize((512, 512))
reference_image = load_image("/Users/himmelroman/Desktop/ref.png")  # Image for IP-Adapter

# Generate image embedding for IP-Adapter
print('Generating IP Embeddings')
stream.generate_image_embedding(reference_image)

# Set up generation parameters
prompt = "rabbit, high quality, best quality"

# Loop through different scales
ip_scales = [0.5, 0.2, 0.4, 0.6, 0.8]
strengths = [0.5, 0.2, 0.4, 0.6, 0.8]
all_images = []

for ip in ip_scales:
    for s in strengths:

        print(f'Preparing for {ip=}, {s=}')
        stream.ip_adapter_scale = ip

        # prepare
        stream.t_list = [99 - int(100 * s)]
        stream.prepare(
            prompt=prompt,
            num_inference_steps=100,
            guidance_scale=0.0,
            seed=int(123)
        )

        print(f'Generating for {ip=}, {s=}')
        img_pt = stream(input_image, encode_input=True, decode_output=True)
        img_pil = postprocess_image(img_pt)[0]
        img_pil = add_label(img_pil, f'str={s}')
        all_images.append(img_pil)

# grid = make_image_grid(all_images, rows=2, cols=len(all_images) // 2)
grid = make_image_grid(all_images, rows=1, cols=len(all_images))
grid.show()
