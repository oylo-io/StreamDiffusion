import torch
from PIL import  ImageDraw, ImageFont

from diffusers import StableDiffusionXLPipeline, AutoencoderTiny, StableDiffusionPipeline
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

device = 'mps'
dtype = torch.float16

# load vae
print('Loading Tiny VAE')
vae = AutoencoderTiny.from_pretrained(
    # "madebyollin/taesd",
    "madebyollin/taesdxl",
    torch_dtype=dtype,
).to(device)

# load pipeline
print('Loading Pipeline')
pipe = StableDiffusionXLPipeline.from_pretrained(
    # 'stabilityai/sd-turbo',
    'stabilityai/sdxl-turbo',
    vae=vae,
    torch_dtype=dtype,
    variant='fp16'
).to(device, dtype=dtype)

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

# Load images
print('Loading Images')
input_image = load_image("/Users/himmelroman/Desktop/albert.png").resize((512, 512))

# Generate prompt embeddings
print('Generating Prompt Embeddings')
prompt = "ibex, high quality, best quality"
stream.update_prompt(prompt)

# Loop through different scales
prompt_weights = [0.1, 0.3, 0.6, 0.9]
all_images = []

for w in range(10):

    w /= 10

    # prepare
    stream.t_list = [20]
    stream.denoising_steps_num = len(stream.t_list)
    stream.prepare(
        num_inference_steps=100,
        seed=123
    )

    print(f'Generating for {w=}')
    prompt = f"medieval hilltop town (snow){0.0 + w * 2}"
    stream.update_prompt(prompt)

    for _ in range(stream.denoising_steps_num - 1):
        stream(input_image, encode_input=True, decode_output=True)

    img_pt = stream(input_image, encode_input=True, decode_output=True)
    img_pil = postprocess_image(img_pt)[0]
    img_pil = add_label(img_pil, f"(ibex){w}, (rabbit){1-w}")
    all_images.append(img_pil)

# grid = make_image_grid(all_images, rows=1, cols=len(all_images))
grid = make_image_grid(all_images, rows=2, cols=5)
grid.save('grid.jpg')
grid.show()
