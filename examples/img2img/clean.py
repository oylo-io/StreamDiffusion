import torch
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, AutoencoderTiny
from diffusers.utils import load_image, make_image_grid

from streamdiffusion import StreamDiffusion

device = 'mps'

# prepare SD pipeline
pipe_type = StableDiffusionXLPipeline
vae_model_id = "madebyollin/taesdxl"

# load vae
vae = AutoencoderTiny.from_pretrained(vae_model_id)

# load pipeline
pipe = pipe_type.from_pretrained(
    'stabilityai/sdxl-turbo',
    vae=vae,
    torch_dtype=torch.float16,
).to(device)

# StreamDiffusion
stream = StreamDiffusion(
    pipe,
    device=device,
    t_index_list=[33],
    original_inference_steps=100,
    torch_dtype=torch.float16,
    do_add_noise=False,
    height=512,
    width=904
)

# Load images
input_image = load_image("/Users/himmelroman/Desktop/sample.png").resize((512, 512))
reference_image = load_image("/Users/himmelroman/Desktop/ref.png")  # Image for IP-Adapter

# Set up generation parameters
prompt = "rabbit, high quality, best quality"

# Loop through different scales
ip_scales = [0.1, 0.3, 0.6]
strengths = [0.1, 0.3, 0.6]
all_images = []

for s in strengths:

    # prepare
    stream.t_list = [99 - int(100*s)]
    stream.prepare(
        prompt=prompt,
        num_inference_steps=100,
        guidance_scale=0.0,
        seed=int(123)
    )
    img = stream(input_image, encode_input=True, decode_output=True)
    all_images.append(img)

grid = make_image_grid(all_images, rows=1, cols=len(all_images))
grid.show()
