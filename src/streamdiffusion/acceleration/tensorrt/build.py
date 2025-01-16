import torch
from diffusers import AutoencoderTiny, StableDiffusionPipeline

from streamdiffusion import StreamDiffusion
from streamdiffusion.acceleration.tensorrt import accelerate_with_tensorrt

# use TinyVAE
tiny_vae = AutoencoderTiny.from_pretrained("madebyollin/taesd")

# create SD pipeline
pipe = StableDiffusionPipeline.from_pretrained(
    'stabilityai/sd-turbo',
    torch_dtype=torch.float16,
    vae=tiny_vae
).to('cuda')

# StreamDiffusion
stream = StreamDiffusion(
    pipe,
    t_index_list=[33],
    torch_dtype=torch.float16,
    height=720,
    width=1280
)

accelerate_with_tensorrt(
    stream,
    '/tmp/engine',
    max_batch_size=1,
    min_batch_size=1,
    use_cuda_graph=False,
    engine_build_options={
        'opt_batch_size': 1,
        'opt_image_height': 720,
        'opt_image_width': 1280,
        'min_image_resolution': 256,
        'max_image_resolution': 1280,
        'build_dynamic_shape': False
    }
)
