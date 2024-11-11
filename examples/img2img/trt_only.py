import io
from pathlib import Path

import fire
import requests
import torch
from PIL import Image
from polygraphy import cuda
from diffusers import AutoencoderTiny, StableDiffusionPipeline

from streamdiffusion import StreamDiffusion
from streamdiffusion.acceleration.tensorrt.engine import AutoencoderKLEngine, UNet2DConditionModelEngine
from streamdiffusion.image_utils import postprocess_image

from diffusers.models.modeling_utils import ModelMixin


class TensorRTVAEWrapper(ModelMixin):
    def __init__(self, trt_vae_engine):
        super().__init__()
        self.trt_vae_engine = trt_vae_engine

    def encode(self, *args, **kwargs):
        # Call the encoding part of your TensorRT VAE engine
        return self.trt_vae_engine.encode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        # Call the decoding part of your TensorRT VAE engine
        return self.trt_vae_engine.decode(*args, **kwargs)

class TensorRTUNetWrapper(ModelMixin):
    def __init__(self, trt_unet_engine):
        super().__init__()
        self.trt_unet_engine = trt_unet_engine

    def forward(self, *args, **kwargs):
        # Call your TensorRT UNet engine's forward method here
        return self.trt_unet_engine(*args, **kwargs)


def run(
        trt_engine_dir: str = '/root/app/tensorrt/trt10/sd-turbo/'
):

    # load trt pipeline
    trt_pipe = load_trt_pipeline(
        model_id="stabilityai/sd-turbo",
        trt_engine_dir=trt_engine_dir
    )

    # init stream diffusion
    stream = StreamDiffusion(
        pipe=trt_pipe,
        t_index_list=[34],
        torch_dtype=torch.float16,
        width=904,
        height=512,
        cfg_type='self'
    )

    # prepare
    image = get_image("https://avatars.githubusercontent.com/u/79290761", 904, 512)
    stream.prepare(
        prompt="test this",
        negative_prompt="low quality, bad quality, blurry, low resolution"
    )

    # pre-process image
    input_latent = stream.image_processor.preprocess(image)

    # warmup
    for _ in range(5):
        stream(image=input_latent)

    # generate final sample image
    output_latent = stream(image=input_latent)

    # post-process image
    image = postprocess_image(output_latent, output_type='pil')
    image.save('/tmp/output.png')

def load_trt_pipeline(model_id, trt_engine_dir, device = "cuda", dtype = torch.float16):

    # process path
    trt_engine_dir = Path(trt_engine_dir)

    # create cuda stream
    cuda_stream = cuda.Stream()

    # load trt engines
    trt_vae = load_trt_vae(
        cuda_stream=cuda_stream,
        vae_encoder_path=str(trt_engine_dir / 'vae_encoder.engine'),
        vae_decoder_path=str(trt_engine_dir / 'vae_decoder.engine'),
        device=device,
        dtype=dtype
    )
    trt_unet = load_trt_unet(
        cuda_stream=cuda_stream,
        unet_path=str(trt_engine_dir / 'unet.engine'),
    )

    # init diffusers pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        vae=TensorRTVAEWrapper(trt_vae),
        unet=TensorRTUNetWrapper(trt_unet),
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False
    )

    return pipe


def load_trt_vae(cuda_stream, vae_encoder_path, vae_decoder_path,
                 device = "cuda", dtype = torch.float16, vae_scale_factor = 8):

    # load tiny vae
    vae = AutoencoderTiny.from_pretrained("madebyollin/taesd").to(
        device=device, dtype=dtype
    )

    # load TRT vae engine
    trt_vae = AutoencoderKLEngine(
        vae_encoder_path,
        vae_decoder_path,
        cuda_stream,
        vae_scale_factor,
        use_cuda_graph=False,
    )

    # take config
    setattr(trt_vae, "config", vae.config)
    setattr(trt_vae, "dtype", vae.dtype)

    return trt_vae

def load_trt_unet(cuda_stream, unet_path):

    # load TRT unet engine
    trt_unet = UNet2DConditionModelEngine(
        unet_path,
        cuda_stream,
        use_cuda_graph=False
    )

    return trt_unet


def get_image(url, width, height):

    # get image
    response = requests.get(url)

    # load & resize
    image = Image.open(io.BytesIO(response.content))
    image = image.resize((width, height))

    # return
    return image

if __name__ == "__main__":
    fire.Fire(run)
