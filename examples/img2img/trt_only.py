import io
from pathlib import Path

import fire
import numpy as np
import torch
import requests
from PIL import Image
from polygraphy import cuda
from diffusers import StableDiffusionPipeline
from diffusers.configuration_utils import FrozenDict
from tqdm import tqdm

from streamdiffusion import StreamDiffusion
from streamdiffusion.image_utils import postprocess_image
from streamdiffusion.acceleration.tensorrt.engine import AutoencoderKLEngine, UNet2DConditionModelEngine

from diffusers.models.modeling_utils import ModelMixin


class TensorRTVAEWrapper(ModelMixin):
    def __init__(self, trt_vae_engine, dtype=torch.float16):
        super().__init__()
        self.trt_vae_engine = trt_vae_engine

        self._dtype = dtype
        self._config = FrozenDict({'in_channels': 3, 'out_channels': 3, 'encoder_block_out_channels': [64, 64, 64, 64], 'decoder_block_out_channels': [64, 64, 64, 64], 'act_fn': 'relu', 'upsample_fn': 'nearest', 'latent_channels': 4, 'upsampling_scaling_factor': 2, 'num_encoder_blocks': [1, 3, 3, 3], 'num_decoder_blocks': [3, 3, 3, 1], 'latent_magnitude': 3, 'latent_shift': 0.5, 'force_upcast': False, 'scaling_factor': 1.0, 'shift_factor': 0.0, 'block_out_channels': [64, 64, 64, 64], '_class_name': 'AutoencoderTiny', '_diffusers_version': '0.30.0', '_name_or_path': 'madebyollin/taesd'})

    @property
    def dtype(self):
        return self._dtype

    @property
    def config(self):
        return self._config

    @property
    def device(self):
        return "cuda"

    def encode(self, *args, **kwargs):
        # Call the encoding part of your TensorRT VAE engine
        return self.trt_vae_engine.encode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        # Call the decoding part of your TensorRT VAE engine
        return self.trt_vae_engine.decode(*args, **kwargs)

class TensorRTUNetWrapper(ModelMixin):
    def __init__(self, trt_unet_engine, dtype=torch.float16):
        super().__init__()
        self.trt_unet_engine = trt_unet_engine

        self._dtype = dtype
        self._config = FrozenDict({'sample_size': 64, 'in_channels': 4, 'out_channels': 4, 'center_input_sample': False, 'flip_sin_to_cos': True, 'freq_shift': 0, 'down_block_types': ['CrossAttnDownBlock2D', 'CrossAttnDownBlock2D', 'CrossAttnDownBlock2D', 'DownBlock2D'], 'mid_block_type': 'UNetMidBlock2DCrossAttn', 'up_block_types': ['UpBlock2D', 'CrossAttnUpBlock2D', 'CrossAttnUpBlock2D', 'CrossAttnUpBlock2D'], 'only_cross_attention': False, 'block_out_channels': [320, 640, 1280, 1280], 'layers_per_block': 2, 'downsample_padding': 1, 'mid_block_scale_factor': 1, 'dropout': 0.0, 'act_fn': 'silu', 'norm_num_groups': 32, 'norm_eps': 1e-05, 'cross_attention_dim': 1024, 'transformer_layers_per_block': 1, 'reverse_transformer_layers_per_block': None, 'encoder_hid_dim': None, 'encoder_hid_dim_type': None, 'attention_head_dim': [5, 10, 20, 20], 'num_attention_heads': None, 'dual_cross_attention': False, 'use_linear_projection': True, 'class_embed_type': None, 'addition_embed_type': None, 'addition_time_embed_dim': None, 'num_class_embeds': None, 'upcast_attention': None, 'resnet_time_scale_shift': 'default', 'resnet_skip_time_act': False, 'resnet_out_scale_factor': 1.0, 'time_embedding_type': 'positional', 'time_embedding_dim': None, 'time_embedding_act_fn': None, 'timestep_post_act': None, 'time_cond_proj_dim': None, 'conv_in_kernel': 3, 'conv_out_kernel': 3, 'projection_class_embeddings_input_dim': None, 'attention_type': 'default', 'class_embeddings_concat': False, 'mid_block_only_cross_attention': None, 'cross_attention_norm': None, 'addition_embed_type_num_heads': 64, '_class_name': 'UNet2DConditionModel', '_diffusers_version': '0.24.0.dev0', '_name_or_path': '/Users/himmelroman/.cache/huggingface/hub/models--stabilityai--sd-turbo/snapshots/b261bac6fd2cf515557d5d0707481eafa0485ec2/unet'})

    @property
    def dtype(self):
        return self._dtype

    @property
    def config(self):
        return self._config

    @property
    def device(self):
        return "cuda"

    def forward(self, *args, **kwargs):
        return self.trt_unet_engine(*args, **kwargs)


class CachedEmbeddingPipeline(StableDiffusionPipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # prompt embedding cache
        self.cached_prompt = None
        self.cached_prompt_embeds = None

    @torch.no_grad()
    def update_prompt(self, prompt: str) -> None:

        # compute embedding
        encoder_output = self.encode_prompt(
            prompt=prompt,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False
        )

        # update cache
        self.cached_prompt = prompt
        self.cached_prompt_embeds = encoder_output[0]

    def __call__(self, prompt, *args, **kwargs):

        # Ensure prompt embedding is up-to-date
        if prompt != self.cached_prompt or self.cached_prompt_embeds is None:
            self.update_prompt(prompt)

        # Call the main pipeline with cached embeddings
        return super().__call__(
            prompt_embeds=self.cached_prompt_embeds,
            *args,
            **kwargs
        )


def run(
        trt_engine_dir: str = '/root/app/tensorrt/trt10/sd-turbo/'
):

    # get input image
    image = get_image("https://avatars.githubusercontent.com/u/79290761", 904, 512)

    # load trt pipeline
    trt_pipe = load_trt_pipeline(
        model_id="stabilityai/sd-turbo",
        trt_engine_dir=trt_engine_dir
    )

    # prepare timers
    timer_event = getattr(torch, "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    pipe_start = timer_event.Event(enable_timing=True)
    pipe_end = timer_event.Event(enable_timing=True)

    # warmup
    for _ in range(3):
        trt_pipe(prompt='warmup',
                 image=image,
                 num_inference_steps=1,
                 guidance_scale=1.0,
                 height=512,
                 width=904)

    results = []
    for _ in tqdm(range(100)):
        pipe_start.record()
        result = trt_pipe(prompt='beautiful female dog',
                          image=image,
                          num_inference_steps=1,
                          guidance_scale=1.0,
                          height=512,
                          width=904
                          ).images
        pipe_end.record()
        timer_event.synchronize()
        results.append(pipe_start.elapsed_time(pipe_end))
    print_results('pipe', results)

    # init stream diffusion
    stream = StreamDiffusion(
        pipe=trt_pipe,
        t_index_list=[34],
        torch_dtype=torch.float16,
        height=512,
        width=904,
        cfg_type='self'
    )

    # prepare
    stream.prepare(
        prompt='beautiful female dog',
        guidance_scale=1.0
    )

    # pre-process image
    input_latent = stream.image_processor.preprocess(image)

    # warmup
    for _ in range(3):
        stream(input_latent)

    results = []
    for _ in tqdm(range(100)):
        pipe_start.record()
        output_latent = stream(input_latent)
        pipe_end.record()
        timer_event.synchronize()
        results.append(pipe_start.elapsed_time(pipe_end))
    print_results('stream', results)

    # post-process image
    image = postprocess_image(output_latent, output_type='pil')
    image[0].save('/tmp/stream_out.png')


def print_results(title, results):

    print(title.upper())

    # print results
    print(f"Average time: {sum(results) / len(results)}ms")
    print(f"Average FPS: {1000 / (sum(results) / len(results))}")

    fps_arr = 1000 / np.array(results)
    print(f"Max FPS: {np.max(fps_arr)}")
    print(f"Min FPS: {np.min(fps_arr)}")
    print(f"Std: {np.std(fps_arr)}")


def load_trt_pipeline(
        trt_engine_dir,
        model_id = "stabilityai/sd-turbo",
        vae_scale_factor = 8,
        device = "cuda",
        dtype=torch.float16):

    # process path
    trt_engine_dir = Path(trt_engine_dir)

    # create cuda stream
    cuda_stream = cuda.Stream()

    # load TensorRT VAE
    trt_vae = AutoencoderKLEngine(
        encoder_path=str(trt_engine_dir / 'vae_encoder.engine'),
        decoder_path=str(trt_engine_dir / 'vae_decoder.engine'),
        stream=cuda_stream,
        scaling_factor=vae_scale_factor
    )

    # load TensorRT UNET
    trt_unet = UNet2DConditionModelEngine(
        filepath=str(trt_engine_dir / 'unet.engine'),
        stream=cuda_stream
    )

    # create SD pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        vae=TensorRTVAEWrapper(trt_vae),
        unet=TensorRTUNetWrapper(trt_unet)
    )

    # Set the device and dtype for text encoder
    pipe.text_encoder.to(device, dtype=dtype)

    return pipe


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


# pip install --no-cache-dir git+https://github.com/himmelroman/StreamDiffusion.git@main#egg=streamdiffusion
# git clone https://github.com/oylo-io/StreamDiffusion.git
# cd StreamDiffusion/examples/img2img/
# python trt_only.py