import io
from pathlib import Path

import fire
import requests
import torch
from PIL import Image
from diffusers.configuration_utils import FrozenDict
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
        self.config = FrozenDict({'in_channels': 3, 'out_channels': 3, 'encoder_block_out_channels': [64, 64, 64, 64], 'decoder_block_out_channels': [64, 64, 64, 64], 'act_fn': 'relu', 'upsample_fn': 'nearest', 'latent_channels': 4, 'upsampling_scaling_factor': 2, 'num_encoder_blocks': [1, 3, 3, 3], 'num_decoder_blocks': [3, 3, 3, 1], 'latent_magnitude': 3, 'latent_shift': 0.5, 'force_upcast': False, 'scaling_factor': 1.0, 'shift_factor': 0.0, 'block_out_channels': [64, 64, 64, 64], '_class_name': 'AutoencoderTiny', '_diffusers_version': '0.30.0', '_name_or_path': 'madebyollin/taesd'})

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
        self.config = FrozenDict({'sample_size': 64, 'in_channels': 4, 'out_channels': 4, 'center_input_sample': False, 'flip_sin_to_cos': True, 'freq_shift': 0, 'down_block_types': ['CrossAttnDownBlock2D', 'CrossAttnDownBlock2D', 'CrossAttnDownBlock2D', 'DownBlock2D'], 'mid_block_type': 'UNetMidBlock2DCrossAttn', 'up_block_types': ['UpBlock2D', 'CrossAttnUpBlock2D', 'CrossAttnUpBlock2D', 'CrossAttnUpBlock2D'], 'only_cross_attention': False, 'block_out_channels': [320, 640, 1280, 1280], 'layers_per_block': 2, 'downsample_padding': 1, 'mid_block_scale_factor': 1, 'dropout': 0.0, 'act_fn': 'silu', 'norm_num_groups': 32, 'norm_eps': 1e-05, 'cross_attention_dim': 1024, 'transformer_layers_per_block': 1, 'reverse_transformer_layers_per_block': None, 'encoder_hid_dim': None, 'encoder_hid_dim_type': None, 'attention_head_dim': [5, 10, 20, 20], 'num_attention_heads': None, 'dual_cross_attention': False, 'use_linear_projection': True, 'class_embed_type': None, 'addition_embed_type': None, 'addition_time_embed_dim': None, 'num_class_embeds': None, 'upcast_attention': None, 'resnet_time_scale_shift': 'default', 'resnet_skip_time_act': False, 'resnet_out_scale_factor': 1.0, 'time_embedding_type': 'positional', 'time_embedding_dim': None, 'time_embedding_act_fn': None, 'timestep_post_act': None, 'time_cond_proj_dim': None, 'conv_in_kernel': 3, 'conv_out_kernel': 3, 'projection_class_embeddings_input_dim': None, 'attention_type': 'default', 'class_embeddings_concat': False, 'mid_block_only_cross_attention': None, 'cross_attention_norm': None, 'addition_embed_type_num_heads': 64, '_class_name': 'UNet2DConditionModel', '_diffusers_version': '0.24.0.dev0', '_name_or_path': '/Users/himmelroman/.cache/huggingface/hub/models--stabilityai--sd-turbo/snapshots/b261bac6fd2cf515557d5d0707481eafa0485ec2/unet'})

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
        stream(input_latent)

    # generate final sample image
    output_latent = stream(input_latent)

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

    # # init diffusers pipeline
    # pipe = StableDiffusionPipeline.from_pretrained(
    #     model_id,
    #     vae=TensorRTVAEWrapper(trt_vae),
    #     unet=TensorRTUNetWrapper(trt_unet),
    #     torch_dtype=dtype,
    #     safety_checker=None,
    #     requires_safety_checker=False
    # ).to(device=device, dtype=dtype)

    # Initialize a partial pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        "stabilityai/sd-turbo",
        torch_dtype=torch.float16,
        vae=TensorRTVAEWrapper(trt_vae),
        unet=TensorRTUNetWrapper(trt_unet)
    )
    pipe.to("cuda")  # Move tokenizer and text encoder to CUDA

    return pipe


def load_trt_vae(cuda_stream, vae_encoder_path, vae_decoder_path,
                 device = "cuda", dtype = torch.float16, vae_scale_factor = 8):

    # load TRT vae engine
    trt_vae = AutoencoderKLEngine(
        vae_encoder_path,
        vae_decoder_path,
        cuda_stream,
        vae_scale_factor,
        use_cuda_graph=False,
    )

    # take config
    # setattr(trt_vae, "dtype", vae.dtype)

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


# pip install --no-cache-dir git+https://github.com/himmelroman/StreamDiffusion.git@main#egg=streamdiffusion
# git clone https://github.com/oylo-io/StreamDiffusion.git
# cd StreamDiffusion/examples/img2img/
# python trt_only.py