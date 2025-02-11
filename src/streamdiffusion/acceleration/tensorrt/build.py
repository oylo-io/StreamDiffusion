import argparse
from pathlib import Path

import torch
from diffusers import AutoencoderTiny, StableDiffusionPipeline, StableDiffusionXLPipeline

from streamdiffusion import StreamDiffusion
from streamdiffusion.acceleration.tensorrt import accelerate_with_tensorrt


def accelerate_pipeline(is_sdxl, model_id, height, width, num_timesteps, export_dir):

    # prepare SD pipeline
    pipe_type = StableDiffusionPipeline
    vae_model_id = "madebyollin/taesd"
    if is_sdxl:
        vae_model_id = "madebyollin/taesdxl"
        pipe_type = StableDiffusionXLPipeline

    # load vae
    vae = AutoencoderTiny.from_pretrained(vae_model_id)

    print(f'Loading {pipe_type=} with {model_id=} and {vae_model_id}')

    # load pipeline
    pipe = pipe_type.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        vae=vae
    ).to('cuda')

    # StreamDiffusion
    stream = StreamDiffusion(
        pipe,
        t_index_list=list(range(num_timesteps)),
        torch_dtype=torch.float16,
        height=height,
        width=width
    )
    print(f'Stream is {stream.sdxl=}')

    # Set batch sizes
    vae_batch_size = 1
    unet_batch_size = num_timesteps

    # build models
    accelerate_with_tensorrt(
        stream=stream,
        engine_dir=str(export_dir),
        unet_batch_size=(unet_batch_size, unet_batch_size),
        vae_batch_size=(vae_batch_size, vae_batch_size),
        unet_engine_build_options={
            'opt_image_height': height,
            'opt_image_width': width,
            'min_image_resolution': min(height, width),
            'max_image_resolution': max(height, width),
            'opt_batch_size': unet_batch_size,
            'build_static_batch': True,
            'build_dynamic_shape': False
        },
        vae_engine_build_options={
            'opt_image_height': height,
            'opt_image_width': width,
            'min_image_resolution': min(height, width),
            'max_image_resolution': max(height, width),
            'opt_batch_size': vae_batch_size,
            'build_static_batch': True,
            'build_dynamic_shape': False
        }
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Accelerate Pipeline with TRT")
    parser.add_argument('--sdxl',
                        type=bool, default=False)
    parser.add_argument('--model_id',
                        type=str, default='stabilityai/sd-turbo')
    # parser.add_argument('--vae_id',
    #                     type=str, default='madebyollin/taesd')
    parser.add_argument('--export_dir',
                        type=Path, required=True, help='Directory for generated models')
    parser.add_argument('--height',
                        type=int, required=True, help='image height')
    parser.add_argument('--width',
                        type=int, required=True, help='image width')
    parser.add_argument('--num_timesteps',
                        type=int, default=1, help='number of timesteps')

    args = parser.parse_args()

    accelerate_pipeline(
        args.sdxl,
        args.model_id,
        # args.vae_id,
        args.height,
        args.width,
        args.num_timesteps,
        args.export_dir
    )

# Usage:
# docker run -it --rm --gpus all -v ~/.cache/huggingface:/root/.cache/huggingface -v ~/oylo/models:/root/app/engines builder
# python3 src/streamdiffusion/acceleration/tensorrt/build.py --height 512 --width 904 --num_timesteps 2 --export_dir /root/app/engines/sd-turbo_b2
# python3 src/streamdiffusion/acceleration/tensorrt/build.py --height 512 --width 904 --num_timesteps 1 --export_dir /root/app/engines/sdxl-turbo --sdxl True