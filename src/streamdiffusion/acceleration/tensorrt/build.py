import argparse
from pathlib import Path

import torch
from diffusers import AutoencoderTiny, StableDiffusionPipeline

from streamdiffusion import StreamDiffusion
from streamdiffusion.acceleration.tensorrt import accelerate_with_tensorrt


def accelerate_pipeline(model_id, vae_id, height, width, timestep_list, export_dir):

    # load vae
    vae = AutoencoderTiny.from_pretrained(vae_id)

    # create SD pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        vae=vae
    ).to('cuda')

    # StreamDiffusion
    stream = StreamDiffusion(
        pipe,
        t_index_list=timestep_list,
        torch_dtype=torch.float16,
        height=height,
        width=width
    )

    # Set batch sizes
    vae_batch_size = 1
    unet_batch_size = len(timestep_list)

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
    parser.add_argument('--model_id',
                        type=str, default='stabilityai/sd-turbo')
    parser.add_argument('--vae_id',
                        type=str, default='madebyollin/taesd')
    parser.add_argument('--export_dir',
                        type=Path, required=True, help='Directory for generated models')
    parser.add_argument('--height',
                        type=int, required=True, help='image height')
    parser.add_argument('--width',
                        type=int, required=True, help='image width')
    parser.add_argument('--timestep_list',
                        type=int, nargs='+', default=[33],
                        help='List of timestep indices for denoising')

    args = parser.parse_args()

    accelerate_pipeline(
        args.model_id,
        args.vae_id,
        args.height,
        args.width,
        args.timestep_list,
        args.export_dir
    )

# Usage:
# docker run -it --rm --gpus all -v ~/.cache/huggingface:/root/.cache/huggingface -v ~/oylo/models:/root/app/engines builder
# python3 src/streamdiffusion/acceleration/tensorrt/build.py --height 512 --width 904 --timestep_list 32 45 --export_dir /root/app/engines/sd-turbo_b2