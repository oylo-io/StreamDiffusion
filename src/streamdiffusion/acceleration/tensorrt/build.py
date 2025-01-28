import argparse
from pathlib import Path

import torch
from diffusers import AutoencoderTiny, StableDiffusionPipeline

from streamdiffusion import StreamDiffusion
from streamdiffusion.acceleration.tensorrt import accelerate_with_tensorrt


def accelerate_pipeline(model_id, vae_id, height, width, export_dir):

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
        t_index_list=[33],
        torch_dtype=torch.float16,
        height=height,
        width=width
    )

    # build models
    accelerate_with_tensorrt(
        stream,
        str(export_dir),
        max_batch_size=1,
        min_batch_size=1,
        use_cuda_graph=False,
        engine_build_options={
            'opt_batch_size': 1,
            'opt_image_height': height,
            'opt_image_width': width,
            'min_image_resolution': min(height, width),
            'max_image_resolution': max(height, width),
            'build_static_batch': True,
            'build_dynamic_shape': False
        }
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Accelerate Pipeline with TRT")
    parser.add_argument('--model_id', type=str, default='stabilityai/sd-turbo')
    parser.add_argument('--vae_id', type=str, default='madebyollin/taesd')
    parser.add_argument('--export_dir', type=Path, required=True, help='Directory for generated models')
    parser.add_argument('--height', type=int, required=True, help='image height')
    parser.add_argument('--width', type=int, required=True, help='image width')
    args = parser.parse_args()

    accelerate_pipeline(
        args.model_id,
        args.vae_id,
        args.height,
        args.width,
        args.export_dir
    )
