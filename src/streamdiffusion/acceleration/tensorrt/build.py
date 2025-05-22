import argparse
from pathlib import Path

import torch
from diffusers import AutoencoderTiny, StableDiffusionPipeline, StableDiffusionXLPipeline

from streamdiffusion import StreamDiffusion
from streamdiffusion.acceleration.tensorrt.compile import accelerate_with_tensorrt
from streamdiffusion.adapters.ip_adapter import patch_attention_processors, patch_unet_ip_adapter_projection


class UNetXLWrapper(torch.nn.Module):
    def __init__(self, unet):
        super(UNetXLWrapper, self).__init__()
        self.unet = unet

    def forward(self, sample, timestep, encoder_hidden_states, text_embeds, time_ids):
        added_cond_kwargs = {
            "text_embeds": text_embeds,
            "time_ids": time_ids
        }
        return self.unet(sample, timestep, encoder_hidden_states, added_cond_kwargs=added_cond_kwargs)


class UNetXLIPAdapterWrapper(torch.nn.Module):
    def __init__(self, unet):
        super(UNetXLIPAdapterWrapper, self).__init__()
        self.unet = unet

    def forward(self, sample, timestep, encoder_hidden_states, text_embeds, time_ids, image_embeds, ip_adapter_scale):
        added_cond_kwargs = {
            "text_embeds": text_embeds,
            "time_ids": time_ids,
            "image_embeds": image_embeds
        }
        cross_attention_kwargs = {
            "ip_adapter_scale": ip_adapter_scale
        }
        return self.unet(sample,
                         timestep,
                         encoder_hidden_states,
                         timestep_cond=None,
                         added_cond_kwargs=added_cond_kwargs, cross_attention_kwargs=cross_attention_kwargs)


def accelerate_pipeline(is_sdxl, model_id, ip_adapter, height, width, num_timesteps, export_dir):

    device = 'cuda'
    dtype = torch.float16

    # prepare SD pipeline
    pipe_type = StableDiffusionPipeline
    vae_model_id = "madebyollin/taesd"
    if is_sdxl:
        vae_model_id = "madebyollin/taesdxl"
        pipe_type = StableDiffusionXLPipeline

    # load vae
    print(f'Loading VAE {vae_model_id}')
    vae = AutoencoderTiny.from_pretrained(
        vae_model_id,
        torch_dtype=dtype
    ).to('cuda')

    # load pipeline
    print(f'Loading Pipeline {pipe_type}')
    pipe = pipe_type.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        variant='fp16' if is_sdxl else None,
        vae=vae
    ).to('cuda')

    # load ip adapter
    if ip_adapter and is_sdxl:
        print(f'Loading IPAdapter')
        pipe.load_ip_adapter(
            "h94/IP-Adapter",
            subfolder="sdxl_models",
            weight_name="ip-adapter_sdxl.safetensors",
            torch_dtype=dtype
        )

        # patch for onnx compatibility
        print('Patching IPAdapter')
        patch_attention_processors(pipe)
        patch_unet_ip_adapter_projection(pipe)

    # StreamDiffusion
    stream = StreamDiffusion(
        pipe,
        device=device,
        t_index_list=list(range(num_timesteps)),
        original_inference_steps=100,
        torch_dtype=dtype,
        height=height,
        width=width
    )

    # Set batch sizes
    vae_batch_size = 1
    unet_batch_size = num_timesteps

    # wrap to handle add_cond_kwargs correctly for sdxl
    if is_sdxl:
        if ip_adapter:
            pipe.unet = UNetXLIPAdapterWrapper(pipe.unet)
        else:
            pipe.unet = UNetXLWrapper(pipe.unet)

    # build models
    accelerate_with_tensorrt(
        stream=stream,
        is_sdxl=is_sdxl,
        ip_adapter=ip_adapter,
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
    parser.add_argument('--export_dir',
                        type=Path, required=True, help='Directory for generated models')
    parser.add_argument('--height',
                        type=int, required=True, help='image height')
    parser.add_argument('--width',
                        type=int, required=True, help='image width')
    parser.add_argument('--num_timesteps',
                        type=int, default=1, help='number of timesteps')
    parser.add_argument('--sdxl', default=False, action='store_true')
    parser.add_argument('--ip_adapter', default=False, action='store_true')

    args = parser.parse_args()

    accelerate_pipeline(
        args.sdxl,
        args.model_id,
        # args.vae_id,
        args.ip_adapter,
        args.height,
        args.width,
        args.num_timesteps,
        args.export_dir
    )

# Usage:
# docker run -it --rm --gpus all -v ~/.cache/huggingface:/root/.cache/huggingface -v ~/oylo/models:/root/app/engines builder
# python3 src/streamdiffusion/acceleration/tensorrt/build.py --height 512 --width 904 --num_timesteps 2 --export_dir /root/app/engines/sd-turbo_b2
# python3 src/streamdiffusion/acceleration/tensorrt/build.py --model_id stabilityai/sdxl-turbo --height 512 --width 904 --num_timesteps 1 --export_dir /root/app/engines/sdxl-turbo --sdxl True