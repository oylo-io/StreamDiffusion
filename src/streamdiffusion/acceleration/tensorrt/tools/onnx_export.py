import argparse
from pathlib import Path

import torch
from diffusers import AutoencoderTiny, StableDiffusionPipeline, StableDiffusionXLPipeline

from streamdiffusion.acceleration.tensorrt.models import UNetXLTurboIPAdapter
from streamdiffusion.ip_adapter import prepare_unet_for_onnx_export, patch_unet_ip_adapter_projection


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


def export(is_sdxl, model_id, ip_adapter, height, width, num_timesteps, export_dir):

    device = 'mps'
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
    ).to(device)

    # load pipeline
    print(f'Loading Pipeline {pipe_type}')
    pipe = pipe_type.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        variant='fp16' if is_sdxl else None,
        vae=vae
    ).to(device)

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
        prepare_unet_for_onnx_export(pipe)
        patch_unet_ip_adapter_projection(pipe)

    # Set batch sizes
    vae_batch_size = 1
    unet_batch_size = num_timesteps

    # wrap to handle add_cond_kwargs correctly for sdxl
    if is_sdxl:
        if ip_adapter:
            pipe.unet = UNetXLIPAdapterWrapper(pipe.unet)
        else:
            pipe.unet = UNetXLWrapper(pipe.unet)

    # export to onnx
    with torch.inference_mode():
        model_data = UNetXLTurboIPAdapter(device=device)
        inputs = model_data.get_sample_input(unet_batch_size, height, width)
        torch.onnx.export(
            pipe.unet,
            inputs,
            str(export_dir / 'unet.onnx'),
            input_names=model_data.get_input_names(),
            output_names=model_data.get_output_names(),
            dynamic_axes=model_data.get_dynamic_axes(),
            opset_version=20,
            export_params=True,
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX_FALLTHROUGH
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

    # verify dir
    Path(args.export_dir).mkdir(parents=True, exist_ok=True)

    export(
        args.sdxl,
        args.model_id,
        args.ip_adapter,
        args.height,
        args.width,
        args.num_timesteps,
        args.export_dir
    )
