from typing import Optional, Union, List

import torch
from diffusers.models.attention_processor import IPAdapterAttnProcessor, IPAdapterAttnProcessor2_0, Attention


class TensorIPAdapterAttnProcessor(IPAdapterAttnProcessor):

    def __call__(
            self,
            attn: Attention,
            hidden_states: torch.Tensor,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            temb: Optional[torch.Tensor] = None,
            ip_adapter_scale: float = 1.0,
            ip_adapter_masks: Optional[torch.Tensor] = None,
    ):
        # override scale
        self.scale = [ip_adapter_scale]

        # Call parent method
        return super().__call__(
            attn=attn,
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            temb=temb,
            ip_adapter_masks=ip_adapter_masks
        )

class TensorIPAdapterAttnProcessor2_0(IPAdapterAttnProcessor2_0):
    """
    ONNX-compatible version of IPAdapterAttnProcessor2_0 that accepts tensor scales
    """

    def __call__(
            self,
            attn: Attention,
            hidden_states: torch.Tensor,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            temb: Optional[torch.Tensor] = None,
            ip_adapter_scale: Optional[Union[float, torch.Tensor, List]] = None,
            ip_adapter_masks: Optional[torch.Tensor] = None,
            **kwargs
    ):
        # override scale
        self.scale = [ip_adapter_scale]

        # Call parent method
        return super().__call__(
            attn=attn,
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            temb=temb,
            ip_adapter_masks=ip_adapter_masks
        )

# Create a mapping from original processor types to ONNX-compatible versions
processor_mapping = {
    IPAdapterAttnProcessor: TensorIPAdapterAttnProcessor,
    IPAdapterAttnProcessor2_0: TensorIPAdapterAttnProcessor2_0,
}

def prepare_unet_for_onnx_export(pipe):

    # Get all the processors
    processors = pipe.unet.attn_processors

    # We need to include ALL processors in our update, not just the ones we're changing
    all_processors = {}

    # Prepare the complete processors dictionary
    for key, processor in processors.items():

        # Check if this processor needs to be replaced
        proc_cls = type(processor)
        if proc_cls not in processor_mapping:

            # copy processor as-is
            all_processors[key] = processor
        else:
            # Determine device of the original processor
            device = processor.to_k_ip[0].weight.device if len(processor.to_k_ip) > 0 else "cpu"

            # Create new processor
            new_processor = processor_mapping[proc_cls](
                hidden_size=processor.hidden_size,
                cross_attention_dim=processor.cross_attention_dim,
                num_tokens=processor.num_tokens,
                scale=processor.scale
            ).to(device)

            # Copy weights
            for i in range(len(processor.to_k_ip)):
                print(f'Source to_k_ip weights: {processor.to_k_ip[i].weight.data.dtype}')
                new_processor.to_k_ip[i].weight.data = processor.to_k_ip[i].weight.data.to(dtype=pipe.unet.dtype)
                print(f'Copied to_k_ip weights: {new_processor.to_k_ip[i].weight.data.dtype}')

                print(f'Source to_v_ip weights: {processor.to_v_ip[i].weight.data.dtype}')
                new_processor.to_v_ip[i].weight.data = processor.to_v_ip[i].weight.data.to(dtype=pipe.unet.dtype)
                print(f'Copied to_v_ip weights: {new_processor.to_v_ip[i].weight.data.dtype}')

            # store the new processor
            all_processors[key] = new_processor

    # set all processors to the model
    pipe.unet.set_attn_processor(all_processors)


class IPAdapterProjection:
    def __init__(self, pipe):
        self.device = pipe.device
        self.dtype = pipe.unet.dtype

        # Extract the projection layers from UNet
        self.image_proj = None

        if hasattr(pipe.unet, "encoder_hid_proj"):
            self.image_proj = pipe.unet.encoder_hid_proj.to(self.device, dtype=self.dtype)

    def __call__(self, image_embeds, text_embeds=None):
        """Project image embeddings and optionally text embeddings"""
        projected_image_embeds = None

        if self.image_proj is not None and image_embeds is not None:
            projected_image_embeds = self.image_proj(image_embeds.to(self.device, dtype=self.dtype))

        return projected_image_embeds


def patch_unet_ip_adapter_projection(pipe):

    # Save the original method
    original_process_encoder_hidden_states = pipe.unet.process_encoder_hidden_states

    # Define the replacement method
    def patched_process_encoder_hidden_states(self, encoder_hidden_states, added_cond_kwargs):

        # Check if pre-projected embeddings are provided
        if "image_embeds" in added_cond_kwargs:

            # Use pre-projected image embeddings directly
            image_embeds = added_cond_kwargs["image_embeds"]

            # Return tuple format expected by attention processors
            return encoder_hidden_states, image_embeds
        else:

            # Fall back to original method for other cases
            return original_process_encoder_hidden_states(self, encoder_hidden_states, added_cond_kwargs)

    # Apply the monkey patch
    pipe.unet.process_encoder_hidden_states = patched_process_encoder_hidden_states.__get__(pipe.unet)

    return original_process_encoder_hidden_states