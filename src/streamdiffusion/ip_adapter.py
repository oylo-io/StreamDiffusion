from typing import Optional, Union, List

import torch
from diffusers.models.attention_processor import IPAdapterAttnProcessor, IPAdapterAttnProcessor2_0, Attention

class TensorIPAdapterScaleExtractor:

    def get_scale(self, ip_adapter_scale, scale_list):

        # result
        result_scale = scale_list

        # If ip_adapter_scale is provided, update self.scale
        if ip_adapter_scale is not None:
            num_adapters = len(scale_list)

            # Handle tensor input
            if isinstance(ip_adapter_scale, torch.Tensor):
                # Single value tensor for all adapters
                if ip_adapter_scale.numel() == 1:
                    result_scale = [ip_adapter_scale.item()] * num_adapters
                # Tensor with one value per adapter
                elif ip_adapter_scale.numel() == num_adapters:
                    result_scale = [ip_adapter_scale[i].item() for i in range(num_adapters)]
                else:
                    raise ValueError(
                        f"ip_adapter_scale tensor has {ip_adapter_scale.numel()} elements, but expected {num_adapters}")

            # Handle float or list inputs
            elif not isinstance(ip_adapter_scale, list):
                result_scale = [ip_adapter_scale] * num_adapters
            elif len(ip_adapter_scale) == num_adapters:
                result_scale = ip_adapter_scale
            else:
                raise ValueError(
                    f"ip_adapter_scale list has {len(ip_adapter_scale)} elements, but expected {num_adapters}")

        return result_scale


class TensorIPAdapterAttnProcessor(IPAdapterAttnProcessor, TensorIPAdapterScaleExtractor):

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


class TensorIPAdapterAttnProcessor2_0(IPAdapterAttnProcessor2_0, TensorIPAdapterScaleExtractor):
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

    # Get all the processors - this returns a copy, not the actual dictionary
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
                new_processor.to_k_ip[i].weight.data.copy_(processor.to_k_ip[i].weight.data)
                new_processor.to_v_ip[i].weight.data.copy_(processor.to_v_ip[i].weight.data)

            # store the new processor
            all_processors[key] = new_processor

    # set all processors to the model
    pipe.unet.set_attn_processor(all_processors)
