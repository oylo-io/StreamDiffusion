import torch
from diffusers import StableDiffusionXLPipeline

from streamdiffusion.acceleration.tensorrt.models import UNetXLTurbo


class UNetXLTurboWrapper(torch.nn.Module):
    def __init__(self, unet):
        super(UNetXLTurboWrapper, self).__init__()
        self.unet = unet

    def forward(self, sample, timestep, encoder_hidden_states, text_embeds, time_ids):
        added_cond_kwargs = {
            "text_embeds": text_embeds,
            "time_ids": time_ids
        }
        return self.unet(sample, timestep, encoder_hidden_states, added_cond_kwargs=added_cond_kwargs)


# load pipeline
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/sdxl-turbo",
    torch_dtype=torch.float16
).to('mps')

# wrap unet (fix added_cond_kwargs argument)
model_metadata = UNetXLTurbo()
model_wrapper = UNetXLTurboWrapper(pipe.unet)

sample, timestep, encoder_hidden_states, text_embeds, time_ids = model_metadata.get_sample_input(
    batch_size=1,
    image_height=512,
    image_width=912
)

torch.onnx.export(
    model_wrapper,
    (sample, timestep, encoder_hidden_states, text_embeds, time_ids),
    "/tmp/unet_sdxl_turbo.onnx",
    export_params=True,
    opset_version=20,
    input_names=model_metadata.get_input_names(),
    output_names=model_metadata.get_output_names(),
    dynamic_axes=None  # Modify if dynamic axes are needed
)
