import io
from typing import *

import fire
import PIL.Image
import requests
import torch
from diffusers import AutoencoderTiny, LCMScheduler, StableDiffusionPipeline
from tqdm import tqdm

from streamdiffusion import StreamDiffusion
from streamdiffusion.image_utils import pil2tensor, postprocess_image


def download_image(url: str):
    response = requests.get(url)
    image = PIL.Image.open(io.BytesIO(response.content))
    return image


def run(
    wamup: int = 10, iterations: int = 50, acceleration: Optional[Literal["xformers", "sfast", "tensorrt"]] = None
):
    pipe: StableDiffusionPipeline = StableDiffusionPipeline.from_single_file("./model.safetensors").to(
        device=torch.device("cuda"),
        dtype=torch.float16,
    )
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd").to(device=pipe.device, dtype=pipe.dtype)
    pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
    pipe.fuse_lora()

    stream = StreamDiffusion(
        pipe,
        [32, 45],
        torch_dtype=torch.float16,
    )

    if acceleration == "xformers":
        pipe.enable_xformers_memory_efficient_attention()
    elif acceleration == "tensorrt":
        from streamdiffusion.acceleration.tensorrt import accelerate_with_tensorrt

        stream = accelerate_with_tensorrt(stream)
    elif acceleration == "sfast":
        from streamdiffusion.acceleration.sfast import accelerate_with_stable_fast

        stream = accelerate_with_stable_fast(stream)

    stream.prepare(
        "Girl with panda ears wearing a hood",
        num_inference_steps=50,
        generator=torch.manual_seed(2),
    )

    image = download_image("https://github.com/ddpn08.png").resize((512, 512))
    input_tensor = pil2tensor(image)

    # warmup
    for _ in range(wamup):
        stream(input_tensor.detach().clone().to(device=stream.device, dtype=stream.dtype))

    results = []

    for _ in tqdm(range(iterations)):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        x_output = stream(input_tensor.detach().clone().to(device=stream.device, dtype=stream.dtype))
        postprocess_image(x_output, output_type="pil")[0]
        end.record()

        torch.cuda.synchronize()
        results.append(start.elapsed_time(end))

    print(f"Average time: {sum(results) / len(results)}ms")
    print(f"Average FPS: {1000 / (sum(results) / len(results))}")


if __name__ == "__main__":
    fire.Fire(run)