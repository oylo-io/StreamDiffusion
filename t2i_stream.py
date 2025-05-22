import torch

from diffusers.utils import load_image
from diffusers import (
    T2IAdapter,
    AutoPipelineForImage2Image,
    MultiAdapter, StableDiffusionXLAdapterPipeline, StableDiffusionXLPipeline
)

import torch.nn.functional as F
import torchvision.transforms as T

from streamdiffusion import StreamDiffusion
from streamdiffusion.adapters.control_adapter import DepthFeatureExtractor, CannyFeatureExtractor


def pre_process_image(image, height, width):
    # Convert to tensor (values 0-255), keeping HWC format
    image_pt = torch.from_numpy(image)

    # Move to device first
    image_pt = image_pt.to(device="mps", dtype=torch.float16)

    # adds the "batch" dimension to make shape (B, H, W, C)
    image_pt = image_pt.unsqueeze(0)

    # Do permute on GPU (BHWC → BCHW)
    image_pt = image_pt.permute(0, 3, 1, 2)

    # resize
    if image_pt.shape[2] != height or image_pt.shape[3] != width:
        image_pt = F.interpolate(image_pt, size=(height, width), mode="bilinear", align_corners=False)

    # Scale to 0-1 range (PyTorch standard)
    image_pt = image_pt / 255.0

    # Normalize to -1-1 range (StableDiffusion standard)
    image_pt = image_pt * 2 - 1

    return image_pt

# Main process
def process_image_with_dual_adapters(
        image_path,
        prompt,
        model_id="stabilityai/sdxl-turbo",
        depth_weight=0.6,
        canny_weight=0.4,
        strength=0.5,
        num_inference_steps=4
):
    # Load original image
    original_image = load_image(image_path).resize((768, 768))

    # Initialize depth estimator
    depth_pipeline = DepthFeatureExtractor(device="mps")
    depth_map = depth_pipeline.generate(original_image)

    # Generate Canny edges on GPU
    image_tensor = T.ToTensor()(original_image).unsqueeze(0).to("mps")
    edge_detector = CannyFeatureExtractor(device="mps")
    canny_tensor = edge_detector.generate(image_tensor)
    canny_image = T.ToPILImage()(canny_tensor[0])

    # Load adapters
    depth_adapter = T2IAdapter.from_pretrained(
        "TencentARC/t2i-adapter-depth-zoe-sdxl-1.0",
        torch_dtype=torch.float16,
    )

    canny_adapter = T2IAdapter.from_pretrained(
        "TencentARC/t2i-adapter-canny-sdxl-1.0",
        torch_dtype=torch.float16,
    )

    # Create MultiAdapter
    multi_adapter = MultiAdapter(
        adapters=[canny_adapter, depth_adapter]
    )

    # Initialize pipeline
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        variant="fp16",
    ).to("mps")

    # init stream
    stream = StreamDiffusion(
        pipeline,
        device = "mps",
        torch_dtype = torch.float16,
        # height = self.image_height,
        # width = self.image_width,
        t_index_list = [33],
        original_inference_steps = 1000,
        do_add_noise = True
    )
    stream.load_control_adapter()

    # generate control images
    input_tensor = pre_process_image(
        image=original_image,
        height=512,
        width=904
    )
    control = stream.generate_control_images(input_tensor)

    result = stream(original_image, control=control, encode_input=True, decode_output=True)

    # Generate image
    # result = pipeline(
    #     prompt,
    #     image=[canny_image, depth_map],# image=original_image,
    #     # adapter_images=[canny_image, depth_map],
    #     num_inference_steps=num_inference_steps,
    #     guidance_scale=0.0,  # For SDXL-Turbo
    #     # strength=strength,
    #     adapter_conditioning_scale=[canny_weight, depth_weight],
    #     adapter_conditioning_factor=1.0
    # )

    return result, depth_map, canny_image


# Use the function
output_image, depth, canny = process_image_with_dual_adapters(
    image_path="/Users/himmelroman/Desktop/images/sample.jpg",
    prompt="nina simone standing next to a screen with a statue of a naked greek goddess",
    depth_weight=0.6,
    canny_weight=1.5,
    strength=0.6
)

# Display results
output_image.show()
