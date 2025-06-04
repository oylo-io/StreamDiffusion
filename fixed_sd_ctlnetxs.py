"""
StreamDiffusion + ControlNetXS Implementation via Latent Interception

CLEAN APPROACH:
1. StreamDiffusion processes input image with its own noise scheduling
2. Intercept the noisy latents before they go into StreamDiffusion's UNet
3. Feed those latents to ControlNetXS pipeline for controlled generation

Two separate proven pipelines, no component surgery!
"""

import os
import torch
import numpy as np
from PIL import Image
import cv2

# Diffusers imports
from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLControlNetXSPipeline,
    ControlNetXSAdapter,
    AutoencoderTiny,
    EulerAncestralDiscreteScheduler
)
from diffusers.image_processor import VaeImageProcessor

# Import your StreamDiffusion class
from streamdiffusion import StreamDiffusion


class LatentInterceptor:
    """
    Simple class to intercept latents from StreamDiffusion
    """
    def __init__(self):
        self.intercepted_latents = None
        self.step_count = 0

    def reset(self):
        self.intercepted_latents = None
        self.step_count = 0


class StreamDiffusionControlNetXSViaInterception:
    """
    StreamDiffusion + ControlNetXS using latent interception approach

    Tests StreamDiffusion's noise scheduling with ControlNetXS by:
    1. StreamDiffusion processes input image with its own optimizations
    2. Intercept the noisy latents before they go into StreamDiffusion's UNet
    3. Feed those latents to ControlNetXS pipeline for controlled generation
    """

    def __init__(self, device="cuda"):
        self.device = device
        self.interceptor = LatentInterceptor()
        self.setup_components()

    def setup_components(self):
        """Initialize StreamDiffusion and ControlNetXS pipeline separately"""
        print("Setting up StreamDiffusion...")

        # 1. Load base SDXL pipeline for StreamDiffusion
        base_pipeline = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/sdxl-turbo",
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16"
        ).to(self.device)

        # Use TinyVAE for efficiency
        print("Loading TinyVAE...")
        tiny_vae = AutoencoderTiny.from_pretrained(
            "madebyollin/taesdxl",
            torch_dtype=torch.float16
        ).to(self.device)
        base_pipeline.vae = tiny_vae

        # 2. Initialize StreamDiffusion
        self.stream_diffusion = StreamDiffusion(
            pipe=base_pipeline,
            t_index_list=[0],  # Single timestep for img2img-like behavior
            torch_dtype=torch.float16,
            width=512,
            height=512,
            do_add_noise=True,
            frame_buffer_size=1,
            cfg_type="none",  # No guidance for latent prep
            device=self.device,
            original_inference_steps=50
        )

        # Store original UNet forward for interception
        self.original_stream_unet_forward = self.stream_diffusion.unet.forward

        print("Setting up ControlNetXS pipeline...")

        # 3. Load ControlNetXS adapter
        controlnet = ControlNetXSAdapter.from_pretrained(
            "UmerHA/Testing-ConrolNetXS-SDXL-canny",
            torch_dtype=torch.float16,
            use_safetensors=True
        ).to(self.device)

        # 4. Load ControlNetXS pipeline (separate)
        self.controlnetxs_pipeline = StableDiffusionXLControlNetXSPipeline.from_pretrained(
            "stabilityai/sdxl-turbo",
            controlnet=controlnet,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16"
        ).to(self.device)

        # Configure scheduler for SDXL-Turbo
        self.controlnetxs_pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
            self.controlnetxs_pipeline.scheduler.config,
            timestep_spacing="trailing"
        )

        # Use same TinyVAE
        self.controlnetxs_pipeline.vae = tiny_vae

        print("Components ready!")
        print(f"Using device: {self.device}")

    def _intercepting_unet_forward(self, *args, **kwargs):
        """
        Monkey patched UNet forward that intercepts the latents from StreamDiffusion
        """
        # Get the noisy latent (first argument to UNet)
        if len(args) > 0:
            latent = args[0]
            if self.interceptor.step_count == 0:  # Only capture first step latents
                self.interceptor.intercepted_latents = latent.clone()
                print(f"Intercepted StreamDiffusion latents: {latent.shape}")
            self.interceptor.step_count += 1

        # Call original UNet (but we'll ignore the result)
        return self.original_stream_unet_forward(*args, **kwargs)

    def generate_control_image(self, input_image, control_type="canny"):
        """Generate control image from input image"""
        if control_type == "canny":
            # Convert PIL to numpy
            img_array = np.array(input_image)

            # Apply Canny edge detection with lower thresholds for more edges
            edges = cv2.Canny(img_array, 30, 120)

            # Convert back to 3-channel PIL image
            edges_3ch = np.stack([edges] * 3, axis=-1)
            control_image = Image.fromarray(edges_3ch)

            return control_image
        else:
            raise ValueError(f"Control type {control_type} not supported")

    def generate_latents_with_stream_diffusion(self, input_image, prompt, strength=1.0):
        """
        Use StreamDiffusion to process input image and intercept the noisy latents

        Args:
            input_image: PIL Image (512x512)
            prompt: Text prompt for StreamDiffusion
            strength: Controls how much processing StreamDiffusion applies (0.0-1.0)
        """
        # Convert PIL to tensor [0, 1] with correct dtype
        img_array = np.array(input_image)
        img_tensor = torch.from_numpy(img_array).float() / 255.0
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)

        # Ensure correct dtype and device
        img_tensor = img_tensor.to(device=self.device, dtype=torch.float16)

        # Set up StreamDiffusion for this generation
        self.stream_diffusion.update_prompt(prompt)
        self.stream_diffusion.set_noise(seed=42)

        # set timestep
        t_index_list = [int(50 - 50 * strength)]

        print(f"Setting StreamDiffusion timesteps: {t_index_list} (strength {strength})")
        self.stream_diffusion.set_timesteps(t_index_list)

        # Reset interceptor
        self.interceptor.reset()

        # Monkey patch StreamDiffusion's UNet to intercept latents
        self.stream_diffusion.unet.forward = self._intercepting_unet_forward

        try:
            # Run StreamDiffusion (we don't care about output, just want the latents)
            with torch.no_grad():
                _ = self.stream_diffusion(
                    input=img_tensor,
                    encode_input=True,
                    decode_output=True
                )

        finally:
            # Restore original UNet
            self.stream_diffusion.unet.forward = self.original_stream_unet_forward

        # Check if we intercepted latents
        if self.interceptor.intercepted_latents is None:
            raise RuntimeError("Failed to intercept latents from StreamDiffusion")

        return self.interceptor.intercepted_latents.clone()


def test_stream_diffusion_interception():
    """Test the StreamDiffusion + ControlNetXS interception approach with parameter matrix"""
    print("=== Testing StreamDiffusion + ControlNetXS via Latent Interception ===")

    # Auto-detect device
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    # Initialize pipeline
    pipeline = StreamDiffusionControlNetXSViaInterception(device=device)

    # Load input image from desktop
    input_image_path = "/Users/himmelroman/Desktop/images/sample.jpg"
    try:
        test_img = Image.open(input_image_path).convert('RGB')
        print(f"✅ Loaded input image: {input_image_path}")
        print(f"Original size: {test_img.size}")
    except Exception as e:
        print(f"❌ Failed to load image from {input_image_path}: {e}")
        print("Creating fallback test image...")
        # Fallback to generated image if file not found
        test_img = Image.new('RGB', (512, 512), color='lightblue')
        from PIL import ImageDraw
        draw = ImageDraw.Draw(test_img)
        draw.rectangle([50, 50, 200, 200], outline='darkblue', width=3)
        draw.ellipse([300, 100, 450, 250], outline='navy', width=3)
        draw.line([100, 300, 400, 400], fill='darkblue', width=5)
        draw.line([400, 300, 100, 400], fill='darkblue', width=5)

    # Test generation parameters
    prompt = "a beautiful mountain landscape with a lake"

    # Parameter matrix to test
    stream_strengths = [0.2, 0.4, 0.6, 0.8]  # StreamDiffusion processing strength
    control_strengths = [0.2, 0.4, 0.6, 0.8]

    print(f"\n=== TESTING PARAMETER MATRIX ===")
    print(f"StreamDiffusion strengths: {stream_strengths}")
    print(f"Control strengths: {control_strengths}")
    print(f"Total combinations: {len(stream_strengths) * len(control_strengths)}")
    print(f"Optimization: Only {len(stream_strengths)} StreamDiffusion runs needed (cached per strength)")

    # Create output directory
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    # Save input image (resized to 512x512 for consistency)
    test_img_resized = test_img.resize((512, 512))
    test_img_resized.save(f"{output_dir}/stream_input_image.png")
    print(f"Saved input image: {output_dir}/stream_input_image.png")

    # Generate and save control image once
    control_image = pipeline.generate_control_image(test_img_resized, "canny")
    control_image.save(f"{output_dir}/stream_control_image_canny_30_120.png")
    print(f"Saved control image: {output_dir}/stream_control_image_canny_30_120.png")

    # PHASE 1: Generate latents with StreamDiffusion for each strength
    print(f"\n=== PHASE 1: GENERATING LATENTS FOR EACH STREAMDIFFUSION STRENGTH ===")
    cached_latents = {}

    for i, stream_strength in enumerate(stream_strengths):
        print(f"\n--- Generating latents {i+1}/{len(stream_strengths)} for StreamDiffusion strength {stream_strength} ---")

        try:
            cached_latents[stream_strength] = pipeline.generate_latents_with_stream_diffusion(
                test_img_resized,
                prompt,
                strength=stream_strength
            )
            print(f"✅ Cached latents for StreamDiffusion strength {stream_strength}: {cached_latents[stream_strength].shape}")

        except Exception as e:
            print(f"❌ Failed to generate latents with StreamDiffusion strength {stream_strength}: {e}")
            continue

    print(f"\n✅ Cached latents for {len(cached_latents)} StreamDiffusion strengths")

    # PHASE 2: Generate all combinations using cached latents
    print(f"\n=== PHASE 2: GENERATING COMBINATIONS USING CACHED LATENTS ===")
    total_combinations = len(stream_strengths) * len(control_strengths)
    current_combination = 0

    for stream_strength in stream_strengths:
        if stream_strength not in cached_latents:
            print(f"⚠️ Skipping StreamDiffusion strength {stream_strength} - no cached latents")
            continue

        for control_strength in control_strengths:
            current_combination += 1

            print(f"\n--- Combination {current_combination}/{total_combinations} ---")
            print(f"StreamDiffusion strength: {stream_strength}, Control strength: {control_strength}")
            print(f"Using cached latents: {cached_latents[stream_strength].shape}")

            try:
                # Set seed for consistency
                torch.manual_seed(42)

                # Use cached latents directly in ControlNetXS pipeline
                with torch.no_grad():
                    result = pipeline.controlnetxs_pipeline(
                        prompt=prompt,
                        image=control_image,  # Control image for ControlNetXS
                        negative_prompt="blurry, ugly, low quality",
                        latents=cached_latents[stream_strength],  # Use cached latents!
                        controlnet_conditioning_scale=control_strength,
                        num_inference_steps=4,
                        guidance_scale=0.0,
                        num_images_per_prompt=1,
                        return_dict=True,
                    )

                generated_image = result.images[0]

                # Create descriptive filename
                filename = f"stream_{stream_strength:.1f}_control_{control_strength:.1f}.png"
                filepath = f"{output_dir}/{filename}"
                generated_image.save(filepath)
                print(f"✅ Saved: {filename}")

            except Exception as e:
                print(f"❌ Failed combination stream={stream_strength}, control={control_strength}: {e}")
                continue

    print(f"\n=== STREAMDIFFUSION + CONTROLNETXS RESULTS ===")
    print(f"Input image: {output_dir}/stream_input_image.png")
    print(f"Control image: {output_dir}/stream_control_image_canny_30_120.png")
    print(f"Generated images saved with pattern: stream_X.X_control_Y.Y.png")

    # Create summary
    print(f"\n=== TEST RESULTS ===")
    print(f"✅ StreamDiffusion strength control: Uses timestep variation for different processing levels")
    print(f"✅ ControlNetXS compatibility: Tests if StreamDiffusion latents work with ControlNetXS")
    print(f"✅ Parameter combinations: {len(stream_strengths)} × {len(control_strengths)} = {total_combinations}")
    print(f"")
    print(f"StreamDiffusion Strength (timesteps/processing intensity):")
    print(f"  0.2 - Light processing, 10 timesteps")
    print(f"  0.4 - Moderate processing, 20 timesteps")
    print(f"  0.6 - Strong processing, 30 timesteps")
    print(f"  0.8 - Heavy processing, 40 timesteps")
    print(f"")
    print(f"Control Strength (how much ControlNetXS influences generation):")
    print(f"  0.2 - Light control influence, edges loosely followed")
    print(f"  0.4 - Moderate control influence")
    print(f"  0.6 - Strong control influence")
    print(f"  0.8 - Very strong control influence, edges strictly followed")
    print(f"")
    print(f"This tests whether StreamDiffusion's timestep-controlled latent preparation")
    print(f"is compatible with ControlNetXS for controlled generation.")
    print(f"")
    print(f"OPTIMIZATION: Used cached latents - only {len(stream_strengths)} StreamDiffusion runs instead of {total_combinations}")

    # List all generated files
    print(f"\n=== GENERATED FILES ===")
    for stream_strength in stream_strengths:
        for control_strength in control_strengths:
            filename = f"stream_{stream_strength:.1f}_control_{control_strength:.1f}.png"
            print(f"  {filename}")


if __name__ == "__main__":
    test_stream_diffusion_interception()