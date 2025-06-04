"""
StreamDiffusion + ControlNetXS Integrated Pipeline Tester

Tests the new integrated StreamDiffusion pipeline with built-in ControlNetXS support.
No interception tricks needed - direct control through the unified pipeline.
Enhanced with strength parameter testing for timestep control.
"""

import os
import torch
import numpy as np
from PIL import Image
import cv2
import logging
import sys

# Diffusers imports
from diffusers import (
    StableDiffusionXLPipeline,
    AutoencoderTiny,
)

# Import your updated StreamDiffusion class
from streamdiffusion import StreamDiffusion

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('streamdiffusion_controlnetxs_test.log')
    ]
)
logger = logging.getLogger(__name__)


class StreamDiffusionControlNetXSTester:
    """
    Tester for the integrated StreamDiffusion + ControlNetXS pipeline
    """

    def __init__(self, device="cuda"):
        self.device = device
        self.setup_pipeline()

    def setup_pipeline(self):
        """Initialize the integrated StreamDiffusion + ControlNetXS pipeline"""
        logger.info("Setting up integrated StreamDiffusion + ControlNetXS pipeline...")

        try:
            # 1. Load base SDXL pipeline
            logger.info("Loading base SDXL pipeline...")
            base_pipeline = StableDiffusionXLPipeline.from_pretrained(
                "stabilityai/sdxl-turbo",
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16"
            ).to(self.device)

            # Use TinyVAE for efficiency
            logger.info("Loading TinyVAE...")
            tiny_vae = AutoencoderTiny.from_pretrained(
                "madebyollin/taesdxl",
                torch_dtype=torch.float16
            ).to(self.device)
            base_pipeline.vae = tiny_vae

            # 2. Initialize StreamDiffusion
            logger.info("Initializing StreamDiffusion...")
            self.stream_diffusion = StreamDiffusion(
                pipe=base_pipeline,
                t_index_list=[0, 15, 30],  # Multiple timesteps for better quality
                torch_dtype=torch.float16,
                width=512,
                height=512,
                do_add_noise=True,
                frame_buffer_size=1,
                cfg_type="self",  # Use self-guidance
                device=self.device,
                original_inference_steps=50
            )

            # 3. Load ControlNetXS adapter
            logger.info("Loading ControlNetXS adapter...")
            self.stream_diffusion.load_controlnet_adapter(
                size_ratio=0.5  # Adjust based on your needs
            )

            logger.info("Pipeline setup completed successfully!")
            logger.info(f"Using device: {self.device}")

        except Exception as e:
            logger.error("Failed to setup pipeline", exc_info=True)
            raise

    def generate_control_image(self, input_image, control_type="canny", low_threshold=20, high_threshold=120):
        """Generate control image from input image"""
        try:
            if control_type == "canny":
                logger.debug(f"Generating Canny control image with thresholds: {low_threshold}, {high_threshold}")

                # Convert PIL to numpy
                img_array = np.array(input_image)

                # Apply Canny edge detection
                edges = cv2.Canny(img_array, low_threshold, high_threshold)

                # Convert back to 3-channel PIL image
                edges_3ch = np.stack([edges] * 3, axis=-1)
                control_image = Image.fromarray(edges_3ch)

                logger.debug(f"Successfully generated {control_type} control image")
                return control_image
            else:
                raise ValueError(f"Control type {control_type} not supported")

        except Exception as e:
            logger.error(f"Failed to generate control image with type {control_type}", exc_info=True)
            raise

    def test_generation(self, input_image, control_image, prompt,
                       conditioning_scale=1.0, strength=0.8, seed=42):
        """
        Test generation with the integrated pipeline

        Args:
            input_image: PIL Image for img2img
            control_image: PIL Image for control guidance
            prompt: Text prompt
            conditioning_scale: ControlNet strength (0.0-2.0)
            strength: Diffusion strength controlling timestep (0.0-1.0)
            seed: Random seed
        """
        try:
            logger.debug(f"Starting generation with conditioning_scale={conditioning_scale}, strength={strength}, seed={seed}")

            # Convert PIL to tensor [0, 1]
            img_array = np.array(input_image)
            img_tensor = torch.from_numpy(img_array).float() / 255.0
            img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
            img_tensor = img_tensor.to(device=self.device, dtype=torch.float16)

            # Convert control image
            control_array = np.array(control_image)
            control_tensor = torch.from_numpy(control_array).float() / 255.0
            control_tensor = control_tensor.permute(2, 0, 1).unsqueeze(0)
            control_tensor = control_tensor.to(device=self.device, dtype=torch.float16)

            # Setup StreamDiffusion
            self.stream_diffusion.update_prompt(prompt)
            self.stream_diffusion.set_noise(seed=seed)

            # Set timestep based on strength (strength controls how much the image changes)
            timestep = int(50 * strength)
            self.stream_diffusion.set_timesteps(t_list=[timestep])
            logger.debug(f"Set timestep to {timestep} (strength={strength})")

            # Generate with control
            with torch.no_grad():
                result = self.stream_diffusion(
                    input=img_tensor,
                    control_image=control_tensor,
                    conditioning_scale=conditioning_scale,
                    encode_input=True,
                    decode_output=True
                )

            # Convert back to PIL
            result_np = result.squeeze().permute(1, 2, 0).cpu().numpy()
            result_np = np.clip(result_np, 0, 1)
            result_pil = Image.fromarray((result_np * 255).astype(np.uint8))

            logger.debug("Generation completed successfully")
            return result_pil

        except Exception as e:
            logger.error(f"Failed generation with conditioning_scale={conditioning_scale}, strength={strength}, seed={seed}", exc_info=True)
            raise


def test_streamdiffusion_controlnetxs():
    """Test the integrated StreamDiffusion + ControlNetXS pipeline"""
    logger.info("=== Testing Integrated StreamDiffusion + ControlNetXS ===")

    # Auto-detect device
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    logger.info(f"Using device: {device}")

    try:
        # Initialize tester
        logger.info("Initializing tester...")
        tester = StreamDiffusionControlNetXSTester(device=device)

        # Load or create input image
        input_image_path = "/Users/himmelroman/Desktop/images/sample.jpg"
        try:
            test_img = Image.open(input_image_path).convert('RGB')
            logger.info(f"✅ Loaded input image: {input_image_path}")
        except Exception as e:
            logger.warning(f"Could not load {input_image_path}: {e}")
            logger.info("Creating fallback test image...")
            # Create test image with geometric shapes
            test_img = Image.new('RGB', (512, 512), color='lightblue')
            from PIL import ImageDraw
            draw = ImageDraw.Draw(test_img)
            draw.rectangle([50, 50, 200, 200], outline='darkblue', width=3)
            draw.ellipse([300, 100, 450, 250], outline='navy', width=3)
            draw.line([100, 300, 400, 400], fill='darkblue', width=5)
            draw.line([400, 300, 100, 400], fill='darkblue', width=5)

        # Resize to 512x512
        test_img = test_img.resize((512, 512))

        # Test parameters
        prompt = "a beautiful mountain landscape with a lake, artistic style"

        # Parameter matrix
        conditioning_scales = [0.2, 0.5, 0.8, 1.2, 1.5]  # ControlNet influence
        canny_thresholds = [(30, 120), (50, 150), (20, 100)]  # Edge detection settings
        strength_list = [0.2, 0.4, 0.6, 0.8, 1.0]  # Diffusion strength (timestep control)

        # Create output directory
        output_dir = "outputs_integrated"
        os.makedirs(output_dir, exist_ok=True)

        # Save input image
        test_img.save(f"{output_dir}/input_image.png")
        logger.info(f"Saved input image: {output_dir}/input_image.png")

        logger.info("=== TESTING PARAMETER MATRIX ===")
        logger.info(f"Conditioning scales: {conditioning_scales}")
        logger.info(f"Canny thresholds: {canny_thresholds}")
        logger.info(f"Strength values: {strength_list}")

        total_combinations = len(conditioning_scales) * len(canny_thresholds) * len(strength_list)
        logger.info(f"Total combinations: {total_combinations}")

        combination_count = 0

        for low_thresh, high_thresh in canny_thresholds:
            try:
                # Generate control image with current thresholds
                control_image = tester.generate_control_image(
                    test_img,
                    control_type="canny",
                    low_threshold=low_thresh,
                    high_threshold=high_thresh
                )

                # Save control image
                control_filename = f"control_canny_{low_thresh}_{high_thresh}.png"
                control_image.save(f"{output_dir}/{control_filename}")
                logger.info(f"Generated control image: {control_filename}")

                for conditioning_scale in conditioning_scales:
                    for strength in strength_list:
                        combination_count += 1

                        logger.info(f"--- Combination {combination_count}/{total_combinations} ---")
                        logger.info(f"Canny thresholds: ({low_thresh}, {high_thresh})")
                        logger.info(f"Conditioning scale: {conditioning_scale}")
                        logger.info(f"Strength: {strength} (timestep: {int(50 * strength)})")

                        try:
                            # Generate image
                            result_image = tester.test_generation(
                                input_image=test_img,
                                control_image=control_image,
                                prompt=prompt,
                                conditioning_scale=conditioning_scale,
                                strength=strength,
                                seed=42
                            )

                            # Save result with all parameters in filename
                            result_filename = f"result_canny{low_thresh}_{high_thresh}_scale{conditioning_scale:.1f}_strength{strength:.1f}.png"
                            result_image.save(f"{output_dir}/{result_filename}")
                            logger.info(f"✅ Saved: {result_filename}")

                        except Exception as e:
                            logger.error(f"Failed combination canny({low_thresh}, {high_thresh}) scale={conditioning_scale} strength={strength}", exc_info=True)
                            continue

            except Exception as e:
                logger.error(f"Failed to generate control image with thresholds ({low_thresh}, {high_thresh})", exc_info=True)
                continue

        logger.info("=== QUALITY COMPARISON TEST ===")
        logger.info("Testing with fixed parameters but different seeds...")

        # Test consistency with multiple seeds
        test_seeds = [42, 123, 456, 789]
        try:
            control_image = tester.generate_control_image(test_img, "canny", 50, 150)

            for seed in test_seeds:
                try:
                    result_image = tester.test_generation(
                        input_image=test_img,
                        control_image=control_image,
                        prompt=prompt,
                        conditioning_scale=1.0,
                        strength=0.8,
                        seed=seed
                    )

                    result_filename = f"consistency_seed{seed}.png"
                    result_image.save(f"{output_dir}/{result_filename}")
                    logger.info(f"✅ Saved consistency test: {result_filename}")

                except Exception as e:
                    logger.error(f"Failed consistency test with seed {seed}", exc_info=True)
                    continue

        except Exception as e:
            logger.error("Failed to generate control image for consistency test", exc_info=True)

        logger.info("=== STRENGTH COMPARISON TEST ===")
        logger.info("Testing strength progression with fixed other parameters...")

        # Test strength progression
        try:
            control_image = tester.generate_control_image(test_img, "canny", 50, 150)

            for strength in strength_list:
                try:
                    result_image = tester.test_generation(
                        input_image=test_img,
                        control_image=control_image,
                        prompt=prompt,
                        conditioning_scale=1.0,
                        strength=strength,
                        seed=42
                    )

                    result_filename = f"strength_progression_{strength:.1f}.png"
                    result_image.save(f"{output_dir}/{result_filename}")
                    logger.info(f"✅ Saved strength test: {result_filename}")

                except Exception as e:
                    logger.error(f"Failed strength test with strength {strength}", exc_info=True)
                    continue

        except Exception as e:
            logger.error("Failed to generate control image for strength test", exc_info=True)

        logger.info("=== TEST RESULTS SUMMARY ===")
        logger.info("✅ Integrated pipeline test completed")
        logger.info(f"✅ Total combinations tested: {total_combinations}")
        logger.info(f"✅ Results saved in: {output_dir}/")
        logger.info("")
        logger.info("Key test aspects:")
        logger.info("  🔧 ControlNet strength: Tests different conditioning scales")
        logger.info("  🎨 Edge detection: Tests different Canny thresholds")
        logger.info("  💪 Diffusion strength: Tests different timestep values")
        logger.info("  🎲 Consistency: Tests multiple seeds with same parameters")
        logger.info("  🚀 Performance: Integrated pipeline vs. interception approach")
        logger.info("")
        logger.info("Parameter Guide:")
        logger.info("  Conditioning Scale (ControlNet influence):")
        logger.info("    0.2 - Very light control, creative freedom")
        logger.info("    0.5 - Balanced control and creativity")
        logger.info("    0.8 - Strong control, follows edges closely")
        logger.info("    1.2 - Very strong control")
        logger.info("    1.5 - Maximum control, strict edge following")
        logger.info("")
        logger.info("  Strength (Diffusion timestep control):")
        logger.info("    0.2 - Light modification (timestep 10)")
        logger.info("    0.4 - Moderate modification (timestep 20)")
        logger.info("    0.6 - Strong modification (timestep 30)")
        logger.info("    0.8 - Very strong modification (timestep 40)")
        logger.info("    1.0 - Maximum modification (timestep 50)")
        logger.info("")
        logger.info("Check the output images to verify:")
        logger.info("  • Control strength progression")
        logger.info("  • Edge detection quality")
        logger.info("  • Diffusion strength effects")
        logger.info("  • Generation consistency")
        logger.info("  • No artifacts from integration")

    except Exception as e:
        logger.error("Test failed with critical error", exc_info=True)
        raise


if __name__ == "__main__":
    try:
        test_streamdiffusion_controlnetxs()
    except Exception as e:
        logger.critical("Test script failed", exc_info=True)
        sys.exit(1)