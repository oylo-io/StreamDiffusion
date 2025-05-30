from typing import Optional, Literal

import torch
from kornia.filters import canny
from kornia.enhance import adjust_contrast
from transformers import pipeline
# from controlnet_aux import OpenposeDetector


class CannyFeatureExtractor:

    def __init__(self, device, low_threshold=0.1, high_threshold=0.2, contrast_factor=1.25):

        self.device = device
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.contrast_factor = contrast_factor

    @torch.inference_mode()
    def generate(self, image_tensor):

        # increase contrast
        contrasted = adjust_contrast(image_tensor, self.contrast_factor)

        # Apply Canny edge detection
        edges, _ = canny(
            contrasted,
            hysteresis=False,
            low_threshold=self.low_threshold,
            high_threshold=self.high_threshold
        )

        # Convert to 3-channel format for the adapter
        edges_3ch = torch.cat([edges, edges, edges], dim=1)

        return edges_3ch


class DepthFeatureExtractor:
    """
    Simple depth feature extractor that wraps any depth model.
    The underlying model can be PyTorch or TensorRT - this class doesn't care.
    """

    def __init__(
            self,
            device: str = "cuda",
            variant: Literal["small", "base", "large"] = "small",
            model_id: Optional[str] = None,
            model=None,
            use_fp16: bool = True,
            output_channels: int = 3
    ):
        """
        Initialize the depth feature extractor.

        Args:
            device: Device to run inference on
            variant: Model variant ('small', 'base', 'large')
            model_id: Custom model ID, overrides variant
            model: Pre-loaded model (if provided, skips loading)
            use_fp16: Use FP16 precision
            output_channels: Number of output channels (1 or 3)
        """
        self.device = device
        self.variant = variant
        self.use_fp16 = use_fp16
        self.output_channels = output_channels

        if model is not None:
            # Use provided model
            self.model = model
        else:
            # Model selection
            v2_model_map = {
                'small': 'depth-anything/Depth-Anything-V2-Small-hf',
                'base': 'depth-anything/Depth-Anything-V2-Base-hf',
                'large': 'depth-anything/Depth-Anything-V2-Large-hf'
            }

            if model_id:
                self.model_id = model_id
            elif variant in v2_model_map:
                self.model_id = v2_model_map[variant]
            else:
                raise ValueError(f"Unknown variant: {variant}")

            # Load the default PyTorch model
            self._load_model()

    def _load_model(self):
        """Load the default PyTorch model"""
        dtype = torch.float16 if self.use_fp16 else torch.float32

        print(f'Loading Depth-Anything V2: {self.model_id} (variant: {self.variant})')

        # Load depth estimation pipeline
        depth_pipeline = pipeline(
            task="depth-estimation",
            model=self.model_id,
            torch_dtype=dtype,
            device=self.device if self.device != "mps" else 0
        )

        # Store just the model
        self.model = depth_pipeline.model.to(dtype=dtype, device=self.device)
        self.model.eval()

    @torch.inference_mode()
    def generate(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Generate depth map from input image tensor.

        Args:
            image_tensor: Input image tensor [B, C, H, W] in range [0, 1]

        Returns:
            Depth tensor [B, output_channels, H, W]
        """
        # Ensure tensor is on correct device
        image_tensor = image_tensor.to(self.device)

        # Run inference using whatever model is currently set
        depth_output = self.model(image_tensor)

        # Handle output format
        if isinstance(depth_output, dict):
            if 'depth' in depth_output:
                depth = depth_output['depth']
            elif 'predicted_depth' in depth_output:
                depth = depth_output['predicted_depth']
            else:
                depth = list(depth_output.values())[0]
        else:
            depth = depth_output

        # Ensure proper shape [B, 1, H, W]
        if len(depth.shape) == 3:
            depth = depth.unsqueeze(1)

        # Adjust channels if needed
        if self.output_channels == 3 and depth.shape[1] == 1:
            depth = torch.cat([depth, depth, depth], dim=1)
        elif self.output_channels == 1 and depth.shape[1] == 3:
            depth = depth.mean(dim=1, keepdim=True)

        return depth

    def __call__(self, *args, **kwargs):
        """Make the class callable"""
        return self.generate(*args, **kwargs)


# class PoseFeatureExtractor:
#     def __init__(self, device):
#         self.device = device
#         self.detector = OpenposeDetector.from_pretrained('lllyasviel/Annotators')
#
#     def generate(self, image):
#         pose_image = self.detector(image, include_hands=True, include_face=True)  # Include hands and face for better control
#         return pose_image
