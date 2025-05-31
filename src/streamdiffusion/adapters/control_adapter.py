import math

import torch
import torch.nn.functional as F

from kornia.filters import canny
from kornia.enhance import adjust_contrast
from transformers import pipeline, AutoModelForDepthEstimation, AutoImageProcessor
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
    Fast tensor-to-tensor depth estimation using DepthAnything V2.

    Dynamically adapts to input sizes while respecting model architecture constraints.
    All preprocessing parameters loaded from model configuration.

    Input: torch.Tensor [B,C,H,W] or [C,H,W] in range [0,1]
    Output: torch.Tensor [B,1,H,W] depth map in range [0,255]
    """

    # Model variants
    models = {
        'small': 'depth-anything/Depth-Anything-V2-Small-hf',
        'base': 'depth-anything/Depth-Anything-V2-Base-hf',
        'large': 'depth-anything/Depth-Anything-V2-Large-hf'
    }

    def __init__(self, device="cuda", dtype=torch.float16, variant="small"):
        self.device = device
        self.dtype = dtype

        # Model mapping
        model_id = self.models[variant]

        # Load model
        self.model = AutoModelForDepthEstimation.from_pretrained(
            model_id,
            torch_dtype=self.dtype
        ).to(device=device, dtype=self.dtype)
        self.model.eval()

        # Load preprocessing configuration from the model's image processor
        print(f"Loading preprocessing config from {model_id}...")
        processor = AutoImageProcessor.from_pretrained(model_id)
        config = processor.to_dict()

        # Extract architecture parameters
        self.patch_size = getattr(self.model.config, 'patch_size', 14)
        print(f"Model patch size: {self.patch_size}")

        # Extract preprocessing parameters from official config
        self.rescale_factor = config.get('rescale_factor', 1.0 / 255.0)
        self.do_rescale = config.get('do_rescale', True)
        self.do_normalize = config.get('do_normalize', True)

        # Get normalization parameters from model config (not hard-coded!)
        image_std = config.get('image_std')     #, [0.229, 0.224, 0.225])
        image_mean = config.get('image_mean')   #, [0.485, 0.456, 0.406])

        print(f"Preprocessing config:")
        print(f"  - Rescale factor: {self.rescale_factor}")
        print(f"  - Normalize: {self.do_normalize}")
        print(f"  - Image mean: {image_mean}")
        print(f"  - Image std: {image_std}")

        # Convert to tensors for efficiency
        self.std = torch.tensor(image_std, dtype=dtype, device=device).view(1, 3, 1, 1)
        self.mean = torch.tensor(image_mean, dtype=dtype, device=device).view(1, 3, 1, 1)

    def _get_optimal_size(self, input_size):
        """
        Calculate optimal processing size based on input and model constraints.

        Args:
            input_size: (H, W) tuple of input dimensions

        Returns:
            (H, W) tuple of optimal processing size
        """
        height, width = input_size

        # Round up to nearest multiple of patch_size
        def round_up_to_multiple(value, multiple):
            return math.ceil(value / multiple) * multiple

        optimal_height = round_up_to_multiple(height, self.patch_size)
        optimal_width = round_up_to_multiple(width, self.patch_size)

        # For very small images, ensure minimum reasonable size
        min_size = self.patch_size * 16  # At least 16 patches per dimension
        optimal_height = max(optimal_height, min_size)
        optimal_width = max(optimal_width, min_size)

        return (optimal_height, optimal_width)

    def _preprocess(self, tensor):
        """
        Preprocess input tensor using dynamic sizing and model config.

        Steps:
        1. Ensure correct device/dtype
        2. Calculate optimal size based on input and model constraints
        3. Resize if needed
        4. Apply normalization from model config
        """
        # Store original size for output resizing
        original_size = (tensor.shape[2], tensor.shape[3])

        # Ensure correct device and dtype
        tensor = tensor.to(device=self.device, dtype=self.dtype)

        # Calculate optimal processing size
        optimal_size = self._get_optimal_size(original_size)

        # Resize only if necessary
        if original_size != optimal_size:
            print(f"Resizing from {original_size} to {optimal_size} (divisible by patch_size={self.patch_size})")
            tensor = F.interpolate(tensor, size=optimal_size, mode='bilinear', align_corners=False)
        else:
            print(f"Input size {original_size} already optimal for patch_size={self.patch_size}")

        # Apply normalization from model config
        if self.do_normalize:
            tensor = (tensor - self.mean) / self.std

        return tensor, original_size, optimal_size

    def _postprocess(self, depth_output, target_size, processed_size):
        """
        Convert model output to final depth map.

        Applies official DepthAnything normalization and resizes to target.
        """
        # Ensure proper tensor dimensions: [H,W] → [B,1,H,W]
        if depth_output.dim() == 2:
            depth_output = depth_output.unsqueeze(0).unsqueeze(0)
        elif depth_output.dim() == 3:
            depth_output = depth_output.unsqueeze(1)

        # Official DepthAnything normalization formula
        depth_normalized = depth_output * 255.0 / depth_output.max()

        # Resize back to target size if needed
        if target_size != processed_size:
            depth_normalized = F.interpolate(
                depth_normalized, size=target_size, mode='bilinear', align_corners=False
            )

        return torch.clamp(depth_normalized, 0, 255)

    @torch.no_grad()
    def generate(self, input_tensor, return_original_size=True):
        """
        Generate depth map from input tensor.

        Args:
            input_tensor: torch.Tensor [B,C,H,W] or [C,H,W] in range [0,1]
            return_original_size: If True, resize output to match input size

        Returns:
            torch.Tensor: Depth map [B,1,H,W] in range [0,255]
        """
        # Ensure batch dimension
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)

        # # Validate input range
        # if input_tensor.min() < 0 or input_tensor.max() > 1:
        #     raise ValueError(
        #         f"Input tensor must be in [0,1] range, got [{input_tensor.min():.3f}, {input_tensor.max():.3f}]")

        # Preprocess with dynamic sizing
        processed_tensor, original_size, processed_size = self._preprocess(input_tensor)

        # Model inference
        outputs = self.model(processed_tensor)
        depth_raw = outputs.predicted_depth

        # Postprocess
        target_size = original_size if return_original_size else processed_size
        depth_final = self._postprocess(depth_raw, target_size, processed_size)

        return depth_final

    def __call__(self, *args, **kwargs):
        return self.generate(*args, **kwargs)


# class PoseFeatureExtractor:
#     def __init__(self, device):
#         self.device = device
#         self.detector = OpenposeDetector.from_pretrained('lllyasviel/Annotators')
#
#     def generate(self, image):
#         pose_image = self.detector(image, include_hands=True, include_face=True)  # Include hands and face for better control
#         return pose_image
