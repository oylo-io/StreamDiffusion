import torch
from kornia.filters import canny
# from transformers import pipeline
# from controlnet_aux import OpenposeDetector


class CannyFeatureExtractor:

    def __init__(self, device, low_threshold=0.05, high_threshold=0.15):

        self.device = device
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    @torch.inference_mode()
    def generate(self, image_tensor):

        # Apply Canny edge detection
        edges, _ = canny(
            image_tensor,
            hysteresis=False,
            low_threshold=self.low_threshold,
            high_threshold=self.high_threshold
        )

        # Convert to 3-channel format for the adapter
        edges_3ch = torch.cat([edges, edges, edges], dim=1)

        return edges_3ch


# class DepthFeatureExtractor:
#     def __init__(self, device, model_id="LiheYoung/depth-anything-base-hf"):
#
#         self.device = device
#         self.model_id = model_id
#
#         # load depth estimator
#         self.depth_estimator = pipeline(
#             task="depth-estimation",
#             model=model_id,
#             device=device
#         )
#
#     def generate(self, image):
#         depth_map = self.depth_estimator(image)["depth"]
#         return depth_map
#
#
# class PoseFeatureExtractor:
#     def __init__(self, device):
#         self.device = device
#         self.detector = OpenposeDetector.from_pretrained('lllyasviel/Annotators')
#
#     def generate(self, image):
#         pose_image = self.detector(image, include_hands=True, include_face=True)  # Include hands and face for better control
#         return pose_image
