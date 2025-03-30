import argparse
from pathlib import Path

from streamdiffusion.acceleration.tensorrt.models import UNetXLTurboIPAdapter
from streamdiffusion.acceleration.tensorrt.utilities import build_engine

def build(onnx_path, engine_path, height, width, batch_size):

    model = UNetXLTurboIPAdapter()

    build_engine(
        onnx_opt_path=onnx_path,
        engine_path=engine_path,
        model_data=model,
        opt_image_height=height,
        opt_image_width=width,
        opt_batch_size=batch_size,
        build_static_batch=True,
        build_dynamic_shape=False,
        build_all_tactics=False,
        build_enable_refit=False
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Accelerate Pipeline with TRT")
    parser.add_argument('--onnx_path', type=Path, required=True)
    parser.add_argument('--engine_path', type=Path, required=True)
    parser.add_argument('--height', type=int, required=True, help='image height')
    parser.add_argument('--width', type=int, required=True, help='image width')
    parser.add_argument('--batch_size', type=int, default=1, help='number of timesteps (batch size)')
    parser.add_argument('--sdxl', default=False, action='store_true')
    parser.add_argument('--ip_adapter', default=False, action='store_true')

    args = parser.parse_args()

    build(
        args.onnx_path,
        args.engine_path,
        args.height,
        args.width,
        args.num_timesteps
    )