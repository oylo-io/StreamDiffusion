import argparse
from pathlib import Path

import onnx
from streamdiffusion.acceleration.tensorrt.optimizer import Optimizer


def optimize_model(model_path, output_path):

    # load model
    graph = onnx.load(model_path)

    # optimize
    opt = Optimizer(graph)
    opt_graph = opt.optimize()

    # save
    onnx.save(opt_graph, output_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Accelerate Pipeline with TRT")
    parser.add_argument('--model_path', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, required=True)
    args = parser.parse_args()

    # verify dir
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    optimize_model(
        args.model_path,
        str(args.output_dir / 'model.onnx')
    )
