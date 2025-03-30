import argparse
from pathlib import Path

import onnx
from streamdiffusion.acceleration.tensorrt.optimizer import Optimizer


def optimize_model(model_path, output_dir, do_fold_constants, do_infer_shapes):

    # load model
    print(f'Loading from {model_path}')
    graph = onnx.load(model_path)
    opt = Optimizer(graph)
    opt.info("original")

    # fold constants
    if do_fold_constants:
        opt.cleanup()
        opt.fold_constants()
        opt.cleanup()
        opt.info("fold constants")

        # save folded
        fold_graph = opt.export_onnx()
        fold_dir = Path(output_dir / 'folded')
        fold_dir.mkdir(parents=True, exist_ok=True)
        onnx.save(
            fold_graph,
            str(fold_dir / 'model.onnx'),
            save_as_external_data=True
        )
        print(f'Saved to {fold_dir}')

    # infer shapes
    if do_infer_shapes:
        opt.cleanup()
        opt.infer_shapes(strict_mode=True)
        opt.cleanup()
        opt.info("shape inference")

        # save inferred
        inf_graph = opt.export_onnx()
        inf_dir = Path(output_dir / 'inferred')
        inf_dir.mkdir(parents=True, exist_ok=True)
        onnx.save(
            inf_graph,
            str(inf_dir / 'model.onnx'),
            save_as_external_data=True
        )
        print(f'Saved to {inf_dir}')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Accelerate Pipeline with TRT")
    parser.add_argument('--model_path', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--do_fold_constants', default=False, action='store_true')
    parser.add_argument('--do_infer_shapes', default=False, action='store_true')
    args = parser.parse_args()

    # verify dir
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    optimize_model(
        args.model_path,
        args.output_dir,
        do_fold_constants=args.do_fold_constants,
        do_infer_shapes=args.do_infer_shapes
    )
