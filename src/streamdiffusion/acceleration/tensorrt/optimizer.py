import onnx_graphsurgeon as gs
from onnx import shape_inference
from polygraphy.backend.onnx.loader import fold_constants


class Optimizer:
    def __init__(self, onnx_graph, verbose=False):
        self.graph = gs.import_onnx(onnx_graph)
        self.verbose = verbose

    def optimize(self):
        self.info("original")
        self.cleanup()
        self.info("cleanup")
        self.fold_constants()
        self.info("fold constants")
        self.infer_shapes()
        self.info("shape inference")
        onnx_opt_graph = self.cleanup(return_onnx=True)
        self.info("finished")

        return onnx_opt_graph

    def info(self, prefix):
        if self.verbose:
            print(
                f"{prefix} .. {len(self.graph.nodes)} nodes, {len(self.graph.tensors().keys())} tensors, {len(self.graph.inputs)} inputs, {len(self.graph.outputs)} outputs"
            )

    def cleanup(self, return_onnx=False):
        self.graph.cleanup().toposort()
        if return_onnx:
            return gs.export_onnx(self.graph)

    def select_outputs(self, keep, names=None):
        self.graph.outputs = [self.graph.outputs[o] for o in keep]
        if names:
            for i, name in enumerate(names):
                self.graph.outputs[i].name = name

    def fold_constants(self, return_onnx=False):
        onnx_graph = fold_constants(gs.export_onnx(self.graph), allow_onnxruntime_shape_inference=True)
        self.graph = gs.import_onnx(onnx_graph)
        if return_onnx:
            return onnx_graph

    def infer_shapes(self, return_onnx=False):
        onnx_graph = gs.export_onnx(self.graph)
        # if onnx_graph.ByteSize() > 2147483648:
        #     raise TypeError("ERROR: model size exceeds supported 2GB limit")
        # else:
        #     onnx_graph = shape_inference.infer_shapes(onnx_graph)

        onnx_graph = shape_inference.infer_shapes(onnx_graph)
        self.graph = gs.import_onnx(onnx_graph)
        if return_onnx:
            return onnx_graph
