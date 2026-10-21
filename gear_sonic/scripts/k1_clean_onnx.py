"""Remove debug strings (source file paths of the exporting machine) from exported ONNX files and rename them.

torch.onnx.export(verbose=True) stores the exporting script's absolute paths in node doc_strings.
Usage: python gear_sonic/scripts/k1_clean_onnx.py <exported_dir> <out_dir> [step]
  <exported_dir>/model_step_<step>_{encoder,decoder,g1,teleop}.onnx ->
  <out_dir>/{model_encoder,model_decoder,policy_g1,policy_teleop}.onnx
"""

import os
import sys

import onnx


def main(src: str, dst: str, step: str = "050000"):
    os.makedirs(dst, exist_ok=True)
    for a, b in (("encoder", "model_encoder"), ("decoder", "model_decoder"), ("g1", "policy_g1"),
                 ("teleop", "policy_teleop")):
        m = onnx.load(os.path.join(src, f"model_step_{step}_{a}.onnx"))
        m.doc_string = ""
        m.graph.doc_string = ""
        for node in m.graph.node:
            node.doc_string = ""
        del m.metadata_props[:]
        onnx.checker.check_model(m)
        onnx.save(m, os.path.join(dst, f"{b}.onnx"))
        print(f"{b}.onnx  in {[[d.dim_value for d in i.type.tensor_type.shape.dim] for i in m.graph.input]}"
              f"  out {[[d.dim_value for d in o.type.tensor_type.shape.dim] for o in m.graph.output]}")


if __name__ == "__main__":
    main(*sys.argv[1:])
