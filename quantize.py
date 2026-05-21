"""
Apply dynamic INT8 and static INT8 quantization to an ONNX model.
Dynamic: weights only — fast, no calibration data needed.
Static : weights + activations — slower to prepare, faster at inference.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import onnx
from onnxruntime.quantization import (
    QuantType, quantize_dynamic, quantize_static, CalibrationDataReader,
)


class RandomCalibrationReader(CalibrationDataReader):
    """Stand-in for real calibration data. For production, replace with a
    reader that yields representative samples from your validation set."""

    def __init__(self, n_samples: int = 64, input_name: str = "input"):
        self.input_name = input_name
        self.data = iter(
            np.random.randn(1, 3, 224, 224).astype(np.float32)
            for _ in range(n_samples)
        )

    def get_next(self):
        try:
            return {self.input_name: next(self.data)}
        except StopIteration:
            return None


def quantize(fp32_path: Path, out_dir: Path):
    name = fp32_path.stem.replace("_fp32", "")
    out_dir.mkdir(parents=True, exist_ok=True)

    dyn_path = out_dir / f"{name}_int8_dynamic.onnx"
    print(f"Dynamic INT8 → {dyn_path}")
    quantize_dynamic(
        model_input=fp32_path,
        model_output=dyn_path,
        weight_type=QuantType.QInt8,
    )
    print(f"  size: {os.path.getsize(dyn_path) / 1e6:.2f} MB")

    stat_path = out_dir / f"{name}_int8_static.onnx"
    print(f"Static INT8 → {stat_path}")
    model = onnx.load(str(fp32_path))
    input_name = model.graph.input[0].name
    quantize_static(
        model_input=fp32_path,
        model_output=stat_path,
        calibration_data_reader=RandomCalibrationReader(input_name=input_name),
        quant_format="QDQ",
        per_channel=True,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QUInt8,
    )
    print(f"  size: {os.path.getsize(stat_path) / 1e6:.2f} MB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("fp32_model")
    ap.add_argument("--out", default="models")
    args = ap.parse_args()
    quantize(Path(args.fp32_model), Path(args.out))
