"""
Export a pretrained vision model to ONNX (FP32) and an FP16 variant.
We use torchvision's MobileNetV3-Small — small enough to run anywhere,
relevant enough for real edge deployment.
"""

import argparse
import os
from pathlib import Path

import torch
from torchvision import models


def export(model_name: str, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    if model_name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    elif model_name == "resnet18":
        model = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1)
    else:
        raise ValueError(f"unknown model: {model_name}")

    model.eval()
    dummy = torch.randn(1, 3, 224, 224)

    fp32_path = out_dir / f"{model_name}_fp32.onnx"
    print(f"Exporting FP32 → {fp32_path}")
    torch.onnx.export(
        model, dummy, fp32_path,
        export_params=True, opset_version=17,
        do_constant_folding=True,
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    )
    print(f"  size: {os.path.getsize(fp32_path) / 1e6:.2f} MB")

    # FP16 conversion via onnxconverter-common
    try:
        from onnxconverter_common import float16
        import onnx
        fp16_path = out_dir / f"{model_name}_fp16.onnx"
        model_fp32 = onnx.load(fp32_path)
        model_fp16 = float16.convert_float_to_float16(model_fp32)
        onnx.save(model_fp16, fp16_path)
        print(f"Exported FP16 → {fp16_path}")
        print(f"  size: {os.path.getsize(fp16_path) / 1e6:.2f} MB")
    except ImportError:
        print("Skipping FP16 (install onnxconverter-common to enable)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mobilenet_v3_small",
                    choices=["mobilenet_v3_small", "resnet18"])
    ap.add_argument("--out",   default="models")
    args = ap.parse_args()
    export(args.model, Path(args.out))
