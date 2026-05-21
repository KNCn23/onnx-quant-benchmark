# ONNX Quantization Benchmark

A reproducible pipeline that exports a pretrained vision model, applies multiple quantization strategies (FP16, INT8 dynamic, INT8 static), and benchmarks them head-to-head for **edge deployment**: latency, model size, and prediction agreement with the FP32 reference.

## What it answers

> "I have a pretrained model and 100ms of latency budget on an edge CPU. Which quantization tier should I ship?"

## Pipeline

```
torchvision → ONNX (FP32) → onnxconverter-common  → FP16
                         → quantize_dynamic       → INT8 (weights only)
                         → quantize_static + QDQ  → INT8 (weights + activations)
                                                     │
              ┌──────────────────────────────────────┘
              ▼
    benchmark.py  →  latency · size · top-5 overlap vs FP32
```

## Run

```bash
pip install -r requirements.txt

# 1. Export the model (defaults to MobileNetV3-Small; pass --model resnet18 to switch)
python export_onnx.py

# 2. Quantize the FP32 ONNX
python quantize.py models/mobilenet_v3_small_fp32.onnx

# 3. Benchmark everything in models/
python benchmark.py
```

## Sample output

```
Reference: mobilenet_v3_small_fp32.onnx

Model                                     Size MB  Median ms   P95 ms  Speedup  Top-5
--------------------------------------------------------------------------------------
mobilenet_v3_small_fp32.onnx                 9.83      14.21    16.04    1.00x    5/5
mobilenet_v3_small_fp16.onnx                 4.93       9.84    11.32    1.44x    5/5
mobilenet_v3_small_int8_dynamic.onnx         3.21       7.62     8.91    1.86x    5/5
mobilenet_v3_small_int8_static.onnx          2.78       5.39     6.40    2.64x    4/5
```

Plus a side-by-side bar chart in `results.png`.

## Why this matters for edge deployment

| Quantization | Size | Latency | Accuracy | When to use |
|---|---|---|---|---|
| **FP32** | 1.00× | 1.00× | reference | baseline |
| **FP16** | 0.50× | ~1.5× faster | usually lossless | GPUs / ANE / NPUs with FP16 ALUs |
| **INT8 dynamic** | ~0.33× | ~2× faster | tiny drop | CPU-only edge, fast to deploy |
| **INT8 static** | ~0.28× | ~2.5–4× faster | small drop | needs calibration; best for mobile / MCU |

## Files

```
├── export_onnx.py     # torchvision → ONNX (FP32 + FP16)
├── quantize.py        # Dynamic + static INT8 with onnxruntime.quantization
├── benchmark.py       # Latency stats, top-5 overlap, JSON + plot output
├── requirements.txt
└── .gitignore
```

## Notes on calibration

`quantize.py` uses random tensors for static-INT8 calibration as a placeholder. **For real deployment, swap `RandomCalibrationReader` with a reader that iterates over 100–500 samples of your actual validation set** — calibration data quality has a bigger impact on final accuracy than the quantization algorithm itself.

## License

MIT
