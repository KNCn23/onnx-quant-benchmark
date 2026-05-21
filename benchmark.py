"""
Benchmark ONNX models for inference latency and output similarity.
Compares every model in a directory against the FP32 reference.
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort


def load_session(path: Path) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def benchmark(session, x, warmup=10, iters=100):
    name = session.get_inputs()[0].name
    for _ in range(warmup): session.run(None, {name: x})
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        out = session.run(None, {name: x})
        samples.append(time.perf_counter() - t0)
    return np.array(samples), out[0]


def top5_overlap(ref_logits, q_logits):
    ref_top = np.argsort(-ref_logits.flatten())[:5]
    q_top   = np.argsort(-q_logits.flatten())[:5]
    return len(set(ref_top) & set(q_top))


def run(model_dir: Path, json_out: Path | None):
    fp32_path = next(model_dir.glob("*_fp32.onnx"))
    print(f"Reference: {fp32_path.name}\n")

    np.random.seed(42)
    x = np.random.randn(1, 3, 224, 224).astype(np.float32)

    ref_session = load_session(fp32_path)
    ref_samples, ref_logits = benchmark(ref_session, x)

    results = []
    for model in sorted(model_dir.glob("*.onnx")):
        size_mb = os.path.getsize(model) / 1e6
        try:
            x_in = x.astype(np.float16) if "fp16" in model.name else x
            sess = load_session(model)
            samples, logits = benchmark(sess, x_in)
        except Exception as e:
            print(f"  {model.name}: SKIP ({type(e).__name__}: {e})")
            continue
        med_ms  = float(np.median(samples) * 1000)
        p95_ms  = float(np.percentile(samples, 95) * 1000)
        speedup = float(np.median(ref_samples) / np.median(samples))
        if logits.dtype != np.float32:
            logits = logits.astype(np.float32)
        overlap = top5_overlap(ref_logits, logits)
        results.append({
            "model": model.name, "size_mb": round(size_mb, 2),
            "median_ms": round(med_ms, 2), "p95_ms": round(p95_ms, 2),
            "speedup_x": round(speedup, 2), "top5_overlap": overlap,
        })

    print(f"{'Model':<40} {'Size MB':>8} {'Median ms':>10} "
          f"{'P95 ms':>8} {'Speedup':>8} {'Top-5':>6}")
    print("-" * 86)
    for r in results:
        print(f"{r['model']:<40} {r['size_mb']:>8.2f} "
              f"{r['median_ms']:>10.2f} {r['p95_ms']:>8.2f} "
              f"{r['speedup_x']:>7.2f}x {r['top5_overlap']:>5}/5")

    if json_out:
        json_out.write_text(json.dumps(results, indent=2))
        print(f"\nWrote {json_out}")

    return results


def plot(results, out_path: Path):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib missing — skip plot")
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    names = [r["model"].replace(".onnx", "") for r in results]
    ax1.barh(names, [r["median_ms"] for r in results], color="#4C72B0")
    ax1.set_xlabel("Median latency (ms)")
    ax1.invert_yaxis()
    ax1.set_title("Inference latency")
    ax2.barh(names, [r["size_mb"] for r in results], color="#55A868")
    ax2.set_xlabel("Model size (MB)")
    ax2.invert_yaxis()
    ax2.set_title("On-disk size")
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models",   default="models")
    ap.add_argument("--json-out", default="results.json")
    ap.add_argument("--plot-out", default="results.png")
    args = ap.parse_args()
    results = run(Path(args.models), Path(args.json_out))
    plot(results, Path(args.plot_out))
