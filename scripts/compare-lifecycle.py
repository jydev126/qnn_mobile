"""核对三条 RF-DETR 路径的每次输出，汇总 API wall time；不是 mAP 验证。"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
SPECS = {
    "boxes": ("<f4", [1, 300, 4], "QNN_DATATYPE_FLOAT_32"),
    "logits": ("<f4", [1, 300], "QNN_DATATYPE_FLOAT_32"),
    "classes": ("<i4", [1, 300], "QNN_DATATYPE_INT_32"),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def latest(root, mode):
    name = (root / f"latest-{mode}.txt").read_text().strip()
    require(re.fullmatch(re.escape(mode) + r"\.[A-Za-z0-9]+", name), "Invalid latest pointer")
    result = root / name
    require((result / "exit-code.txt").read_text().strip() == "0", f"Failed run: {result}")
    return result


def hashes(result):
    rows = {}
    for line in (result / "sha256.txt").read_text().splitlines():
        digest, path = line.split(maxsplit=1)
        rows[Path(path).name] = digest
    return rows


def output_path(result, mode, index, name):
    suffix = "_native" if name == "classes" and mode != "cpp" else ""
    return result / f"Result_{index}" / f"{name}{suffix}.raw"


def load_run(result, mode):
    env = dict(line.split("=", 1) for line in (result / "environment.txt").read_text().splitlines() if "=" in line)
    count = int(env["num_inferences"])
    require(env["output_dtype"] == "native", "Only native outputs are comparable here")
    if mode == "cpp":
        with (result / "tensors.tsv").open() as f:
            specs = {r["name"]: r for r in csv.DictReader(f, delimiter="\t") if r["direction"] == "output"}
        require(set(specs) == set(SPECS), "Unexpected C++ output names")
        for name, (dtype, shape, _) in SPECS.items():
            r = specs[name]
            require(r["dtype"] == np.dtype(dtype).name and r["shape"] == "x".join(map(str, shape)),
                    f"C++ metadata mismatch: {name}")
    else:
        metadata = yaml.safe_load((result / "execution_metadata.yaml").read_text())
        require(metadata["inferences_completed"] == count, "Incomplete inference count")
        require("--use_native_output_files" in metadata["command"], "net-run did not use native output")
        require(len(metadata["graphs"]) == 1, "Expected one graph")
        specs = {r["tensor_name"]: r for r in metadata["graphs"][0]["output_tensors"]}
        require(set(specs) == set(SPECS), "Unexpected net-run output names")
        for name, (_, shape, qnn_type) in SPECS.items():
            require(specs[name]["dimensions"] == shape and specs[name]["datatype"] == qnn_type,
                    f"net-run metadata mismatch: {name}")

    outputs = []
    for i in range(count):
        arrays = {}
        for name, (dtype, shape, _) in SPECS.items():
            path = output_path(result, mode, i, name)
            require(path.stat().st_size == int(np.prod(shape)) * 4, f"Wrong output byte count: {path}")
            array = np.fromfile(path, dtype=dtype)
            require(np.isfinite(array).all(), f"Nonfinite output: {path}")
            arrays[name] = array
        outputs.append(arrays)
    return outputs


def read_timing(result, mode):
    if mode == "cpp":
        with (result / "timings.csv").open() as file:
            phases = {r["phase"]: float(r["milliseconds"]) for r in csv.DictReader(file)}
        executes = [v for k, v in phases.items() if k.startswith("execute_")]
        return phases, executes
    # viewer 的 CSV 有版本信息前言。保留原始文件，仅提取 NETRUN ROOT / US。
    lines = (result / "profile.csv").read_text().splitlines()
    header = next(i for i, line in enumerate(lines) if line.startswith("Msg Timestamp,"))
    phases, executes = {}, []
    for row in csv.DictReader(lines[header:], skipinitialspace=True):
        if row["Timing Source"] != "NETRUN" or row["Event Level"] != "ROOT" or row["Unit of Measurement"] != "US":
            continue
        ms = float(row["Time"]) / 1000
        if row["Message"] == "EXECUTE":
            executes.append(ms)
        else:
            phases[row["Message"]] = ms
    return phases, executes


def timing_summary(result, mode, count):
    phases, execute = read_timing(result, mode)
    require(len(execute) == count, f"Profiling count mismatch: {result}")
    subsequent = execute[1:]
    return {
        "phases_ms": phases,
        "execute_ms": execute,
        "first_execute_ms": execute[0],
        "subsequent_mean_ms": float(np.mean(subsequent)) if subsequent else None,
        "subsequent_min_ms": min(subsequent) if subsequent else None,
        "subsequent_max_ms": max(subsequent) if subsequent else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "output/lifecycle")
    args = parser.parse_args()
    runs = {mode: latest(args.root, mode) for mode in ("dlc", "context", "cpp")}
    build = latest(args.root, "build-context")
    identities = {mode: hashes(path) for mode, path in runs.items()}
    for name in ("model.dlc", "image.raw", "input_list.txt", "libQnnHtp.so", "libQnnSystem.so", "libQnnHtpV79Stub.so", "libQnnHtpV79Skel.so"):
        require(len({h[name] for h in identities.values()}) == 1, f"Different assets between paths: {name}")
    for mode in ("context", "cpp"):
        require((runs[mode] / "context-source.txt").read_text().strip() == build.name,
                "Latest context build does not match run; rerun context-run/cpp-run")
        require(identities[mode]["rf_detr.bin"] == hashes(build)["rf_detr.bin"], "Context hashes differ")
    # 对本地已拉取 binary 再核对一次，防止它后来被替换。
    with (build / "rf_detr.bin").open("rb") as file:
        require(hashlib.file_digest(file, "sha256").hexdigest() == hashes(build)["rf_detr.bin"],
                "Local context binary changed")

    outputs = {mode: load_run(path, mode) for mode, path in runs.items()}
    baseline = outputs["dlc"][0]
    comparison = []
    passed = True
    # 每条路径每次执行均与 DLC 第 0 次比较；同时检查同路径重复执行的一致性。
    for mode, items in outputs.items():
        for i, arrays in enumerate(items):
            for name, values in arrays.items():
                ref = baseline[name]
                difference = values.astype(np.float64) - ref.astype(np.float64)
                exact = bool(np.array_equal(ref, values))
                ok = exact if name == "classes" else bool(np.allclose(values, ref, rtol=1e-4, atol=1e-4))
                passed &= ok
                comparison.append({"mode": mode, "iteration": i, "tensor": name,
                                   "exact": exact, "pass": ok,
                                   "max_abs": float(np.abs(difference).max()),
                                   "rmse": float(np.sqrt(np.mean(difference ** 2)))})
    timing = {mode: timing_summary(path, mode, len(outputs[mode])) for mode, path in runs.items()}
    build_phases, _ = read_timing(build, "build-context")
    summary = {
        "pass": passed, "reference": f"{runs['dlc'].name}/Result_0",
        "tolerance": {"float_rtol": 1e-4, "float_atol": 1e-4, "classes": "exact"},
        "runs": {mode: path.name for mode, path in runs.items()}, "context_build": build.name,
        "context_bytes": (build / "rf_detr.bin").stat().st_size,
        "comparisons": comparison, "timing": timing, "build_phases_ms": build_phases,
        "limitations": "Same-image runtime consistency only; no PyTorch/mAP validation. Timings have different logging/profiling overhead and uncontrolled device clocks; subsequent runs are not proven steady state. INIT may contain compose/finalize; do not sum nested phases.",
    }
    (args.root / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
    for mode in runs:
        rows = [r for r in comparison if r["mode"] == mode]
        print(f"{mode} ({runs[mode].name}): {len(outputs[mode])} executions, exact={all(r['exact'] for r in rows)}, "
              f"max_abs={max(r['max_abs'] for r in rows):.8g}")
        t = timing[mode]
        print(f"  first={t['first_execute_ms']:.3f} ms; subsequent mean={t['subsequent_mean_ms']} ms")
    print(f"{'[OK]' if passed else '[FAIL]'} {args.root / 'comparison.json'}")
    return 0 if passed else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError, StopIteration) as error:
        print(f"[FAIL] {error}", file=sys.stderr)
        sys.exit(1)
