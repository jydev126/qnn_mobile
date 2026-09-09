"""在临时副本里破坏结果，确认验收程序不会把错误结果报告为通过。

依赖已完成的本机生命周期实验；不连接手机、不修改原始证据。
"""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class ComparisonRejectionTests(unittest.TestCase):
    def setUp(self):
        source = ROOT / "output/lifecycle"
        if not (source / "comparison.json").exists():
            self.skipTest("先完成 make lifecycle-compare，当前没有真实实验 fixture")
        self.temp = tempfile.TemporaryDirectory(prefix="qnn-comparison-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.names = {}
        for mode in ("dlc", "context", "cpp", "build-context"):
            pointer = source / f"latest-{mode}.txt"
            name = pointer.read_text().strip()
            self.names[mode] = name
            shutil.copy2(pointer, self.root / pointer.name)
            shutil.copytree(source / name, self.root / name,
                            ignore=shutil.ignore_patterns("*.bin", "qnn-context-runner"))
        binary = source / self.names["build-context"] / "rf_detr.bin"
        # 只读取大 binary，不为小型验收测试复制 61 MB。
        (self.root / self.names["build-context"] / "rf_detr.bin").symlink_to(binary)

    def compare(self):
        return subprocess.run([sys.executable, str(ROOT / "scripts/compare-lifecycle.py"),
                               "--root", str(self.root)], capture_output=True, text=True)

    def test_original_results_pass(self):
        result = self.compare()
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_changed_class_is_rejected(self):
        path = self.root / self.names["cpp"] / "Result_0/classes.raw"
        values = np.fromfile(path, dtype="<i4")
        values[0] += 1
        values.tofile(path)
        result = self.compare()
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertFalse(json.loads((self.root / "comparison.json").read_text())["pass"])

    def test_different_input_identity_is_rejected(self):
        path = self.root / self.names["cpp"] / "sha256.txt"
        lines = path.read_text().splitlines()
        lines = ["0" * 64 + "  " + line.split(maxsplit=1)[1]
                 if line.endswith("/image.raw") else line for line in lines]
        path.write_text("\n".join(lines) + "\n")
        result = self.compare()
        self.assertEqual(result.returncode, 1)
        self.assertIn("Different assets between paths: image.raw", result.stderr)


if __name__ == "__main__":
    unittest.main()
