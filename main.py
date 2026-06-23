#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
统一命令行入口，无算法运行 UI。

示例：
python main.py preprocess --input_dir data/raw --output_dir data/processed
python main.py experiment --data_dir data/processed --output_dir results
python main.py all --input_dir data/raw --processed_dir data/processed --output_dir results
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def run(cmd):
    print("[RUN]", " ".join(map(str, cmd)))
    subprocess.run(cmd, check=True)

def main():
    parser = argparse.ArgumentParser(description="Dry Bean 多分类实验系统")
    sub = parser.add_subparsers(dest="command", required=True)

    p1 = sub.add_parser("preprocess", help="数据加载、数据清洗与特征工程")
    p1.add_argument("--input_dir", default="data/raw")
    p1.add_argument("--output_dir", default="data/processed")

    p2 = sub.add_parser("experiment", help="训练模型、测试、绘图与鲁棒性分析")
    p2.add_argument("--data_dir", default="data/processed")
    p2.add_argument("--output_dir", default="results")
    p2.add_argument("--skip_robustness", action="store_true")

    p3 = sub.add_parser("all", help="完整运行预处理和实验")
    p3.add_argument("--input_dir", default="data/raw")
    p3.add_argument("--processed_dir", default="data/processed")
    p3.add_argument("--output_dir", default="results")
    p3.add_argument("--skip_robustness", action="store_true")

    args = parser.parse_args()

    if args.command == "preprocess":
        run([sys.executable, ROOT / "scripts" / "preprocess_drybean.py",
             "--input_dir", args.input_dir, "--output_dir", args.output_dir])

    elif args.command == "experiment":
        cmd = [sys.executable, ROOT / "scripts" / "drybean_main_experiment.py",
               "--data_dir", args.data_dir, "--output_dir", args.output_dir]
        if args.skip_robustness:
            cmd.append("--skip_robustness")
        run(cmd)

    elif args.command == "all":
        run([sys.executable, ROOT / "scripts" / "preprocess_drybean.py",
             "--input_dir", args.input_dir, "--output_dir", args.processed_dir])
        cmd = [sys.executable, ROOT / "scripts" / "drybean_main_experiment.py",
               "--data_dir", args.processed_dir, "--output_dir", args.output_dir]
        if args.skip_robustness:
            cmd.append("--skip_robustness")
        run(cmd)

if __name__ == "__main__":
    main()
