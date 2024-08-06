import torch
import ditorch
import argparse
from op_tools.op_runner import OpRunner


def main():
    parser = argparse.ArgumentParser(
        description="Run the operator from the data captured by the OpCapture tool"
    )

    parser.add_argument("dir", type=str, help="data dir")
    args = parser.parse_args()
    runner = OpRunner(args.dir)
    runner.run_forward()


if __name__ == "__main__":
    main()
