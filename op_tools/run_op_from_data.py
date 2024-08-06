import torch
import ditorch
import argparse
from op_tools.op_runner import OpRunner, SyncExecuteTimer


def main():
    parser = argparse.ArgumentParser(
        description="Run the operator from the data captured by the OpCapture tool"
    )

    parser.add_argument("dir", type=str, help="data dir")
    args = parser.parse_args()
    timer = SyncExecuteTimer()
    runner = OpRunner(args.dir, timer)
    for i in range(100):
        runner.run_forward()
        # runner.run_backward()


if __name__ == "__main__":
    main()
