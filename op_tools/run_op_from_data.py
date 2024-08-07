import torch
import ditorch
import argparse
from op_tools.op_runner import OpRunner, SyncExecuteTimer


def main():
    parser = argparse.ArgumentParser(
        description="Run the operator from the data captured by the OpCapture tool"
    )

    parser.add_argument("dir", type=str, help="data dir")
    parser.add_argument(
        "--run_times",
        type=int,
        default=100,
        help="The number of times the operator is repeated",
    )
    parser.add_argument(
        "--only_run_forward",
        type=bool,
        default=False,
        help="Only the forward calculation of the operator is run, not the backward calculation",
    )
    args = parser.parse_args()
    timer = SyncExecuteTimer()
    runner = OpRunner(args.dir, timer)
    for i in range(args.run_times):
        runner.run_forward()
        if not args.only_run_forward:
            runner.run_backward()


if __name__ == "__main__":
    main()
