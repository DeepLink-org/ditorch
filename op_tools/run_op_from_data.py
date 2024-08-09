import torch
import ditorch
import argparse
from op_tools.op_runner import OpRunner, SyncExecuteTimer, OpAccyChecker


def parse_args():
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
    parser.add_argument(
        "--sync_time_measure",
        type=bool,
        default=True,
        help="Run the operator synchronously and test the operator running time",
    )

    parser.add_argument(
        "--acc_check",
        type=bool,
        default=True,
        help="Run the operator and test for accuracy",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    runner = OpRunner(args.dir)
    if args.sync_time_measure:
        timer = SyncExecuteTimer()
        runner.add_hook(timer)

    if args.acc_check:
        acc_checker = OpAccyChecker()
        runner.add_hook(acc_checker)

    for i in range(args.run_times):
        runner.run_forward()
        if not args.only_run_forward:
            runner.run_backward()


if __name__ == "__main__":
    main()
