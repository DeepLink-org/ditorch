import psutil
import time
import argparse
import subprocess


is_ascend_npu_env = subprocess.run("npu-smi info", shell=True, capture_output=True, text=True).returncode == 0


def parse_args():
    parser = argparse.ArgumentParser(description="Monitor process")
    parser.add_argument(
        "--pid",
        type=int,
        help="id of the process to monitor",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=1,
        help="interval of monitoring in seconds",
    )
    return parser.parse_args()


def get_host_mem_usage(pid):
    process = psutil.Process(pid)
    memory_info = process.memory_info()
    info = {"rss": f"{memory_info.rss >> 20} MB", "cpu_percent": f"{process.cpu_percent()}%"}
    return info


def get_ascend_device_mem_usage(pid):
    command = R"npu-smi info | awk -v pid=" + str(pid) + R" 'pid==$5  {print $9}'"
    device_memusage = subprocess.run(command, shell=True, capture_output=True, text=True).stdout.replace("\n", " ") + " MB"
    info = {"device_memusage": device_memusage}
    return info


if __name__ == "__main__":
    args = parse_args()
    pid = args.pid
    interval = args.interval
    while True:
        process = psutil.Process(pid)
        memory_info = process.memory_info()
        info = {"time": "%.7f" % time.time(), "time_str": time.strftime("%Y-%m-%d %H:%M:%S")}
        info.update(get_host_mem_usage(pid))
        if is_ascend_npu_env:
            info.update(get_ascend_device_mem_usage(pid))

        info_str = ""
        for key, value in info.items():
            info_str += f"{key}: {value} \t"
        print(info_str)
        time.sleep(interval)
