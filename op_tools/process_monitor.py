import psutil
import time
import argparse
import subprocess
from prettytable import PrettyTable

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


def get_ascend_device_utilization(pid):
    command = R"npu-smi info | awk -v pid=" + str(pid) + R" 'pid==$5  {print $2}'"
    device_cards = subprocess.run(command, shell=True, capture_output=True, text=True).stdout.replace("\n", " ").strip().split(" ")
    info = {"device_cards": device_cards if len(device_cards) > 1 else int(device_cards[0].strip())}
    for device_card in device_cards:
        device_card = int(device_card.strip())
        command = f"npu-smi info  -t usages -i {device_card}"
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        contents = result.stdout.strip()
        for content in contents.split("\n"):
            if ":" in content:
                key, value = content.split(":")
                if len(device_cards) > 1:
                    key = f"{device_card}_{key.strip()}"
                else:
                    key = key.strip()
                info[f"{key}"] = value.strip()
    return info


def print_dict_info(info):
    table = PrettyTable()
    table.field_names = ["key", "value"]
    for key, value in info.items():
        table.add_row([key, value])
    print(table)
    print("\n" * 2)


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
            info.update(get_ascend_device_utilization(pid))

        print_dict_info(info)

        time.sleep(interval)
