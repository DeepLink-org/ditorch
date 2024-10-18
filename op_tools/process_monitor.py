import psutil
import time
import argparse
import subprocess
import os
from prettytable import PrettyTable
from op_tools.pretty_print import dict_data_list_to_table

is_ascend_npu_env = subprocess.run("npu-smi info", shell=True, capture_output=True, text=True).returncode == 0
is_camb_mlu_env = subprocess.run("cnmon", shell=True, capture_output=True, text=True).returncode == 0


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
    info = {"rss": f"{memory_info.rss >> 20} MB", "cpu_percent": f"{process.cpu_percent(interval=1)}%"}
    return info


def get_camb_device_mem_usage(pid):
    command = R"cnmon | awk -v pid=" + str(pid) + R" 'pid==$4  {print $6}'"
    device_memusage = subprocess.run(command, shell=True, capture_output=True, text=True).stdout.replace("\n", " ") + " MB"
    info = {"device_memusage": device_memusage}
    return info


def get_camb_device_utilization(pid):
    command = R"cnmon | awk -v pid=" + str(pid) + R" 'pid==$4  {print $2}'"
    device_cards = subprocess.run(command, shell=True, capture_output=True, text=True).stdout.replace("\n", " ").strip().split(" ")
    device_cards = device_cards if len(device_cards) > 1 else int(device_cards[0].strip())
    info = {"device_cards": device_cards}

    command = f"cnmon info --card {device_cards} --bandwidth --util --memory | grep MLU"
    raw_output = subprocess.run(command, shell=True, capture_output=True, text=True).stdout
    items = raw_output.strip().split("\n")
    for item in items:
        item_list = item.split(":")
        if len(item_list) != 2 :
            continue
        key, value = item_list
        info.update({key.strip() : value.strip()})

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


class ResultCache:
    global_result = []

    def __init__(self, pid) -> None:
        device_name = ""
        if is_ascend_npu_env:
            device_name = "ascend"
        if is_camb_mlu_env:
            device_name = "camb"
        self.file_name = f"op_tools_results/process_monitor_result_{device_name}_pid{pid}_{os.getenv('label', '')}_{time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())}.csv"  # noqa: E501
        self.dir = self.file_name[0 : self.file_name.rfind("/")]

    def append(self, info):
        self.global_result.append(info)

        if len(self.global_result) > int(os.getenv("OP_TOOLS_MAX_CACHE_SIZE", "1")):
            self.write_to_file()

    def write_to_file(self):
        if len(self.global_result) == 0:
            return
        table = dict_data_list_to_table(self.global_result)
        self.global_result.clear()
        data_string = table.get_csv_string()

        if os.path.exists(self.file_name):
            data_string = data_string[data_string.find("\n") + 1:]

        os.makedirs(self.dir, exist_ok=True)
        with open(self.file_name, "a+") as f:
            f.write(data_string)
            f.close
        print(f"op process monitor result saved to {self.file_name}")


if __name__ == "__main__":
    args = parse_args()
    pid = args.pid
    interval = args.interval
    result_cache = ResultCache(pid=pid)
    while True:
        process = psutil.Process(pid)
        memory_info = process.memory_info()
        info = {"time": "%.7f" % time.time(), "time_str": time.strftime("%Y-%m-%d %H:%M:%S")}
        info.update(get_host_mem_usage(pid))
        if is_ascend_npu_env:
            info.update(get_ascend_device_mem_usage(pid))
            info.update(get_ascend_device_utilization(pid))

        if is_camb_mlu_env:
            info.update(get_camb_device_mem_usage(pid))
            info.update(get_camb_device_utilization(pid))

        print_dict_info(info)
        result_cache.append(info)
        time.sleep(interval)
