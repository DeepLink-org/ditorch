import subprocess
import json
import os
from multiprocessing import Process


class CommandRunner:
    def __init__(self, commands, max_workers=4, output_dir="pytorch_test_result", cwd=None):
        """
        初始化 CommandRunner 类

        :param commands: 包含命令的列表，每个命令应为 (id, command) 的元组
        :param max_workers: 最大并行进程数
        :param cwd: 子进程当前工作路径
        """
        self.commands = commands
        self.max_workers = max_workers
        self.output_dir = output_dir
        self.cwd = cwd
        os.makedirs(f"{self.output_dir}", exist_ok=True)
        os.makedirs(f"{self.output_dir}/passed", exist_ok=True)
        os.makedirs(f"{self.output_dir}/failed", exist_ok=True)

    def _run_command(self, command_id, command):
        """运行命令并将结果写入独立的 JSON 文件中"""
        result_data = {"command_id": command_id, "returncode": 0, "command": command, "stdout": "", "stderr": "", "exception": ""}
        try:
            result = subprocess.run(
                command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False, shell=True, cwd=self.cwd
            )
        except Exception as e:
            result_data["exception"] = str(e)
        result_data["stdout"] = result.stdout.strip()
        result_data["stderr"] = result.stderr.strip()
        result_data["returncode"] = result.returncode

        # 将结果写入 JSON 文件
        output_file = f'{self.output_dir}/{"passed" if result.returncode == 0 else "failed"}/result_{command_id}.json'
        with open(output_file, "w") as f:
            json.dump(result_data, f, indent=4)

        print(f'\"{command}\" exit {result.returncode} {output_file}')

    def run(self):
        processes = []

        for command_id, command in self.commands:
            while len(processes) >= self.max_workers:
                # 等待某个进程完成
                for p in processes:
                    if not p.is_alive():
                        processes.remove(p)
                        break

            process = Process(target=self._run_command, args=(command_id, command))
            processes.append(process)
            process.start()

        # 等待所有进程完成
        for p in processes:
            p.join()


# 使用示例
if __name__ == "__main__":
    commands = [
        ("echo_command", ["echo", "Hello from process 1"]),
        ("ls_command", ["ls", "-l"]),
        (3, ["sleep", "2"]),
        (4, ["echo", "Hello from process 2"]),
    ]

    runner = CommandRunner(commands, max_workers=2)
    runner.run()

    print("所有命令已完成，结果已保存为 JSON 格式到独立文件中")
