import shutil
import os
import torch

from utils.process_test import process_src_code
from utils.utils import sparse_checkout
from utils.unnecessary_tests import unnecessary_tests

MAIN_DIR = os.path.dirname(os.path.abspath(__file__))
ORIGIN_TORCH_PATH = MAIN_DIR + "/origin_torch/"
TORCH_TEST_PATH = ORIGIN_TORCH_PATH + "test"
TORCH_URL = "https://github.com/pytorch/pytorch.git"


torch_tag = "v" + torch.__version__.split("+")[0]

if not os.path.exists(ORIGIN_TORCH_PATH):
    sparse_checkout(TORCH_URL, ORIGIN_TORCH_PATH, ["test"], torch_tag)


# 从torch_test_path拷贝测试脚本，如果对应的脚本在unnecessary_tests列表中，则跳过
shutil.copytree(
    TORCH_TEST_PATH,
    MAIN_DIR + "/processed_tests/",
    ignore=lambda dir, contents: [
        name for name in contents if name in unnecessary_tests
    ],
    dirs_exist_ok=True,
)


os.chdir(MAIN_DIR + "/processed_tests")
for item in os.listdir(os.getcwd()):
    if item.startswith("test_") and item.endswith(".py"):
        process_src_code(item)
