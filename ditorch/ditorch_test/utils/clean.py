import shutil
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
dirtorch_test_dir = os.path.dirname(current_dir)

# shutil.rmtree(os.path.join(dirtorch_test_dir, "origin_torch"), ignore_errors=True)
# shutil.rmtree(os.path.join(dirtorch_test_dir, "processed_tests"), ignore_errors=True)

for filename in os.listdir(os.path.join(dirtorch_test_dir, "failed_tests_record")):
    if filename.endswith("json"):
        file_path = os.path.join(dirtorch_test_dir, "failed_tests_record", filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
