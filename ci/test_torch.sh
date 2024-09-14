set -ex
export DISABLED_TESTS_FILE=ditorch/ditorch_test/unsupported_test_cases/torch_npu_disabled_tests.json

python ditorch/ditorch_test/utils/clean.py
# python ditorch/ditorch_test/main.py
python ditorch/ditorch_test/processed_tests/test_nn.py

if [ "$(ls -A ditorch/ditorch_test/failed_tests_record)" ]; then
    echo "Test torch failed! You can check failed_tests_record for more info!"
    exit 1
else
    echo "Test torch passed!"
    exit 0
fi