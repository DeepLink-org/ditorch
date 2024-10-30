#!/bin/bash
DEVICE=$1

date

pip install -r ditorch/test/test_mock_npu/requirements.txt
case "$DEVICE" in
    "npu")
        pytest ditorch/test/test_mock_npu
        ;;
    "camb")
        echo "Option camb selected to run"
        ;;
    "dipu")
        echo "Option dipu selected to run"
        ;;
    *)
        echo "unexpected option value: ${DEVICE}"
        exit 1
        ;;
esac
