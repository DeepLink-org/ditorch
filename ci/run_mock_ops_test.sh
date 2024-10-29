#!/bin/bash
DEVICE=$1

date

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
        echo "Unknown option selected"
        ;;
esac
