date
find op_tools/ -name test*.py | xargs -I {} bash -c ' echo "start run {}";date;time python {} && echo "Test {} PASSED\n\n\n" || echo "Test {} FAILED\n\n\n"' 2>&1 | tee test_op_tools_cases.log

# Check if any tests failed
if grep -Eq "FAILED" test_op_tools_cases.log; then
    echo "tests failed"
    exit 1
else
    echo "all tests passed"
    exit 0
fi