find . -name test*.py | xargs -t -I {} python {} 2>&1 | tee test.log

# Check if any tests failed
if grep -Eq "FAILED|AssertionError" test.log; then
    echo "Tests failed"
    exit 1
else
    echo "Tests passed"
    exit 0
fi