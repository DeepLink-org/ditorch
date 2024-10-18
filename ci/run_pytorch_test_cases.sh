export DITORCH_ROOT=$(pwd)
export PYTHONPATH=$PYTHONPATH:$DITORCH_ROOT
export EXTRA_ARGS=" --verbose --save-xml=$DITORCH_ROOT/pytorch_test_result/xml  --failfast "
python ditorch/test/copy_and_process_pytorch_test_scripts.py  # copy and add "import ditorch" to test script file
python ditorch/test/discover_pytorch_test_case.py # discover test cases and write test case ids to json file

python ditorch/test/generate_test_shell_script_from_testcase_json.py  --test_case_num_per_process 50
python ditorch/test/generate_test_shell_script_from_testcase_json.py --test_case_id_json_path pytorch_test_result/test_case_ids/all_cpu_test_cases.json  --test_case_num_per_process 100


cd pytorch_test_temp/test/

# run device test cases
bash $DITORCH_ROOT/pytorch_test_result/test_case_ids/ditorch_run_all_device_test_cases.json.sh

# run cpu test cases
# bash $DITORCH_ROOT/pytorch_test_result/test_case_ids/ditorch_run_all_cpu_test_cases.json.sh
ls test_*.py | xargs -I {} --verbose  python {} -k CPU $EXTRA_ARGS

# python ditorch/test/generate_test_shell_script_from_testcase_json.py --test_case_id_json_path pytorch_test_result/never_tested_device_test_case.json  --test_case_num_per_process 1


cd $DITORCH_ROOT
python ditorch/test/summary_test_results.py # generate test summary report
python ditorch/test/generate_test_shell_script_from_testcase_json.py  --test_case_num_per_process 1
python ditorch/test/generate_test_shell_script_from_testcase_json.py --test_case_id_json_path pytorch_test_result/never_tested_cpu_test_case.json --test_case_num_per_process 1
python ditorch/test/generate_test_shell_script_from_testcase_json.py --test_case_id_json_path pytorch_test_result/never_tested_device_test_case.json --test_case_num_per_process 1


export EXTRA_ARGS=" --verbose --save-xml=$DITORCH_ROOT/pytorch_test_result/xml "
cd $DITORCH_ROOT/pytorch_test_temp/test/
bash $DITORCH_ROOT/pytorch_test_result/ditorch_run_never_tested_cpu_test_case.json.sh
bash $DITORCH_ROOT/pytorch_test_result/ditorch_run_never_tested_device_test_case.json.sh