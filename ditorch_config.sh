# export env_name=${env_name:-env_default_value}

export DITORCH_SHOW_DEVICE_AS_CUDA=${DITORCH_SHOW_DEVICE_AS_CUDA:-1}


export OP_TOOLS_PRINT_STACK=${OP_TOOLS_PRINT_STACK:-0}

export OP_TOOLS_MAX_CACHE_SIZE=${OP_TOOLS_MAX_CACHE_SIZE:-1000}


export OP_AUTOCOMPARE_DISABLE_LIST=${OP_AUTOCOMPARE_DISABLE_LIST:-"torch.rand,torch.randn,torch_mlu.*,torch_npu.*"}
export OP_AUTOCOMPARE_LIST=${OP_AUTOCOMPARE_LIST:-".*"}


export OP_CAPTURE_DISABLE_LIST=${OP_CAPTURE_DISABLE_LIST:-""}
export OP_CAPTURE_LIST=${OP_CAPTURE_LIST:-".*"}

export OP_DTYPE_CAST_DISABLE_LIST=${OP_DTYPE_CAST_DISABLE_LIST:-""}
export OP_DTYPE_CAST_LIST=${OP_DTYPE_CAST_LIST:-".*"}

export OP_FALLBACK_DISABLE_LIST=${OP_FALLBACK_DISABLE_LIST:-""}
export OP_FALLBACK_LIST=${OP_FALLBACK_LIST:-".*"}

export OP_OBSERVE_DISABLE_LIST=${OP_OBSERVE_DISABLE_LIST:-""}
export OP_OBSERVE_LIST=${OP_OBSERVE_LIST:-".*"}


export OP_OVERFLOW_CHECK_DISABLE_LIST=${OP_OVERFLOW_CHECK_DISABLE_LIST:-""}
export OP_OVERFLOW_CHECK_LIST=${OP_OVERFLOW_CHECK_LIST:-".*"}


export OP_TIME_MEASURE_DISABLE_LIST=${OP_TIME_MEASURE_DISABLE_LIST:-""}
export OP_TIME_MEASURE_LIST=${OP_TIME_MEASURE_LIST:-".*"}


# for autocompare and op_dtype_cast tools
# Set the dtype used by the CPU for autocompare
# Set the dtype used by the DEVICE for op_dtype_cast
# for special op
export LINEAR_OP_DTYPE_CAST_DICT=${LINEAR_OP_DTYPE_CAST_DICT:-"torch.float16->torch.float64,torch.bfloat16->torch.float64,torch.float32->torch.float64"}
export EMBEDDING_OP_DTYPE_CAST_DICT=${EMBEDDING_OP_DTYPE_CAST_DICT:-"torch.float16->torch.float64,torch.bfloat16->torch.float64,torch.float32->torch.float64"}
export NORMALIZE_OP_DTYPE_CAST_DICT=${NORMALIZE_OP_DTYPE_CAST_DICT:-"torch.float16->torch.float64,torch.bfloat16->torch.float64,torch.float32->torch.float64"}
export NORM_OP_DTYPE_CAST_DICT=${NORM_OP_DTYPE_CAST_DICT:-"torch.float16->torch.float64,torch.bfloat16->torch.float64,torch.float32->torch.float64"}
export CROSS_ENTROPY_OP_DTYPE_CAST_DICT=${CROSS_ENTROPY_OP_DTYPE_CAST_DICT:-"torch.float16->torch.float64,torch.bfloat16->torch.float64,torch.float32->torch.float64"}
export MUL_OP_DTYPE_CAST_DICT=${MUL_OP_DTYPE_CAST_DICT:-"torch.float16->torch.float64,torch.bfloat16->torch.float64,torch.float32->torch.float64"}
export MATMUL_OP_DTYPE_CAST_DICT=${MATMUL_OP_DTYPE_CAST_DICT:-"torch.float16->torch.float64,torch.bfloat16->torch.float64,torch.float32->torch.float64"}
export STD_OP_DTYPE_CAST_DICT=${STD_OP_DTYPE_CAST_DICT:-"torch.float16->torch.float64,torch.bfloat16->torch.float64,torch.float32->torch.float64"}
export EXP_OP_DTYPE_CAST_DICT=${EXP_OP_DTYPE_CAST_DICT:-"torch.float16->torch.float64,torch.bfloat16->torch.float64,torch.float32->torch.float64"}
# for generally op
export OP_DTYPE_CAST_DICT=${OP_DTYPE_CAST_DICT:-"torch.float16->torch.float32,torch.bfloat16->torch.float32"}


export AUTOCOMPARE_ERROR_TOLERANCE=${AUTOCOMPARE_ERROR_TOLERANCE:-"1e-3,1e-3"}
export AUTOCOMPARE_ERROR_TOLERANCE_FLOAT16=${AUTOCOMPARE_ERROR_TOLERANCE_FLOAT16:-"1e-4,1e-4"}
export AUTOCOMPARE_ERROR_TOLERANCE_BFLOAT16=${AUTOCOMPARE_ERROR_TOLERANCE_BFLOAT16:-"1e-3,1e-3"}
export AUTOCOMPARE_ERROR_TOLERANCE_FLOAT32=${AUTOCOMPARE_ERROR_TOLERANCE_FLOAT32:-"1e-5,1e-5"}
export AUTOCOMPARE_ERROR_TOLERANCE_FLOAT64=${AUTOCOMPARE_ERROR_TOLERANCE_FLOAT64:-"1e-8,1e-8"}

export LINEAR_AUTOCOMPARE_ERROR_TOLERANCE_BFLOAT16=${LINEAR_AUTOCOMPARE_ERROR_TOLERANCE_BFLOAT16:-"1e-2,1e-2"}
