from typing import Optional, Union
import torch
import numpy as np
import numpy.typing as npt
import os


def flush_print(*args, **kwargs):
    kwargs["flush"] = True
    print(*args, **kwargs)


def load_pt(
    path: str, pt_dict_key: str
) -> Union[torch.Tensor, tuple[Optional[torch.Tensor]]]:
    pt = torch.load(path, map_location="cpu")
    # flush_print(f"{pt.keys() = }")
    # flush_print(f"{pt = }")
    obj = pt[pt_dict_key]
    # flush_print(type(obj))
    # flush_print(f"{obj = }")
    return obj


def tensor_to_numpy_fp64(tensor: torch.Tensor) -> npt.NDArray[np.float64]:
    return tensor.detach().to(torch.float64).numpy()


def calc_cos_sim(expected: npt.NDArray, actual: npt.NDArray) -> npt.NDArray:
    # Ref: https://gitee.com/ascend/mstt/blob/master/debug/accuracy_tools/ptdbg_ascend/src/python/ptdbg_ascend/parse_tool/lib/compare.py#L132
    expected_flat = expected.reshape(-1)
    actual_flat = actual.reshape(-1)
    return np.dot(expected_flat, actual_flat) / (
        np.sqrt(np.dot(expected_flat, expected_flat))
        * np.sqrt(np.dot(actual_flat, actual_flat))
    )


def calc_diff_norm(expected: npt.NDArray, actual: npt.NDArray, ord: int) -> np.float64:
    expected_flat = expected.reshape(-1)
    actual_flat = actual.reshape(-1)
    return np.linalg.norm(np.abs(expected_flat - actual_flat), ord=ord)


def calc_abs_err(
    expected: npt.NDArray, actual: npt.NDArray, atol: float
) -> tuple[npt.NDArray, float]:
    abs_err = np.abs(expected - actual)
    abs_err_max_val = abs_err.max()
    abs_err_out_of_atol_pct = (abs_err > atol).sum() / abs_err.size
    return abs_err_max_val, abs_err_out_of_atol_pct


def calc_allclose(
    expected: npt.NDArray, actual: npt.NDArray, atol: float, rtol: float
) -> tuple[float, float]:
    # Ref: yangbo's view_no_fix_fwd.py
    abs_err = np.abs(expected - actual)
    out_of_tol_idx = np.abs(abs_err) > atol + rtol * np.abs(expected)
    out_of_tol_num = out_of_tol_idx.sum().item()
    out_of_tol_pct = out_of_tol_num / actual.size
    return out_of_tol_num, out_of_tol_pct


def calc_then_print_all_metrics(
    expected_pt_path: str,
    actual_pt_path: str,
    pt_dict_key: str,
    abs_err_atol: float,
    allclose_atol: float,
    allclose_rtol: float,
):
    expected_obj = load_pt(expected_pt_path, pt_dict_key)
    actual_obj = load_pt(actual_pt_path, pt_dict_key)
    expected_tensors: tuple[Optional[torch.Tensor]] = (
        (expected_obj,) if isinstance(expected_obj, torch.Tensor) else expected_obj
    )
    actual_tensors: tuple[Optional[torch.Tensor]] = (
        (actual_obj,) if isinstance(actual_obj, torch.Tensor) else actual_obj
    )
    assert len(expected_tensors) == len(actual_tensors)

    for i, (expected, actual) in enumerate(zip(expected_tensors, actual_tensors)):
        if len(expected_tensors) > 1:
            flush_print(f"---------- {i} ----------")

        if expected is None and actual is None:
            flush_print("None equals to None")
            continue
        assert isinstance(expected, torch.Tensor)
        assert isinstance(actual, torch.Tensor)

        if (0 == expected).all() and (0 == actual).all():
            flush_print(f"all zeros equals to all zeros")
            continue

        expected = tensor_to_numpy_fp64(expected)
        actual = tensor_to_numpy_fp64(actual)
        flush_print(f"{expected.shape = } | {actual.shape = }")

        cos_sim = calc_cos_sim(expected, actual)
        diff_norm_1 = calc_diff_norm(expected, actual, ord=1)
        diff_norm_2 = calc_diff_norm(expected, actual, ord=2)
        flush_print(
            f"similarity check: {cos_sim = } | {diff_norm_1 = } | {diff_norm_2 = }"
        )

        abs_err_max_val, abs_err_out_of_atol_pct = calc_abs_err(
            expected, actual, atol=abs_err_atol
        )
        flush_print(
            f"absolute error check (atol={abs_err_atol}): {abs_err_max_val = } | {abs_err_out_of_atol_pct = }"
        )

        out_of_tol_num, out_of_tol_pct = calc_allclose(
            expected, actual, atol=allclose_atol, rtol=allclose_rtol
        )
        flush_print(
            f"allclose check (atol={allclose_atol}, rtol={allclose_rtol}): {out_of_tol_num = } | {out_of_tol_pct = }"
        )

def fwd_layer_name(fwd_path: str):
    fwd_layer_name_path = os.path.join(fwd_path, "../modules_full_name_run.txt")
    assert os.path.isfile(fwd_layer_name_path), f"the modules_full_name_run.txt doesn't exist in {fwd_layer_name_path}."
    with open(fwd_layer_name_path, "r") as f:
        fwd_layer_names = [line.strip() for line in f]
    return fwd_layer_names


def compare_fwd_bwd(
    expected_fwd_dir_path: str,
    actual_fwd_dir_path: str,
    expected_bwd_dir_path: str,
    actual_bwd_dir_path: str,
    abs_err_atol: float,
    allclose_atol: float,
    allclose_rtol: float,
):
    # fwd_layer_file_names = os.listdir(expected_fwd_dir_path)
    fwd_layer_names = fwd_layer_name(expected_fwd_dir_path)
    fwd_layer_file_names = [layer_name + ".pt" for layer_name in fwd_layer_names]
    for fwd_layer_file_name in fwd_layer_file_names:
        pt_dict_key = "output"
        flush_print(
            f"==================== FWD ({pt_dict_key}) - {fwd_layer_file_name} ===================="
        )
        expected_pt_path = os.path.join(expected_fwd_dir_path, fwd_layer_file_name)
        real_no_fix_pt_path = os.path.join(actual_fwd_dir_path, fwd_layer_file_name)
        calc_then_print_all_metrics(
            expected_pt_path,
            real_no_fix_pt_path,
            pt_dict_key=pt_dict_key,
            abs_err_atol=abs_err_atol,
            allclose_atol=allclose_atol,
            allclose_rtol=allclose_rtol,
        )

    flush_print("\n")

    # bwd_layer_file_names = os.listdir(expected_bwd_dir_path)
    bwd_layer_file_names = fwd_layer_file_names
    for bwd_layer_file_name in bwd_layer_file_names[::-1]:
        pt_dict_key = "grad_input"
        flush_print(
            f"==================== BWD ({pt_dict_key}) - {bwd_layer_file_name} ====================",
        )
        expected_pt_path = os.path.join(expected_bwd_dir_path, bwd_layer_file_name)
        real_no_fix_pt_path = os.path.join(actual_bwd_dir_path, bwd_layer_file_name)
        if not os.path.isfile(expected_pt_path):
            print("backward of f{bwd_layer_file_name} not run and comparing is skipped")
            continue
        calc_then_print_all_metrics(
            expected_pt_path,
            real_no_fix_pt_path,
            pt_dict_key=pt_dict_key,
            abs_err_atol=abs_err_atol,
            allclose_atol=allclose_atol,
            allclose_rtol=allclose_rtol,
        )


abs_err_atol = 1e-3
allclose_atol = 1e-3
allclose_rtol = 1e-3

base_path = "/sensechat_c/yangbo1/910B_exps/easyllm_pack_liuwei10/module_dump/rank0"
expected_fwd_dir_path = os.path.join(base_path, "expected/forward")
expected_bwd_dir_path = os.path.join(base_path, "expected/backward")

# real_no_fix_fwd_dir_path = os.path.join(base_path, "real_no_fix/forward")
# real_no_fix_bwd_dir_path = os.path.join(base_path, "real_no_fix/backward")
# compare_fwd_bwd(
#     expected_fwd_dir_path,
#     real_no_fix_fwd_dir_path,
#     expected_bwd_dir_path,
#     real_no_fix_bwd_dir_path,
#     abs_err_atol,
#     allclose_atol,
#     allclose_rtol,
# )

# real_fix_fwd_dir_path = os.path.join(base_path, "real_fix_fwd/forward")
# real_fix_bwd_dir_path = os.path.join(base_path, "real_fix_bwd/backward")
# compare_fwd_bwd(
#     expected_fwd_dir_path,
#     real_fix_fwd_dir_path,
#     expected_bwd_dir_path,
#     real_fix_bwd_dir_path,
#     abs_err_atol,
#     allclose_atol,
#     allclose_rtol,
# )

# real_fix_rms_norm_no_fuse_fp32_fwd_dir_path = os.path.join(
#     base_path, "real-fix-rms_norm_no_fuse_fp32-fwd/forward"
# )
# real_fix_rms_norm_no_fuse_fp32_bwd_dir_path = os.path.join(
#     base_path, "real-fix-rms_norm_no_fuse_fp32-bwd/backward"
# )
# compare_fwd_bwd(
#     expected_fwd_dir_path,
#     real_fix_rms_norm_no_fuse_fp32_fwd_dir_path,
#     expected_bwd_dir_path,
#     real_fix_rms_norm_no_fuse_fp32_bwd_dir_path,
#     abs_err_atol,
#     allclose_atol,
#     allclose_rtol,
# )

# real_fix_rms_norm_no_fuse_fp32_matmul_linear_fp32_fwd_dir_path = os.path.join(
#     base_path, "real-fix-rms_norm_no_fuse_fp32-matmul_linear_fp32-fwd/forward"
# )
# real_fix_rms_norm_no_fuse_fp32_matmul_linear_fp32_bwd_dir_path = "I DON'T EXIST"
# compare_fwd_bwd(
#     expected_fwd_dir_path,
#     real_fix_rms_norm_no_fuse_fp32_matmul_linear_fp32_fwd_dir_path,
#     expected_bwd_dir_path,
#     real_fix_rms_norm_no_fuse_fp32_matmul_linear_fp32_bwd_dir_path,
#     abs_err_atol,
#     allclose_atol,
#     allclose_rtol,
# )

# real_no_fix_rms_norm_no_fuse_fp32_matmul_linear_fp32_fwd_dir_path = os.path.join(
#     base_path, "real-no_fix-rms_norm_no_fuse_fp32-matmul_linear_fp32-fwd/forward"
# )
# real_no_fix_rms_norm_no_fuse_fp32_matmul_linear_fp32_bwd_dir_path = "I DON'T EXIST"
# compare_fwd_bwd(
#     expected_fwd_dir_path,
#     real_no_fix_rms_norm_no_fuse_fp32_matmul_linear_fp32_fwd_dir_path,
#     expected_bwd_dir_path,
#     real_no_fix_rms_norm_no_fuse_fp32_matmul_linear_fp32_bwd_dir_path,
#     abs_err_atol,
#     allclose_atol,
#     allclose_rtol,
# )

# real_only_fix_to_parallel_transformer_rms_norm_no_fuse_fp32_matmul_linear_fp32_fwd_dir_path = os.path.join(
#     base_path, "real-only_fix_to_parallel_transformer-rms_norm_no_fuse_fp32-matmul_linear_fp32-fwd/forward"
# )
# real_only_fix_to_parallel_transformer_norm_rms_norm_no_fuse_fp32_matmul_linear_fp32_bwd_dir_path = "I DON'T EXIST"
# compare_fwd_bwd(
#     expected_fwd_dir_path,
#     real_only_fix_to_parallel_transformer_rms_norm_no_fuse_fp32_matmul_linear_fp32_fwd_dir_path,
#     expected_bwd_dir_path,
#     real_only_fix_to_parallel_transformer_norm_rms_norm_no_fuse_fp32_matmul_linear_fp32_bwd_dir_path,
#     abs_err_atol,
#     allclose_atol,
#     allclose_rtol,
# )

# real_only_fix_to_final_rms_norm_rms_norm_no_fuse_fp32_matmul_linear_fp32_fwd_dir_path = os.path.join(
#     base_path, "real-only_fix_to_final_rms_norm-rms_norm_no_fuse_fp32-matmul_linear_fp32-fwd/forward"
# )
# real_only_fix_to_final_rms_norm_rms_norm_no_fuse_fp32_matmul_linear_fp32_bwd_dir_path = "I DON'T EXIST"
# compare_fwd_bwd(
#     expected_fwd_dir_path,
#     real_only_fix_to_final_rms_norm_rms_norm_no_fuse_fp32_matmul_linear_fp32_fwd_dir_path,
#     expected_bwd_dir_path,
#     real_only_fix_to_final_rms_norm_rms_norm_no_fuse_fp32_matmul_linear_fp32_fwd_dir_path,
#     abs_err_atol,
#     allclose_atol,
#     allclose_rtol,
# )

# real_no_fix_cross_entropy_modellink_fwd_dir_path = os.path.join(
#     base_path, "real-no_fix-cross_entropy_modellink/forward"
# )
# real_no_fix_cross_entropy_modellink_bwd_dir_path = os.path.join(
#     base_path, "real-no_fix-cross_entropy_modellink/backward"
# )
# compare_fwd_bwd(
#     expected_fwd_dir_path,
#     real_no_fix_cross_entropy_modellink_fwd_dir_path,
#     expected_bwd_dir_path,
#     real_no_fix_cross_entropy_modellink_bwd_dir_path,
#     abs_err_atol,
#     allclose_atol,
#     allclose_rtol,
# )

real_no_fix_cross_entropy_modellink_cpu_fwd_dir_path = os.path.join(
    base_path, "real-no_fix-cross_entropy_modellink/forward"
)
real_no_fix_cross_entropy_modellink_cpu_bwd_dir_path = os.path.join(
    base_path, "real-no_fix-cross_entropy_modellink/backward"
)

real_fwd_dir = os.path.join(base_path, "real_rms_fp32/forward")
real_bwd_dir = os.path.join(base_path, "real_rms_fp32/backward")

# real_fwd_dir = os.path.join(base_path, "real/forward")
# real_bwd_dir = os.path.join(base_path, "real/backward")

compare_fwd_bwd(
    expected_fwd_dir_path,
    real_fwd_dir,
    expected_bwd_dir_path,
    real_bwd_dir,
    abs_err_atol,
    allclose_atol,
    allclose_rtol,
)
