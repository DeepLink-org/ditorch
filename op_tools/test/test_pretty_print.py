from op_tools.save_op_args import serialize_args_to_dict
from op_tools.pretty_print import pretty_print_op_args
from op_tools.utils import compare_result
import torch
import ditorch

x = torch.randn(3, 4, device="cuda")
y = torch.randn(3, 4, 7, 8, device="cpu")

pretty_print_op_args(
    op_name="torch.add",
    inputs_dict=serialize_args_to_dict(x, x, x),
    outputs_dict=serialize_args_to_dict(x),
)
pretty_print_op_args(
    op_name="torch.stack",
    inputs_dict=serialize_args_to_dict([x, x, x], dim=1),
    outputs_dict=serialize_args_to_dict(x),
)
