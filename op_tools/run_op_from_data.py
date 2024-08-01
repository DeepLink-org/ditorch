import importlib
import torch


def get_function_from_string(func_str):
    """
    Convert a function string like 'torch.Tensor.sum' to an actual function.

    Args:
        func_str (str): The function string to convert.

    Returns:
        function: The actual function object.
    """
    parts = func_str.split(".")
    module_name = ".".join(parts[:-1])
    function_name = parts[-1]

    module = importlib.import_module(module_name)

    func = module
    for part in parts[1:-1]:
        func = getattr(func, part)

    func = getattr(func, function_name)

    return func


# Example usage
# func_str = 'torch.Tensor.sum'
func_str = "torch.sum"
sum_func = get_function_from_string(func_str)

# Create a tensor and call the function
x = torch.tensor([[1, 2, 3], [4, 5, 6]])
result = sum_func(x)
print(result)  # Output: tensor(21)
