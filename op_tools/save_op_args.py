import torch
import os
from datetime import datetime
import json


def serialize_args_to_dict(*args, **kwargs):
    def tensor_to_dict(tensor):
        return {
            "shape": tensor.shape,
            "stride": tensor.stride(),
            "numel": tensor.numel(),
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
            "requires_grad": tensor.requires_grad,
            "layout": str(tensor.layout),
            "data": tensor.data_ptr(),
        }

    def serialize_value(value):
        if isinstance(value, torch.Tensor):
            return tensor_to_dict(value)
        if isinstance(value, (torch.device, torch.dtype)):
            return str(value)
        elif isinstance(value, (list, tuple)):
            return type(value)([serialize_value(v) for v in value])
        elif isinstance(value, dict):
            return {k: serialize_value(v) for k, v in value.items()}
        else:
            return value

    data = {
        "args": type(args)([serialize_value(arg) for arg in args]),
        "kwargs": {k: serialize_value(v) for k, v in kwargs.items()},
    }
    return data


def save_op_args(name, identifier, *args, **kwargs):
    obj = dict()
    obj["name"] = name
    obj["args"] = args
    obj["kwargs"] = kwargs
    obj["identifier"] = identifier
    filename = f"op_capture_result/{torch.cuda.current_device()}/{datetime.now().strftime('%Y-%m-%d--%H-%M')}/{name}/{identifier}.pth"
    path = filename[: filename.rfind("/")]
    os.makedirs(path, exist_ok=True)
    try:
        torch.save(obj, filename)
        print(f"{filename} saved")

        json_content = serialize_args_to_dict(*args, **kwargs)
        json_content["op"] = name

        json_filename = filename + ".json"
        with open(json_filename, "w") as file:
            json.dump(json_content, file, indent=4)

        print(f"{json_filename} saved")
    except Exception as e:
        print(f"{args} {kwargs} can not save for {name} because {e}")
