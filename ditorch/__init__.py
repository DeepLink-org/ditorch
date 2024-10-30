# Copyright (c) 2024, DeepLink.
import os
adapter = None
try:
    import ditorch.torch_npu_adapter as adapter  # noqa: F811
except ImportError as e:  # noqa: F841
    pass


try:
    import ditorch.torch_dipu_adapter as adapter  # noqa: F811

except ImportError as e:  # noqa: F841
    pass

try:
    import ditorch.torch_mlu_adapter as adapter  # noqa: F811
except ImportError as e:  # noqa: F841
    pass

try:
    import ditorch.torch_biren_adapter as adapter  # noqa: F811
except ImportError as e:  # noqa: F841
    pass


from ditorch import common_adapter  # noqa: F401,E402

if adapter is not None and int(os.getenv("DITORCH_DISABLE_MOCK", "0")) <= 0:
    adapter.mock()
    common_adapter.mock_common()

    print(f"ditorch: {adapter.arch} {adapter.framework.__name__}:{adapter.framework.__version__} pid: {os.getpid()}")
