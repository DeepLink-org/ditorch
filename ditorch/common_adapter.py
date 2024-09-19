import torch
import torch.testing._internal.common_utils as common_utils


class CudaNonDefaultStream:
    def __enter__(self):
        # Before starting CUDA test save currently active streams on all
        # CUDA devices and set new non default streams to all CUDA devices
        # to ensure CUDA tests do not use default stream by mistake.
        beforeDevice = torch.cuda.current_device()
        self.beforeStreams = []
        for d in range(torch.cuda.device_count()):
            self.beforeStreams.append(torch.cuda.current_stream(d))
            deviceStream = torch.cuda.Stream(device=d)
            self.beforeStreams[-1].synchronize()
            """
            torch._C._cuda_setStream(stream_id=deviceStream.stream_id,
                                     device_index=deviceStream.device_index,
                                     device_type=deviceStream.device_type)
            """
            torch.cuda.set_stream(deviceStream)

        # torch._C._cuda_setDevice(beforeDevice)
        torch.cuda.set_device(beforeDevice)

    def __exit__(self, exec_type, exec_value, traceback):
        # After completing CUDA test load previously active streams on all
        # CUDA devices.
        beforeDevice = torch.cuda.current_device()
        for d in range(torch.cuda.device_count()):
            """
            torch._C._cuda_setStream(stream_id=self.beforeStreams[d].stream_id,
                                     device_index=self.beforeStreams[d].device_index,
                                     device_type=self.beforeStreams[d].device_type)
            """
            torch.cuda.set_stream(self.beforeStreams[d])
        # torch._C._cuda_setDevice(beforeDevice)
        torch.cuda.set_device(beforeDevice)


common_utils.CudaNonDefaultStream = CudaNonDefaultStream
