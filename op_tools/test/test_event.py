import torch
import ditorch
import unittest


class TestEvent(unittest.TestCase):

    def test_event_measure_device_time(self):
        x = torch.randn(3, 4).cuda()

        start_event = torch.cuda.Event(
            enable_timing=True, blocking=False, interprocess=False
        )
        end_event = torch.cuda.Event(
            enable_timing=True, blocking=False, interprocess=False
        )

        start_event.record(torch.cuda.current_stream())

        y = x + x

        end_event.record(torch.cuda.current_stream())
        device_time = start_event.elapsed_time(end_event)
        print(f"device_time:{device_time} milliseconds")
        self.assertEqual(device_time > 0.0, True)


if __name__ == "__main__":
    unittest.main()
