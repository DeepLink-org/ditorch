import ditorch
import torch
import unittest


class TestCurrentStream(unittest.TestCase):
    def test_current_stream(self):
        stream = torch.cuda.current_stream()
        self.assertIsInstance(stream, torch.cuda.Stream)

    def test_current_stream_device(self):
        for device in range(torch.cuda.device_count()):
            stream = torch.cuda.current_stream(device)
            self.assertIsInstance(stream, torch.cuda.Stream)


if __name__ == "__main__":
    unittest.main()
