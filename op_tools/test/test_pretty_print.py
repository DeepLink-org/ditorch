from op_tools.save_op_args import serialize_args_to_dict
from op_tools.pretty_print import dict_data_list_to_table, packect_data_to_dict_list
import torch
import ditorch

import unittest


class TestPrettyPrint(unittest.TestCase):
    def test_pretty_print(self):
        x = torch.randn(3, 4, device="cuda")
        y = torch.randn(3, 4, 7, 8, device="cpu")

        data_list1 = packect_data_to_dict_list("torch.add", serialize_args_to_dict(x, x))
        data_list2 = packect_data_to_dict_list("torch.stack", serialize_args_to_dict([y, y, y], dim=1))

        data_list = data_list1 + data_list2

        self.assertTrue(len(data_list) == 6)

        table = dict_data_list_to_table(data_list)

        csv_str = table.get_csv_string()

        print(csv_str)

        print(table)


if __name__ == "__main__":
    unittest.main()
