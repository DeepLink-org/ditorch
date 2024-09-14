from prettytable import PrettyTable


def dict_data_list_to_table(data_dict_list):
    table = PrettyTable()
    keys = list()
    for data_dict in data_dict_list:
        if isinstance(data_dict, dict):
            for key in data_dict.keys():
                if key not in keys:
                    keys.append(key)
        else:
            assert False, "data_dict should be dict"
    table.field_names = keys
    for data_dict in data_dict_list:
        table.add_row([data_dict.get(key, "") for key in keys])
    return table


def packect_data_to_dict_list(op_name, inputs_dict, prefix):
    data_dict_list = []
    args = inputs_dict.get("args", [])
    kwargs = inputs_dict.get("kwargs", {})
    arg_index = -1
    for arg in args:
        arg_index += 1
        if isinstance(arg, dict):
            item_name = op_name + f" {prefix}" + (f"[{arg_index}]" if len(args) > 1 else "")
            data_dict = {"name": item_name}
            data_dict.update(arg)
            data_dict_list.append(data_dict)
        elif isinstance(arg, (tuple, list)):
            arg_sub_index = -1
            for item in arg:
                arg_sub_index += 1
                item_name = op_name + f" {prefix}[{arg_index}]" + f"[{arg_sub_index}]"
                if isinstance(item, dict):
                    data_dict = {"name": item_name}
                    data_dict.update(item)
                    data_dict_list.append(data_dict)
                else:
                    data_dict_list.append({"name": item_name, "value": item})
    for key, value in kwargs.items():
        data_dict_list.append({"name": op_name + f" [{key}]", "value": value})

    return data_dict_list


def pretty_print_op_args(op_name, inputs_dict, outputs_dict=None):

    input_data_dict_list = packect_data_to_dict_list(op_name, inputs_dict, "inputs")
    output_data_dict_list = packect_data_to_dict_list(op_name, outputs_dict, "outputs")
    data_dict_list = input_data_dict_list + output_data_dict_list
    table = dict_data_list_to_table(data_dict_list)
    if len(data_dict_list) > 0:
        print(table)
    return table
