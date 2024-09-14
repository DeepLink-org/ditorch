# Owner(s): ["module: inductor"]

import torch
from torch._dynamo.test_case import run_tests, TestCase
from torch._dynamo.utils import counters
from torch.testing._internal.common_utils import IS_LINUX
from torch.testing._internal.inductor_utils import HAS_CUDA


def patch(f):
    f = torch._inductor.config.patch(split_cat_fx_passes=True)(f)
    return f


class TestSplitCatFxPasses(TestCase):
    @patch
    def test_split_normalization(self):
        def arg_only(x):
            return [torch.relu(s) for s in torch.split(x, 2, 1)]

        def arg_only_dim0(x):
            return [torch.relu(s) for s in torch.split(x, 2, 0)]

        def kwarg1(x):
            return [torch.relu(s) for s in torch.split(x, 2, dim=1)]

        def kwarg2(x):
            return [
                torch.relu(s) for s in torch.split(x, split_size_or_sections=2, dim=1)
            ]

        def kwarg3(x):
            return [
                torch.relu(s)
                for s in torch.split(tensor=x, split_size_or_sections=2, dim=-1)
            ]

        def list_replace(x):
            return [torch.relu(s) for s in torch.split(x, [16, 16], dim=1)]

        def multi_split(x):
            return [torch.split(s, 2, 1) for s in torch.split(x, 2, 1)]

        def unequal_split(x):
            return [torch.relu(s) for s in torch.split(x, 3, 1)]

        def arg_only_cm(x):
            return [torch.relu(s) for s in x.split(2, 1)]

        def kwarg1_cm(x):
            return [torch.relu(s) for s in x.split(2, dim=1)]

        def kwarg2_cm(x):
            return [torch.relu(s) for s in x.split(split_size=2, dim=1)]

        def multi_split_cm(x):
            return [s.split(2, 1) for s in x.split(2, 1)]

        def unequal_split_cm(x):
            return [torch.relu(s) for s in x.split(3, 1)]

        def cm_with_list(x):
            return [torch.relu(s) for s in x.split([16, 16], dim=-1)]

        args = [
            torch.randn(2, 32),
        ]
        for fn, expected_split_norm_count in [
            (arg_only, 1),
            (arg_only_dim0, 1),
            (kwarg1, 1),
            (kwarg2, 1),
            (kwarg3, 1),
            (list_replace, 1),
            (multi_split, 17),
            (unequal_split, 1),
            (arg_only_cm, 1),
            (kwarg1_cm, 1),
            (kwarg2_cm, 1),
            (multi_split_cm, 17),
            (unequal_split_cm, 1),
            (cm_with_list, 1),
        ]:
            expected = fn(*args)
            actual = torch.compile(fn)(*args)

            torch.testing.assert_close(actual, expected)
            self.assertEqual(
                counters["inductor"]["split_cat_norm"],
                expected_split_norm_count,
            )
            counters.clear()

    @patch
    def test_consecutive_split_merge(self):
        def multi_split(x):
            return [torch.split(s, 2, 1) for s in torch.split(x, 2, 1)]

        def multi_split_2(x):
            return [torch.split(s, 1, 1) for s in torch.split(x, 2, 1)]

        def multi_split_2_neg_dim(x):
            return [torch.split(s, 1, 1) for s in torch.split(x, 2, -1)]

        def multi_split_with_sizes(x):
            return [torch.split(s, 2, 1) for s in torch.split(x, [16, 16], 1)]

        def multi_split_kwarg1(x):
            return [torch.split(s, 2, dim=1) for s in torch.split(x, 2, dim=1)]

        def multi_split_kwarg2(x):
            return [
                torch.split(s, split_size_or_sections=2, dim=1)
                for s in torch.split(x, split_size_or_sections=2, dim=1)
            ]

        def unequal_multi_split(x):
            fs = torch.split(x, [10, 10, 12], dim=1)
            item0 = fs[0]
            item1 = fs[1]
            item2 = fs[2]

            final_items = []
            final_items.extend(item0.split([4, 6], 1))
            final_items.extend(item1.split([6, 4], 1))
            final_items.extend(item2.split([4, 4, 4], 1))

            return [torch.relu(s) for s in final_items]

        def unequal_multi_split_neg_index(x):
            fs = torch.split(x, [10, 10, 12], dim=1)
            item0 = fs[-3]
            item1 = fs[-2]
            item2 = fs[-1]

            final_items = []
            final_items.extend(item0.split([4, 6], 1))
            final_items.extend(item1.split([6, 4], 1))
            final_items.extend(item2.split([4, 4, 4], 1))

            return [torch.relu(s) for s in final_items]

        # Shouldn't merge
        def diff_dims(x):
            return [torch.split(s, 2, dim=0) for s in torch.split(x, 2, dim=1)]

        def some_users_not_splits(x):
            fs = torch.split(x, [10, 10, 12], dim=1)
            item0 = fs[0]
            item1 = fs[1]
            item2 = fs[2]

            final_items = []
            final_items.extend(item0.split([4, 6], 1))
            final_items.extend(item1.split([6, 4], 1))
            final_items.append(torch.sin(item2))

            return [torch.relu(s) for s in final_items]

        def split_with_cat(x):
            fs = torch.split(x, [4, 4, 24], dim=1)
            item0 = fs[0]
            item1 = fs[1]
            item2 = fs[2]

            final_items = [item0, item1]
            final_items.extend(item2.split((4, 4, 4, 4, 4, 4), 1))

            return torch.cat(final_items, dim=1)

        def duplicate_getitems(x):
            fs = torch.split(x, [10, 10, 12], dim=1)
            item0 = fs[0]
            item1_1 = fs[1]
            item1_2 = fs[1]
            item2 = fs[2]

            final_items = []
            final_items.extend(item0.split([4, 6], 1))
            final_items.extend(item1_1.split([6, 4], 1))
            final_items.extend(item1_2)
            final_items.append(torch.sin(item2))

            return [torch.relu(s) for s in final_items]

        def duplicate_getitems_neg_index(x):
            fs = torch.split(x, [10, 10, 12], dim=1)
            item0 = fs[0]
            item1_1 = fs[1]
            item1_2 = fs[-2]  # negative index
            item2 = fs[2]

            final_items = []
            final_items.extend(item0.split([4, 6], 1))
            final_items.extend(item1_1.split([6, 4], 1))
            final_items.extend(item1_2)
            final_items.append(torch.sin(item2))

            return [torch.relu(s) for s in final_items]

        def split_getitem_gap(x):
            fs = torch.split(x, [4, 4, 24], dim=1)
            item0 = fs[0]
            item2 = fs[2]

            final_items = [
                item0,
            ]
            final_items.extend(item2.split((4, 4, 4, 4, 4, 4), 1))

            return torch.cat(final_items, dim=1)

        def split_getitem_out_of_order(x):
            fs = torch.split(x, [4, 4, 4, 20], dim=1)
            item0 = fs[0]
            item2 = fs[2]
            item1 = fs[1]
            item3 = fs[3]

            final_items = [item0, item2, item1]
            final_items.extend(item3.split((4, 4, 4, 4, 4), 1))

            return torch.cat(final_items, dim=1)

        args = [
            torch.randn(2, 32),
        ]
        for fn, expected_split_merged in [
            (multi_split, 16),
            (multi_split_2, 16),
            (multi_split_2_neg_dim, 16),
            (multi_split_with_sizes, 2),
            (multi_split_kwarg1, 16),
            (multi_split_kwarg2, 16),
            (unequal_multi_split, 3),
            (unequal_multi_split_neg_index, 3),
            (diff_dims, 0),
            (some_users_not_splits, 2),
            (split_with_cat, 1),
            (duplicate_getitems, 1),
            (duplicate_getitems_neg_index, 1),
            (split_getitem_gap, 1),
            (split_getitem_out_of_order, 1),
        ]:
            expected = fn(*args)
            actual = torch.compile(fn)(*args)

            torch.testing.assert_close(actual, expected)
            self.assertEqual(
                counters["inductor"]["consecutive_split_merged"],
                expected_split_merged,
            )
            counters.clear()

    @patch
    def test_split_cat_merge(self):
        def simple_split_cat(x):
            return torch.cat(torch.split(x, 4, dim=1), dim=1)

        def simple_split_cat_argspec1(x):
            return torch.cat(torch.split(x, 4, dim=1), 1)

        def simple_split_cat_argspec2(x):
            return torch.cat(tensors=torch.split(x, 4, dim=1), dim=1)

        def simple_split_cat_argspec3(x):
            return torch.cat(torch.split(x, 4, dim=1), -2)

        def simple_split_cat_argspec4(x):
            return torch.cat(tensors=torch.split(x, 4, dim=1), dim=-2)

        def simple_split_stack(x):
            return torch.stack(torch.split(x, 4, dim=1), dim=1)

        def simple_split_stack_argspec1(x):
            return torch.stack(torch.split(x, 4, dim=1), 1)

        def simple_split_stack_argspec2(x):
            return torch.stack(tensors=torch.split(x, 4, dim=1), dim=1)

        def split_cat_addn_args(x):
            split_output = list(torch.split(x, 4, dim=1))
            return torch.cat(
                [torch.ones(2, 5, 32, 16)] + split_output + [torch.ones(2, 6, 32, 16)],
                dim=1,
            )

        def split_stack_addn_args(x):
            split_output = list(torch.split(x, 4, dim=1))
            return torch.stack(
                [torch.ones(2, 4, 32, 16)]
                + split_output
                + [torch.ones(2, 4, 32, 16), torch.ones(2, 4, 32, 16)],
                dim=1,
            )

        def split_cat_addn_args_dim2(x):
            split_output = list(torch.split(x, 4, dim=2))
            return torch.cat(
                [torch.ones(2, 32, 5, 16)] + split_output + [torch.ones(2, 32, 6, 16)],
                dim=2,
            )

        # split_dim=1, cat_dim=2
        def split_cat_dim_mismatch(x):
            split_output = list(torch.split(x, 4, dim=1))
            return torch.cat(
                [torch.ones(2, 4, 32, 16)] + split_output + [torch.ones(2, 4, 32, 16)],
                dim=2,
            )

        def split_stack_dim_mismatch(x):
            split_output = list(torch.split(x, 4, dim=1))
            return torch.stack(
                [torch.ones(2, 4, 32, 16)] + split_output + [torch.ones(2, 4, 32, 16)],
                dim=2,
            )

        # split_dim=1, cat_dim=3
        def split_cat_dim_mismatch2(x):
            split_output = list(torch.split(x, 4, dim=1))
            return torch.cat(
                [torch.ones(2, 4, 32, 16)] + split_output + [torch.ones(2, 4, 32, 16)],
                dim=3,
            )

        def split_stack_dim_mismatch2(x):
            split_output = list(torch.split(x, 4, dim=1))
            return torch.stack(
                [torch.ones(2, 4, 32, 16)] + split_output + [torch.ones(2, 4, 32, 16)],
                dim=3,
            )

        # split_dim=2, cat_dim=0
        def split_cat_dim_mismatch3(x):
            split_output = list(torch.split(x, 4, dim=2))
            return torch.cat(
                [torch.ones(2, 32, 4, 16)] + split_output + [torch.ones(2, 32, 4, 16)],
                dim=0,
            )

        def split_stack_dim_mismatch3(x):
            split_output = list(torch.split(x, 4, dim=2))
            return torch.stack(
                [torch.ones(2, 32, 4, 16)] + split_output + [torch.ones(2, 32, 4, 16)],
                dim=0,
            )

        def input_shuffling(x):
            split_output = list(torch.split(x, 4, dim=1))
            return torch.cat(
                [torch.ones(2, 4, 32, 16)]
                + [split_output[1], split_output[2], split_output[3]]
                + [torch.ones(2, 4, 32, 16)]
                + [split_output[5], split_output[6], split_output[7]]
                + [torch.ones(2, 4, 32, 16)],
                dim=1,
            )

        def input_shuffling_stack(x):
            split_output = list(torch.split(x, 4, dim=1))
            return torch.stack(
                [torch.ones(2, 4, 32, 16)]
                + [split_output[1], split_output[2], split_output[3]]
                + [torch.ones(2, 4, 32, 16)]
                + [split_output[5], split_output[6], split_output[7]]
                + [torch.ones(2, 4, 32, 16)],
                dim=1,
            )

        def input_shuffling_dim_mismatch(x):
            split_output = list(torch.split(x, 4, dim=1))
            return torch.cat(
                [torch.ones(2, 4, 32, 16)]
                + [split_output[1], split_output[2], split_output[3]]
                + [torch.ones(2, 4, 32, 16)]
                + [split_output[5], split_output[6], split_output[7]]
                + [torch.ones(2, 4, 32, 16)],
                dim=2,
            )

        def input_shuffling_dim_mismatch_stack(x):
            split_output = list(torch.split(x, 4, dim=1))
            return torch.stack(
                [torch.ones(2, 4, 32, 16)]
                + [split_output[1], split_output[2], split_output[3]]
                + [torch.ones(2, 4, 32, 16)]
                + [split_output[5], split_output[6], split_output[7]]
                + [torch.ones(2, 4, 32, 16)],
                dim=2,
            )

        def input_shuffling_multiple_output(x):
            split_output = list(torch.split(x, 4, dim=1))
            cat1 = torch.cat(
                [torch.ones(2, 4, 32, 16)]
                + [split_output[1], split_output[2], split_output[3]]
                + [torch.ones(2, 4, 32, 16)],
                dim=2,
            )
            stack1 = torch.stack(
                [
                    torch.ones(2, 4, 32, 16),
                    split_output[4],
                    split_output[5],
                    torch.ones(2, 4, 32, 16),
                ],
                dim=1,
            )

            relu1 = torch.relu(split_output[6])

            return cat1, stack1, relu1

        def input_shuffling_direct_output(x):
            split_output = list(torch.split(x, 4, dim=1))
            cat1 = torch.cat(
                [torch.ones(2, 4, 32, 16)]
                + [split_output[1], split_output[2], split_output[3]]
                + [torch.ones(2, 4, 32, 16)],
                dim=2,
            )
            stack1 = torch.stack(
                [
                    torch.ones(2, 4, 32, 16),
                    split_output[4],
                    split_output[5],
                    torch.ones(2, 4, 32, 16),
                ],
                dim=1,
            )

            return cat1, stack1, split_output[6]

        def input_shuffling_multiple_output_same_ranges(x):
            split_output = list(torch.split(x, 4, dim=1))
            cat1 = torch.cat(
                [torch.ones(2, 4, 32, 16)]
                + [split_output[1], split_output[2], split_output[3]]
                + [torch.ones(2, 4, 32, 16)],
                dim=2,
            )

            cat2 = torch.cat(
                [torch.ones(2, 4, 32, 16)]
                + [split_output[1], split_output[2], split_output[3]]
                + [torch.ones(2, 4, 32, 16)],
                dim=2,
            )
            stack1 = torch.stack(
                [
                    torch.ones(2, 4, 32, 16),
                    split_output[4],
                    split_output[5],
                    torch.ones(2, 4, 32, 16),
                ],
                dim=1,
            )

            relu1 = torch.relu(split_output[6])

            return cat1, cat2, stack1, relu1

        def unequal_split_multiple_output(x):
            split_output = list(torch.split(x, [2, 4, 4, 4, 4, 4, 8, 2], dim=1))
            cat1 = torch.cat(
                [torch.ones(2, 4, 32, 16)]
                + [split_output[1], split_output[2], split_output[3]]
                + [torch.ones(2, 4, 32, 16)],
                dim=2,
            )
            stack1 = torch.stack(
                [
                    torch.ones(2, 4, 32, 16),
                    split_output[4],
                    split_output[5],
                    torch.ones(2, 4, 32, 16),
                ],
                dim=1,
            )

            relu1 = torch.relu(split_output[6])

            return cat1, stack1, relu1

        def multi_split_cat(x1, x2):
            split_output_1 = list(torch.split(x1, 4, dim=1))
            split_output_2 = list(torch.split(x2, 4, dim=1))
            cat1 = torch.cat(
                [torch.ones(2, 4, 32, 16)]
                + [split_output_1[1], split_output_1[2], split_output_1[3]]
                + [torch.ones(2, 4, 32, 16)]
                + [split_output_2[1], split_output_2[2], split_output_2[3]]
                + [torch.ones(2, 4, 32, 16)],
                dim=2,
            )
            stack1 = torch.stack(
                [
                    torch.ones(2, 4, 32, 16),
                    split_output_1[4],
                    split_output_1[5],
                    torch.ones(2, 4, 32, 16),
                    split_output_2[4],
                    split_output_2[5],
                    torch.ones(2, 4, 32, 16),
                ],
                dim=1,
            )

            relu1 = torch.relu(split_output_1[6])
            relu2 = torch.relu(split_output_2[6])

            return cat1, stack1, relu1, relu2

        # TODO: Add more tests:
        # * Cases where replacement shouldn't happen
        default_args = [
            torch.randn(2, 32, 32, 16),
        ]
        multi_args = [
            torch.randn(2, 32, 32, 16),
            torch.randn(2, 32, 32, 16),
        ]
        for (
            fn,
            expected_split_added,
            expected_split_removed,
            expected_cat_added,
            expected_cat_removed,
            expected_sections_removed,
            args,
        ) in [
            (simple_split_cat, 0, 1, 0, 1, 7, default_args),
            (simple_split_cat_argspec1, 0, 1, 0, 1, 7, default_args),
            (simple_split_cat_argspec2, 0, 1, 0, 1, 7, default_args),
            (simple_split_cat_argspec3, 0, 1, 0, 1, 7, default_args),
            (simple_split_cat_argspec4, 0, 1, 0, 1, 7, default_args),
            (simple_split_stack, 0, 1, 0, 1, 7, default_args),
            (simple_split_stack_argspec1, 0, 1, 0, 1, 7, default_args),
            (simple_split_stack_argspec2, 0, 1, 0, 1, 7, default_args),
            (split_cat_addn_args, 0, 1, 1, 1, 7, default_args),
            (split_stack_addn_args, 0, 1, 1, 1, 7, default_args),
            (split_cat_addn_args_dim2, 0, 1, 1, 1, 7, default_args),
            (split_cat_dim_mismatch, 0, 1, 1, 1, 7, default_args),
            (split_stack_dim_mismatch, 0, 1, 1, 1, 7, default_args),
            (split_cat_dim_mismatch2, 0, 1, 1, 1, 7, default_args),
            (split_stack_dim_mismatch2, 0, 1, 1, 1, 7, default_args),
            (split_cat_dim_mismatch3, 0, 1, 1, 1, 7, default_args),
            (split_stack_dim_mismatch3, 0, 1, 1, 1, 7, default_args),
            (input_shuffling, 1, 1, 1, 1, 4, default_args),
            (input_shuffling_stack, 1, 1, 1, 1, 4, default_args),
            (input_shuffling_dim_mismatch, 1, 1, 1, 1, 4, default_args),
            (input_shuffling_dim_mismatch_stack, 1, 1, 1, 1, 4, default_args),
            (input_shuffling_multiple_output, 1, 1, 2, 2, 3, default_args),
            (input_shuffling_direct_output, 1, 1, 2, 2, 3, default_args),
            (unequal_split_multiple_output, 1, 1, 2, 2, 3, default_args),
            (multi_split_cat, 2, 2, 4, 4, 3, multi_args),
        ]:
            expected = fn(*args)
            actual = torch.compile(fn)(*args)

            torch.testing.assert_close(actual, expected)
            self.assertEqual(
                counters["inductor"]["scmerge_split_added"],
                expected_split_added,
            )
            self.assertEqual(
                counters["inductor"]["scmerge_split_removed"],
                expected_split_removed,
            )
            self.assertEqual(
                counters["inductor"]["scmerge_cat_added"],
                expected_cat_added,
            )
            self.assertEqual(
                counters["inductor"]["scmerge_cat_removed"],
                expected_cat_removed,
            )
            self.assertEqual(
                counters["inductor"]["scmerge_split_sections_removed"],
                expected_sections_removed,
            )
            counters.clear()

    @torch._inductor.config.patch(split_cat_fx_passes=False)
    def test_config_flag_is_respected(self):
        def split_with_cat(x):
            fs = torch.split(x, [4, 4, 24], dim=-1)
            item0 = fs[0]
            item1 = fs[1]
            item2 = fs[2]

            final_items = [item0, item1]
            final_items.extend(item2.split((4, 4, 4, 4, 4, 4), 1))

            return torch.cat(final_items, dim=1)

        args = [
            torch.randn(2, 32),
        ]

        expected = split_with_cat(*args)
        actual = torch.compile(split_with_cat)(*args)

        torch.testing.assert_close(actual, expected)
        self.assertEqual(
            counters["inductor"]["consecutive_split_merged"],
            0,
        )
        self.assertEqual(
            counters["inductor"]["split_cat_norm"],
            0,
        )

    @patch
    def test_split_cat_merge_mutation(self):
        args = [
            torch.randn(2, 32, 32, 16),
        ]

        def split_cat_mutation(x):
            splits = torch.split(x, 4, dim=1)
            splits[1].copy_(splits[0])
            return torch.cat(splits, dim=1)

        expected = split_cat_mutation(*args)
        actual = torch.compile(split_cat_mutation)(*args)

        torch.testing.assert_close(actual, expected)

        self.assertEqual(counters["inductor"]["scmerge_split_removed"], 0)
        self.assertEqual(counters["inductor"]["scmerge_cat_removed"], 0)

    @patch
    def test_split_squeeze(self):
        def split_squeeze_stack(x):
            items = list(torch.split(x, 1, dim=1))
            split_items = [torch.squeeze(s, 1) for s in items]
            return torch.stack(split_items)

        def split_squeeze_stack_callmethod(x):
            items = list(torch.split(x, 1, dim=1))
            split_items = [s.squeeze(1) for s in items]
            return torch.stack(split_items)

        def split_squeeze_stack_callmethod_none_dim(x):
            items = list(torch.split(x, 1, dim=1))
            split_items = [s.squeeze() for s in items]
            return torch.stack(split_items)

        def split_squeeze_stack_kwarg1(x):
            items = list(torch.split(x, 1, dim=1))
            split_items = [torch.squeeze(s, dim=1) for s in items]
            return torch.stack(split_items)

        def split_squeeze_stack_kwarg1_callmethod(x):
            items = list(torch.split(x, 1, dim=1))
            split_items = [s.squeeze(dim=1) for s in items]
            return torch.stack(split_items)

        def split_squeeze_multi_squeeze_users(x):
            items = list(torch.split(x, 1, dim=1))
            split_items = [torch.squeeze(s, 1) for s in items]
            return (
                torch.stack(split_items),
                torch.relu(split_items[0]),
                torch.tanh(split_items[1]),
            )

        def split_size_not_1(x):
            items = list(torch.split(x, 2, dim=1))
            split_items = [torch.squeeze(s, 1) for s in items]
            return torch.stack(split_items)

        def dim_mismatch(x):
            items = list(torch.split(x, 1, dim=1))
            split_items = [torch.squeeze(s, 0) for s in items]
            return torch.stack(split_items)

        def other_users(x):
            items = list(torch.split(x, 1, dim=1))
            split_items = [torch.squeeze(s, 1) for s in items]
            return torch.stack(split_items), torch.relu(items[0])

        def other_users_2(x):
            items = list(torch.split(x, 1, dim=1))
            split_items = [torch.squeeze(s, 1) for s in items[1:]]
            return torch.stack(split_items), torch.relu(items[0])

        def graph_should_be_topological_sorted(x):
            output = []
            for t in x.split(1):
                output.append(torch.sin(t.squeeze(dim=0)))
            output = torch.stack(output)
            return output

        args = [
            torch.randn(2, 32),
        ]
        for fn, split_squeeze_replaced in [
            (split_squeeze_stack, 1),
            (split_squeeze_stack_callmethod, 1),
            # TODO handle none dim
            (split_squeeze_stack_callmethod_none_dim, 0),
            (split_squeeze_stack_kwarg1, 1),
            (split_squeeze_stack_kwarg1_callmethod, 1),
            (split_squeeze_multi_squeeze_users, 1),
            (split_size_not_1, 0),
            (dim_mismatch, 0),
            (other_users, 0),
            (other_users_2, 0),
            (graph_should_be_topological_sorted, 1),
        ]:
            expected = fn(*args)
            actual = torch.compile(fn)(*args)

            torch.testing.assert_close(actual, expected)
            self.assertEqual(
                counters["inductor"]["split_squeeze_replaced"],
                split_squeeze_replaced,
            )
            counters.clear()

    @patch
    def test_unbind_stack(self):
        def unbind_stack(x):
            return torch.stack(torch.unbind(x, dim=1), 1)

        def unbind_cat(x):
            return torch.cat(torch.unbind(x, dim=1), 1)

        def unbind_stack_argspec1(x):
            return torch.stack(torch.unbind(x, dim=1), dim=1)

        def unbind_stack_argspec2(x):
            return torch.stack(tensors=torch.unbind(x, dim=1), dim=1)

        def dim_mismatch(x):
            return torch.stack(torch.unbind(x, dim=1), 0)

        def split_squeeze_stack(x):
            items = list(torch.split(x, 1, dim=1))
            split_items = [torch.squeeze(s, 1) for s in items]
            return torch.stack(split_items, 1)

        def split_squeeze_stack_callmethod(x):
            items = list(torch.split(x, 1, dim=1))
            split_items = [torch.squeeze(s, 1) for s in items]
            return torch.stack(split_items, 1)

        def other_users(x):
            items = list(torch.split(x, 1, dim=1))
            split_items = [torch.squeeze(s, 1) for s in items]
            return torch.stack(split_items, 1), torch.relu(items[0])

        def other_users_2(x):
            items = list(torch.split(x, 1, dim=1))
            split_items = [torch.squeeze(s, 1) for s in items[1:]]
            return torch.stack(split_items, 1), torch.relu(items[0])

        def unbind_cat_addn_args(x):
            split_output = list(torch.unbind(x, dim=1))

            return torch.cat(
                [torch.ones(2, 32, 16)] + split_output + [torch.ones(2, 32, 16)],
                dim=1,
            )

        def unbind_stack_addn_args(x):
            split_output = list(torch.unbind(x, dim=1))
            return torch.stack(
                [torch.ones(2, 32, 16)]
                + split_output
                + [torch.ones(2, 32, 16), torch.ones(2, 32, 16)],
                dim=1,
            )

        def unbind_cat_addn_args_dim2(x):
            split_output = list(torch.unbind(x, dim=2))
            return torch.cat(
                [torch.ones(2, 32, 16)] + split_output + [torch.ones(2, 32, 16)],
                dim=2,
            )

        # split_dim=1, cat_dim=2
        def unbind_cat_dim_mismatch(x):
            split_output = list(torch.unbind(x, dim=1))
            return torch.cat(
                [torch.ones(2, 32, 16)] + split_output + [torch.ones(2, 32, 16)],
                dim=2,
            )

        def unbind_stack_dim_mismatch(x):
            split_output = list(torch.unbind(x, dim=1))
            return torch.stack(
                [torch.ones(2, 32, 16)] + split_output + [torch.ones(2, 32, 16)],
                dim=2,
            )

        def unbind_cat_multi_users(x):
            split_output = list(torch.unbind(x, dim=1))
            return torch.cat(
                [torch.ones(2, 32, 16)] + split_output + [torch.ones(2, 32, 16)],
                dim=1,
            ), torch.stack(
                [torch.ones(2, 32, 16)]
                + split_output
                + [torch.ones(2, 32, 16), torch.ones(2, 32, 16)],
                dim=1,
            )

        def unbind_cat_multi_users_diff_dims(x):
            split_output = list(torch.unbind(x, dim=1))
            return torch.cat(
                [torch.ones(2, 32, 16)] + split_output + [torch.ones(2, 32, 16)],
                dim=1,
            ), torch.stack(
                [torch.ones(2, 32, 16)] + split_output + [torch.ones(2, 32, 16)],
                dim=2,
            )

        args = [
            torch.randn(2, 32, 32, 16),
        ]
        for (
            fn,
            expected_unbind_added,
            expected_unbind_removed,
            expected_cat_added,
            expected_cat_removed,
            expected_sections_removed,
        ) in [
            (unbind_stack, 0, 1, 0, 1, 31),
            (unbind_stack_argspec1, 0, 1, 0, 1, 31),
            (unbind_stack_argspec2, 0, 1, 0, 1, 31),
            (dim_mismatch, 0, 1, 0, 1, 31),
            (split_squeeze_stack, 0, 1, 0, 1, 31),
            (split_squeeze_stack_callmethod, 0, 1, 0, 1, 31),
            (other_users, 0, 0, 0, 0, 0),
            (other_users_2, 0, 0, 0, 0, 0),
            (unbind_cat_addn_args, 0, 1, 1, 1, 31),
            (unbind_stack_addn_args, 0, 1, 1, 1, 31),
            (unbind_cat_addn_args_dim2, 0, 1, 1, 1, 31),
            (unbind_cat_dim_mismatch, 0, 1, 1, 1, 31),
            (unbind_stack_dim_mismatch, 0, 1, 1, 1, 31),
            (unbind_cat_multi_users, 0, 1, 2, 2, 31),
            (unbind_cat_multi_users_diff_dims, 0, 1, 2, 2, 31),
        ]:
            print()
            print(fn)
            expected = fn(*args)
            actual = torch.compile(fn)(*args)

            torch.testing.assert_close(actual, expected)
            self.assertEqual(
                counters["inductor"]["scmerge_split_added"],
                expected_unbind_added,
            )
            self.assertEqual(
                counters["inductor"]["scmerge_split_removed"],
                expected_unbind_removed,
            )
            self.assertEqual(
                counters["inductor"]["scmerge_cat_added"],
                expected_cat_added,
            )
            self.assertEqual(
                counters["inductor"]["scmerge_cat_removed"],
                expected_cat_removed,
            )
            self.assertEqual(
                counters["inductor"]["scmerge_split_sections_removed"],
                expected_sections_removed,
            )
            counters.clear()


if __name__ == "__main__":
    if IS_LINUX and HAS_CUDA:
        run_tests()
